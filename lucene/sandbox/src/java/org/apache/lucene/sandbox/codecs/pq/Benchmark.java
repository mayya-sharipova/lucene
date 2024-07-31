/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.sandbox.codecs.pq;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** Benchmark for Product Quantization. */
@SuppressForbidden(reason = "System.out required: command line tool")
public class Benchmark {
  private static final int NUM_DIMS = 768;
  private static final int NUM_DOCS = 934_024;
  private static final int NUM_QUERIES = 1_000;
  public static final int FILE_VECTOR_OFFSET = Float.BYTES;
  public static final int FILE_BYTESIZE = NUM_DIMS * Float.BYTES + FILE_VECTOR_OFFSET;
  private static final Path dirPath = Paths.get("/Users/mayya/Elastic/knn/ann-prototypes/data");
  private static final String vectorFile = "corpus-wiki-cohere.fvec";
  private static final String queryFile = "queries-wiki-cohere.fvec";
  private static String groundTruthFile = "queries-wiki-cohere-ground-truth.fvec";
  private static VectorSimilarityFunction vectorFunction = VectorSimilarityFunction.COSINE;
  private static int[] numBooks =
      new int[] {NUM_DIMS * 4 / 32, NUM_DIMS * 4 / 64, NUM_DIMS * 4 / 96, NUM_DIMS * 4 / 128};
  private static float anisotropicThreshold = 0.0f;

  private static boolean useHnsw = true;
  private static final int[] topKs = new int[] {10};
  private static final int[] rerankFactors = new int[] {1, 2, 4, 8, 10};
  private static long seed = 42L;

  public static void main(String[] args) throws Exception {
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      switch (arg) {
        case "-seed":
          seed = Long.parseLong(args[++i]);
          break;
        case "-metric":
          String metric = args[++i];
          switch (metric) {
            case "cosine":
              vectorFunction = VectorSimilarityFunction.COSINE;
              break;
            case "l2":
              vectorFunction = VectorSimilarityFunction.EUCLIDEAN;
              break;
            case "ip":
              vectorFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
              break;
            default:
              usage();
              throw new IllegalArgumentException("-metric can be 'cosine', 'l2' or 'ip' only");
          }
          break;
        case "-numBooks":
          numBooks = new int[] {Integer.parseInt(args[++i])};
          break;
        case "-anisotropic_threshold":
          anisotropicThreshold = Float.parseFloat(args[++i]);
          break;
        case "-useHnsw":
          useHnsw = Boolean.parseBoolean(args[++i]);
          break;
        case "-groundTruthFile":
          groundTruthFile = args[++i];
          break;
        default:
          usage();
          throw new IllegalArgumentException("unknown argument " + arg);
      }
    }
    new Benchmark().runBenchmark();
  }

  private static void usage() {
    String error = "Usage: Benchmark [-metric N] [-seed N] [-useHnsw B]";
    System.err.println(error);
  }

  private void runBenchmark() throws Exception {
    for (int nbooks : numBooks) {
      final ProductQuantizer pq;
      try (MMapDirectory directory = new MMapDirectory(dirPath);
          IndexInput vectorInput = directory.openInput(vectorFile, IOContext.DEFAULT);
          IndexInput queryInput = directory.openInput(queryFile, IOContext.READONCE); ) {
        RandomAccessVectorValues.Floats vectorValues =
            new VectorsReaderWithOffset(
                vectorInput, NUM_DOCS, NUM_DIMS, FILE_BYTESIZE, FILE_VECTOR_OFFSET);
        RandomAccessVectorValues.Floats queryVectorValues =
            new VectorsReaderWithOffset(
                queryInput, NUM_QUERIES, NUM_DIMS, FILE_BYTESIZE, FILE_VECTOR_OFFSET);

        pq =
            ProductQuantizer.create(
                vectorValues, nbooks, vectorFunction, seed, useHnsw, anisotropicThreshold);

        int[][] groundTruths = new int[NUM_QUERIES][];
        Path path = directory.getDirectory().resolve(groundTruthFile);
        if (Files.exists(path)) {
          // reading the ground truths from the file
          try (IndexInput queryGroudTruthInput =
              directory.openInput(groundTruthFile, IOContext.DEFAULT)) {
            for (int i = 0; i < NUM_QUERIES; i++) {
              int length = queryGroudTruthInput.readInt();
              groundTruths[i] = new int[length];
              for (int j = 0; j < length; j++) {
                groundTruths[i][j] = queryGroudTruthInput.readInt();
              }
            }
          }
        } else {
          // writing to the ground truth file
          try (IndexOutput queryGroudTruthOutput =
              directory.createOutput(groundTruthFile, IOContext.DEFAULT)) {
            for (int i = 0; i < NUM_QUERIES; i++) {
              float[] candidate = queryVectorValues.vectorValue(i);
              groundTruths[i] = getNN(vectorValues, candidate, topKs[topKs.length - 1]);
              queryGroudTruthOutput.writeInt(groundTruths[i].length);
              for (int doc : groundTruths[i]) {
                queryGroudTruthOutput.writeInt(doc);
              }
            }
          }
        }
        float[] recalls = new float[topKs.length * rerankFactors.length];
        long[] elapsedCodes = new long[topKs.length * rerankFactors.length];
        int row = 0;
        for (int topK : topKs) {
          for (int rerankFactor : rerankFactors) {
            int totalMatches = 0;
            int totalResults = 0;
            long elapsedCodeCmp = 0;
            for (int i = 0; i < NUM_QUERIES; i++) {
              float[] candidate = queryVectorValues.vectorValue(i);
              long startCodeCmp = System.nanoTime();
              int[] results = pq.getTopDocs(candidate, topK * rerankFactor);
              elapsedCodeCmp += System.nanoTime() - startCodeCmp;
              totalMatches += compareNN(groundTruths[i], results, topK);
              totalResults += topK;
            }
            float recall = totalMatches / (float) totalResults;
            elapsedCodes[row] = TimeUnit.NANOSECONDS.toMillis(elapsedCodeCmp);
            recalls[row++] = recall;
          }
        }
        System.out.println("Recall:");
        System.out.print("[PQ");
        for (int topK : topKs) {
          for (int rerankFactor : rerankFactors) {
            System.out.print(", " + topK + "|" + topK * rerankFactor);
          }
        }
        System.out.println("]");
        System.out.print("['" + nbooks + "'");
        for (float recall : recalls) {
          System.out.print(", " + recall);
        }
        System.out.println("]");

        System.out.println("Performance (avg per query in ms):");
        System.out.print("[PQ");
        for (int topK : topKs) {
          for (int rerankFactor : rerankFactors) {
            System.out.print(", " + topK + "|" + topK * rerankFactor);
          }
        }
        System.out.println("]");
        System.out.print("['" + nbooks + "'");
        float totalAverPerQuery = 0f;
        for (long elapsedCode : elapsedCodes) {
          double averPerQuery = elapsedCode * 1.0 / NUM_QUERIES;
          totalAverPerQuery += averPerQuery;
          System.out.print(", " + averPerQuery);
        }
        System.out.println("]");
        totalAverPerQuery = totalAverPerQuery / elapsedCodes.length;
        System.out.println("Performance, total avg per query in ms: " + totalAverPerQuery);
        System.out.println("____________________________________________________________");
      }
    }
  }

  private static int[] getNN(RandomAccessVectorValues.Floats reader, float[] query, int topK)
      throws IOException {
    int[] result = new int[topK];
    NeighborQueue queue = new NeighborQueue(topK, false);
    for (int j = 0; j < NUM_DOCS; j++) {
      float[] doc = reader.vectorValue(j);
      float dist = vectorFunction.compare(query, doc);
      queue.insertWithOverflow(j, dist);
    }
    for (int k = topK - 1; k >= 0; k--) {
      result[k] = queue.topNode();
      queue.pop();
    }
    return result;
  }

  private int compareNN(int[] expected, int[] results, int topK) {
    int matched = 0;
    Set<Integer> expectedSet = new HashSet<>();
    for (int i = 0; i < topK; i++) {
      expectedSet.add(expected[i]);
    }
    for (int scoreDoc : results) {
      if (expectedSet.contains(scoreDoc)) {
        ++matched;
      }
    }
    return matched;
  }
}
