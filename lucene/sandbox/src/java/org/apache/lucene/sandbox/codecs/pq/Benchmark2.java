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

import static org.apache.lucene.util.hnsw.HnswGraphBuilder.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.util.hnsw.HnswGraphBuilder.DEFAULT_MAX_CONN;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.PrintStreamInfoStream;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.hnsw.*;

/** Benchmark for Product Quantization. */
@SuppressForbidden(reason = "System.out required: command line tool")
public class Benchmark2 {
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

  private static final int[] topKs = new int[] {10};
  private static final int[] rerankFactors = new int[] {1, 2, 4, 8, 10};

  public static void main(String[] args) throws Exception {
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      switch (arg) {
        case "-metric":
          String metric = args[++i];
          switch (metric) {
            case "cosine":
              vectorFunction = VectorSimilarityFunction.COSINE;
              break;
            case "dot_product":
              vectorFunction = VectorSimilarityFunction.DOT_PRODUCT;
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
        case "-groundTruthFile":
          groundTruthFile = args[++i];
          break;
        default:
          usage();
          throw new IllegalArgumentException("unknown argument " + arg);
      }
    }
    new Benchmark2().runBenchmark();
  }

  private static void usage() {
    String error = "Usage: Benchmark [-metric N]";
    System.err.println(error);
  }

  private void runBenchmark() throws Exception {
    try (MMapDirectory directory = new MMapDirectory(dirPath);
        IndexInput vectorInput =
            directory.openInput(vectorFile, IOContext.DEFAULT.withReadAdvice(ReadAdvice.RANDOM));
        IndexInput queryInput = directory.openInput(queryFile, IOContext.READONCE); ) {
      long start;
      long elapsed;
      RandomAccessVectorValues.Floats vectorValues =
          new VectorsReaderWithOffset(
              vectorInput, NUM_DOCS, NUM_DIMS, FILE_BYTESIZE, FILE_VECTOR_OFFSET);
      //      start = System.nanoTime();
      //      float[][] vectors = new float[NUM_DOCS][NUM_DIMS];
      //      for (int i = 0; i < NUM_DOCS; i++) {
      //        vectors[i] = vectorValues0.vectorValue(i).clone();
      //      }
      //      RandomAccessVectorValues.Floats vectorValues = new FloatVectors(vectors);
      //      elapsed = System.nanoTime() - start;
      //      System.out.format(
      //          "Reading vectors from the file took: %d ms, vectors size: %d docs, dims:%d %n",
      //          TimeUnit.NANOSECONDS.toMillis(elapsed), vectorValues.size(),
      // vectorValues.dimension());

      FlatVectorsScorer vectorsScorer = DefaultFlatVectorScorer.INSTANCE;
      RandomVectorScorerSupplier scorerSupplier =
          vectorsScorer.getRandomVectorScorerSupplier(vectorFunction, vectorValues);
      start = System.nanoTime();
      HnswGraphBuilder hnswGraphBuilder =
          HnswGraphBuilder.create(
              scorerSupplier,
              DEFAULT_MAX_CONN,
              DEFAULT_BEAM_WIDTH,
              HnswGraphBuilder.randSeed,
              vectorValues.size());
      hnswGraphBuilder.setInfoStream(new PrintStreamInfoStream(System.out));
      OnHeapHnswGraph hnswGraph = hnswGraphBuilder.build(vectorValues.size());
      elapsed = System.nanoTime() - start;
      System.out.format(
          "Building HNSW graph took: %d ms, vectors size: %d docs, dims:%d %n",
          TimeUnit.NANOSECONDS.toMillis(elapsed), vectorValues.size(), vectorValues.dimension());

      RandomAccessVectorValues.Floats queryVectorValues =
          new VectorsReaderWithOffset(
              queryInput, NUM_QUERIES, NUM_DIMS, FILE_BYTESIZE, FILE_VECTOR_OFFSET);

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
            RandomVectorScorer scorer =
                vectorsScorer.getRandomVectorScorer(vectorFunction, vectorValues, candidate);
            KnnCollector knnCollector =
                HnswGraphSearcher.search(
                    scorer, topK * rerankFactor, hnswGraph, null, Integer.MAX_VALUE);
            ScoreDoc[] scoreDocs = knnCollector.topDocs().scoreDocs;
            int[] results = new int[scoreDocs.length];
            for (int doc = 0; doc < scoreDocs.length; doc++) {
              results[doc] = scoreDocs[doc].doc;
            }
            elapsedCodeCmp += System.nanoTime() - startCodeCmp;
            totalMatches += compareNN(groundTruths[i], results, topK);
            totalResults += topK;
          }
          float recall = totalMatches / (float) totalResults;
          elapsedCodes[row] = elapsedCodeCmp;
          recalls[row++] = recall;
        }
      }
      System.out.println("Recall:");
      System.out.print("[");
      for (int topK : topKs) {
        for (int rerankFactor : rerankFactors) {
          System.out.print(topK + "|" + topK * rerankFactor + ", ");
        }
      }
      System.out.println("]");
      System.out.print("[");
      for (float recall : recalls) {
        System.out.print(recall + ", ");
      }
      System.out.println("]");

      System.out.println("Performance:");
      System.out.print("[");
      for (int topK : topKs) {
        for (int rerankFactor : rerankFactors) {
          System.out.print(topK + "|" + topK * rerankFactor + ", ");
        }
      }
      System.out.println("]");
      System.out.print("[");
      for (long elapsedCode : elapsedCodes) {
        System.out.print(TimeUnit.NANOSECONDS.toMillis(elapsedCode) + ", ");
      }
      System.out.println("]");
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
