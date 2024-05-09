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

import static org.apache.lucene.sandbox.codecs.pq.ProductQuantizer.DistanceFunction;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** Benchmark for Product Quantization. */
@SuppressForbidden(reason = "System.out required: command line tool")
public class Benchmark {
  private static final Path dirPath = Paths.get("/Users/mayya/Elastic/knn/ann-prototypes/data");
  private static final String vectorFile = "corpus-quora-E5-small.fvec";
  private static final String queryFile = "queries-quora-E5-small.fvec";
  private static final int numDims = 384; //768;
  private static final int numDocs = 522_931;
  private static final int numQuery = 1_000;
  private static DistanceFunction distanceFunction = DistanceFunction.COSINE;
  private static int[] numSubQuantizers = new int[] {24};
  //new int[] {numDims / 32, numDims / 16, numDims / 8, numDims / 4};
  private static final int[] topKs = new int[] {10, 100};
  private static final int[] rerankFactors = new int[] {1, 10, 100};
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
              distanceFunction = DistanceFunction.COSINE;
              break;
            case "l2":
              distanceFunction = DistanceFunction.L2;
              break;
            case "ip":
              distanceFunction = DistanceFunction.INNER_PRODUCT;
              break;
            default:
              usage();
              throw new IllegalArgumentException("-metric can be 'cosine', 'l2' or 'ip' only");
          }
          break;
        default:
          usage();
          throw new IllegalArgumentException("unknown argument " + arg);
      }
    }
    new Benchmark().runBenchmark();
  }

  private static void usage() {
    String error = "Usage: Benchmark [-metric N] [-seed N]";
    System.err.println(error);
  }

  private void runBenchmark() throws Exception {
    final int byteSize = (numDims +  1) * Float.BYTES;
    for (int numSubQuantizer : numSubQuantizers) {
      final ProductQuantizer pq;
      try (MMapDirectory directory = new MMapDirectory(dirPath);
          IndexInput vectorInput = directory.openInput(vectorFile, IOContext.DEFAULT);
          IndexInput queryInput = directory.openInput(queryFile, IOContext.READONCE); ) {
        RandomAccessVectorValues.Floats vectorValues =
            new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                numDims, numDocs, vectorInput, byteSize);
        RandomAccessVectorValues.Floats queryVectorValues =
            new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                numDims, numQuery, queryInput, byteSize);

        long start = System.nanoTime();
        pq = ProductQuantizer.create(vectorValues, numSubQuantizer, distanceFunction, seed);
        long elapsed = System.nanoTime() - start;
        System.out.format(
            "Create product quantizer from %d  vectors and %d sub-quantizers in %d milliseconds%n",
            numDocs, numSubQuantizer, TimeUnit.NANOSECONDS.toMillis(elapsed));

        final byte[][] codes = new byte[numDocs][];
        long startEncode = System.nanoTime();
        for (int i = 0; i < vectorValues.size(); i++) {
          codes[i] = pq.encode(vectorValues.vectorValue(i));
        }
        long elapsedEncode = System.nanoTime() - startEncode;
        System.out.format(
            Locale.ROOT,
            "Encode %d vectors with %d sub-quantizers in %d milliseconds%n",
            numDocs,
            numSubQuantizer,
            TimeUnit.NANOSECONDS.toMillis(elapsedEncode));

        float[] recalls = new float[topKs.length * rerankFactors.length];
        long[] elapsedCodes = new long[topKs.length * rerankFactors.length];
        int row = 0;
        for (int topK : topKs) {
          int[][] groundTruths = new int[numQuery][];
          for (int i = 0; i < numQuery; i++) {
            float[] candidate = queryVectorValues.vectorValue(i);
            groundTruths[i] = getNN(vectorValues, candidate, topK);
          }
          for (int rerankFactor : rerankFactors) {
            int totalMatches = 0;
            int totalResults = 0;
            long elapsedCodeCmp = 0;
            for (int i = 0; i < numQuery; i++) {
              float[] candidate = queryVectorValues.vectorValue(i);
              long startCodeCmp = System.nanoTime();
              int[] results = getTopDocs(pq, codes, candidate, topK * rerankFactor);
              elapsedCodeCmp += System.nanoTime() - startCodeCmp;
              totalMatches += compareNN(groundTruths[i], results);
              totalResults += topK;
            }
            float recall = totalMatches / (float) totalResults;
            elapsedCodes[row] = elapsedCodeCmp;
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
        System.out.print("['" + numSubQuantizer + "'");
        for (float recall : recalls) {
          System.out.print(", " + recall);
        }
        System.out.println("]");

        System.out.println("Performance:");
        System.out.print("[PQ");
        for (int topK : topKs) {
          for (int rerankFactor : rerankFactors) {
            System.out.print(", " + topK + "|" + topK * rerankFactor);
          }
        }
        System.out.println("]");

        System.out.print("['" + numSubQuantizer + "'");
        for (long elapsedCode : elapsedCodes) {
          System.out.print(", " + TimeUnit.NANOSECONDS.toMillis(elapsedCode));
        }
        System.out.println("]");
      }
    }
  }

  private int[] getTopDocs(ProductQuantizer quantizer, byte[][] codes, float[] query, int topK) {
    NeighborQueue pq = new NeighborQueue(topK, false);
    ProductQuantizer.DistanceRunner runner = quantizer.createDistanceRunner(query);
    for (int i = 0; i < codes.length; i++) {
      float res = runner.distance(codes[i]);
      pq.insertWithOverflow(i, res);
    }
    int[] topDocs = new int[topK];
    for (int k = topK - 1; k >= 0; k--) {
      topDocs[k] = pq.topNode();
      pq.pop();
    }
    return topDocs;
  }

  private int[] getNN(RandomAccessVectorValues.Floats reader, float[] query, int topK)
      throws IOException {
    int[] result = new int[topK];
    NeighborQueue queue = new NeighborQueue(topK, false);
    for (int j = 0; j < numDocs; j++) {
      float[] doc = reader.vectorValue(j);
      float dist;
      switch (distanceFunction) {
        case COSINE -> dist = VectorUtil.cosine(query, doc);
        case L2 -> dist = 1f - VectorUtil.squareDistance(query, doc);
        case INNER_PRODUCT -> dist = VectorUtil.dotProduct(query, doc);
        default -> throw new IllegalArgumentException("Not implemented");
      }
      queue.insertWithOverflow(j, dist);
    }
    for (int k = topK - 1; k >= 0; k--) {
      result[k] = queue.topNode();
      queue.pop();
    }
    return result;
  }

  private int compareNN(int[] expected, int[] results) {
    int matched = 0;
    Set<Integer> expectedSet = new HashSet<>();
    for (int i = 0; i < expected.length; i++) {
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
