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
import java.util.Random;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** KMeans clustering algorithm. */
public class KMeans {
  private static final int NUM_ITERS = 10;
  private final RandomAccessVectorValues.Floats reader;
  private final int numDocs;
  private int startOffset;
  private int endOffset;
  private final int numCentroids;
  private final Random random;

  public KMeans(
      RandomAccessVectorValues.Floats reader,
      int startOffset,
      int endOffset,
      int numCentroids,
      long seed) {
    this.reader = reader;
    this.numDocs = reader.size();
    this.numCentroids = numCentroids;
    this.startOffset = startOffset;
    this.endOffset = endOffset;
    this.random = new Random(seed);
  }

  public float[][] computeCentroids() throws IOException {
    float[][] initialCentroids = new float[numCentroids][];
    for (int index = 0; index < numDocs; index++) {
      float[] value = reader.vectorValue(index);
      if (index < numCentroids) {
        initialCentroids[index] = ArrayUtil.copyOfSubArray(value, startOffset, endOffset);
      } else if (random.nextDouble() < numCentroids * (1.0 / index)) {
        int c = random.nextInt(numCentroids);
        initialCentroids[c] = ArrayUtil.copyOfSubArray(value, startOffset, endOffset);
      }
    }

    return runKMeans(initialCentroids);
  }

  private float[][] runKMeans(float[][] centroids) throws IOException {
    int[] documentCentroids = new int[numDocs];
    for (int iter = 0; iter < NUM_ITERS; iter++) {
      centroids = runKMeansStep(centroids, documentCentroids);
    }
    return centroids;
  }

  private float[][] runKMeansStep(float[][] centroids, int[] documentCentroids) throws IOException {
    float[][] newCentroids = new float[centroids.length][centroids[0].length];
    int[] newCentroidSize = new int[centroids.length];

    for (int docID = 0; docID < numDocs; docID++) {
      float[] value = reader.vectorValue(docID);
      float[] subVector = ArrayUtil.copyOfSubArray(value, startOffset, endOffset);

      int bestCentroid = -1;
      float bestDist = Float.NEGATIVE_INFINITY;
      for (int c = 0; c < centroids.length; c++) {
        float dist = 1f - VectorUtil.squareDistance(centroids[c], subVector);
        if (dist > bestDist) {
          bestCentroid = c;
          bestDist = dist;
        }
      }
      newCentroidSize[bestCentroid]++;
      for (int v = 0; v < subVector.length; v++) {
        newCentroids[bestCentroid][v] += subVector[v];
      }
      documentCentroids[docID] = bestCentroid;
    }

    for (int c = 0; c < newCentroids.length; c++) {
      for (int v = 0; v < newCentroids[c].length; v++) {
        newCentroids[c][v] /= newCentroidSize[c];
      }
    }
    return newCentroids;
  }
}
