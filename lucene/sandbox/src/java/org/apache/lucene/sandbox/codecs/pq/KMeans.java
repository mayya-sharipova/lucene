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
import java.util.Arrays;
import java.util.Random;

import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** KMeans clustering algorithm. */
public class KMeans {
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

  public float[][] computeCentroids(int restarts, int iters) throws IOException {
    int[] documentCentroids = new int[numDocs];
    double minSquaredDist = Double.MAX_VALUE;
    double squaredDist = 0;
    float[][] bestCentroids = null;

    for (int restart = 0; restart < restarts; restart++) {
      float[][] centroids = initializeCentroidsSimple();
      for (int iter = 0; iter < iters; iter++) {
        squaredDist = runKMeansStep(centroids, documentCentroids);
      }
      if (squaredDist < minSquaredDist) {
        minSquaredDist = squaredDist;
        bestCentroids = centroids;
      }
    }
    return bestCentroids;
  }


  private float[][] initializeCentroidsSimple() throws IOException {
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
    return  initialCentroids;
  }

  private float[][] initializeCentroidsKMeansPlusPlus() throws IOException {
    float[][] initialCentroids = new float[numCentroids][];
    // Choose the first centroid uniformly at random
    int firstIndex = random.nextInt(numDocs);
    float[] value = reader.vectorValue(firstIndex);
    initialCentroids[0] = ArrayUtil.copyOfSubArray(value, startOffset, endOffset);

    // Store distances of each point to the nearest centroid
    float[] minDistances = new float[numDocs];
    Arrays.fill(minDistances, Float.MAX_VALUE);

    // Step 2 and 3: Select remaining centroids
    for (int i = 1; i < numCentroids; i++) {
      // Update distances with the new centroid
      double totalSum = 0;
      for (int j = 0; j < numDocs; j++) {
        value = reader.vectorValue(j);
        float dist = VectorUtil.squareDistance(ArrayUtil.copyOfSubArray(value, startOffset, endOffset), initialCentroids[i-1]);
        if (dist < minDistances[j]) {
          minDistances[j] = dist;
        }
        totalSum += minDistances[j];
      }

      // Randomly select next centroid
      double r = totalSum * random.nextDouble();
      double cummulativeSum = 0;
      int nextCentroidIndex = -1;
      for (int j = 0; j < numDocs; j++) {
        cummulativeSum += minDistances[j] ;
        if (cummulativeSum >= r) {
          nextCentroidIndex = j;
          break;
        }
      }
      // Update centroid
      initialCentroids[i] = ArrayUtil.copyOfSubArray(reader.vectorValue(nextCentroidIndex), startOffset, endOffset);
    }
    return  initialCentroids;
  }


  private double runKMeansStep(float[][] centroids, int[] documentCentroids) throws IOException {
    float[][] newCentroids = new float[centroids.length][centroids[0].length];
    int[] newCentroidSize = new int[centroids.length];

    double sumSquaredDist = 0;
    for (int docID = 0; docID < numDocs; docID++) {
      float[] value = reader.vectorValue(docID);
      float[] subVector = ArrayUtil.copyOfSubArray(value, startOffset, endOffset);

      int bestCentroid = -1;
      float minSquaredDist = Float.MAX_VALUE;
      for (int c = 0; c < centroids.length; c++) {
        float squareDist = VectorUtil.squareDistance(centroids[c], subVector);
        if (squareDist < minSquaredDist) {
          bestCentroid = c;
          minSquaredDist = squareDist;
        }
      }
      sumSquaredDist += minSquaredDist;

      newCentroidSize[bestCentroid]++;
      for (int v = 0; v < subVector.length; v++) {
        newCentroids[bestCentroid][v] += subVector[v];
      }
      documentCentroids[docID] = bestCentroid;
    }

    for (int c = 0; c < newCentroids.length; c++) {
      for (int v = 0; v < newCentroids[c].length; v++) {
        centroids[c][v] = newCentroids[c][v] / newCentroidSize[c];
      }
    }
    return sumSquaredDist;
  }
}
