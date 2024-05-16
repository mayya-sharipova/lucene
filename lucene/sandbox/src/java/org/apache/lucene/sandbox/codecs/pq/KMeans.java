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
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** KMeans clustering algorithm. */
public class KMeans {

  private final RandomAccessVectorValues.Floats vectors;
  private final int numDocs;
  private final int numCentroids;
  private final Random random;
  private final int initializationMethod;

  public KMeans(RandomAccessVectorValues.Floats vectors, int numCentroids, long seed) {
    this.vectors = vectors;
    this.numDocs = vectors.size();
    this.numCentroids = numCentroids;
    this.random = new Random(seed);
    this.initializationMethod = 0;
  }

  public float[][] computeCentroids(
      int restarts, int iters, Consumer<float[][]> centersTransformFunc) throws IOException {
    short[] docCentroids = new short[numDocs];
    double minSquaredDist = Double.MAX_VALUE;
    double squaredDist = 0;
    float[][] bestCentroids = null;

    for (int restart = 0; restart < restarts; restart++) {
      float[][] centroids =
          switch (initializationMethod) {
            case 1 -> initializeCentroidsSimple();
            case 2 -> initializeCentroidsPlusPlus();
            default -> initializeForgy();
          };

      for (int iter = 0; iter < iters; iter++) {
        squaredDist = runKMeansStep(vectors, centroids, docCentroids, false, centersTransformFunc);
      }
      if (squaredDist < minSquaredDist) {
        minSquaredDist = squaredDist;
        bestCentroids = centroids;
      }
    }
    return bestCentroids;
  }

  private float[][] initializeForgy() throws IOException {
    Set<Integer> selection = new HashSet<>();
    while (selection.size() < numCentroids) {
      int cand = random.nextInt(numDocs);
      selection.add(cand);
    }

    float[][] initialCentroids = new float[numCentroids][];
    int i = 0;
    for (Integer selectedIdx : selection) {
      float[] vector = vectors.vectorValue(selectedIdx);
      initialCentroids[i] = ArrayUtil.copyOfSubArray(vector, 0, vector.length);
      i++;
    }
    return initialCentroids;
  }

  private float[][] initializeCentroidsSimple() throws IOException {
    float[][] initialCentroids = new float[numCentroids][];
    for (int index = 0; index < numDocs; index++) {
      float[] vector = vectors.vectorValue(index);
      if (index < numCentroids) {
        initialCentroids[index] = ArrayUtil.copyOfSubArray(vector, 0, vector.length);
      } else if (random.nextDouble() < numCentroids * (1.0 / index)) {
        int c = random.nextInt(numCentroids);
        initialCentroids[c] = ArrayUtil.copyOfSubArray(vector, 0, vector.length);
      }
    }
    return initialCentroids;
  }

  private float[][] initializeCentroidsPlusPlus() throws IOException {
    float[][] initialCentroids = new float[numCentroids][];
    // Choose the first centroid uniformly at random
    int firstIndex = random.nextInt(numDocs);
    float[] value = vectors.vectorValue(firstIndex);
    initialCentroids[0] = ArrayUtil.copyOfSubArray(value, 0, value.length);

    // Store distances of each point to the nearest centroid
    float[] minDistances = new float[numDocs];
    Arrays.fill(minDistances, Float.MAX_VALUE);

    // Step 2 and 3: Select remaining centroids
    for (int i = 1; i < numCentroids; i++) {
      // Update distances with the new centroid
      double totalSum = 0;
      for (int j = 0; j < numDocs; j++) {
        float dist = VectorUtil.squareDistance(vectors.vectorValue(j), initialCentroids[i - 1]);
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
        cummulativeSum += minDistances[j];
        if (cummulativeSum >= r) {
          nextCentroidIndex = j;
          break;
        }
      }
      // Update centroid
      value = vectors.vectorValue(nextCentroidIndex);
      initialCentroids[i] = ArrayUtil.copyOfSubArray(value, 0, value.length);
    }
    return initialCentroids;
  }

  public static double runKMeansStep(
      RandomAccessVectorValues.Floats vectors,
      float[][] centroids,
      short[] docCentroids,
      boolean useKahanSummation,
      Consumer<float[][]> centersTransformFunc)
      throws IOException {
    short numCentroids = (short) centroids.length;

    float[][] newCentroids = new float[numCentroids][centroids[0].length];
    int[] newCentroidSize = new int[numCentroids];
    float[][] compensations = null;
    if (useKahanSummation) {
      compensations = new float[numCentroids][centroids[0].length];
    }

    double sumSquaredDist = 0;
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);

      short bestCentroid;
      if (numCentroids == 1) {
        bestCentroid = 0;
      } else {
        bestCentroid = -1;
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          float squareDist = VectorUtil.squareDistance(centroids[c], vector);
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
        sumSquaredDist += minSquaredDist;
      }

      newCentroidSize[bestCentroid]++;
      for (int dim = 0; dim < vector.length; dim++) {
        // For large datasets use Kahan summation to accumulate the new centres,
        // since we can easily reach the limits of float precision
        if (useKahanSummation) {
          float y = vector[dim] - compensations[bestCentroid][dim];
          float t = newCentroids[bestCentroid][dim] + y;
          compensations[bestCentroid][dim] = (t - newCentroids[bestCentroid][dim]) - y;
          newCentroids[bestCentroid][dim] = t;
        } else {
          newCentroids[bestCentroid][dim] += vector[dim];
        }
      }
      docCentroids[docID] = bestCentroid;
    }

    for (int c = 0; c < numCentroids; c++) {
      for (int dim = 0; dim < newCentroids[c].length; dim++) {
        centroids[c][dim] = newCentroids[c][dim] / newCentroidSize[c];
      }
    }
    if (centersTransformFunc != null) {
      centersTransformFunc.accept(centroids);
    }
    return sumSquaredDist;
  }
}
