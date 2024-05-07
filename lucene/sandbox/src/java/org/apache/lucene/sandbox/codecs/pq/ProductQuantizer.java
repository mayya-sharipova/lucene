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
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** ProductQuantizer is a quantization algorithm that quantizes a vector into a byte array. */
public class ProductQuantizer {
  enum DistanceFunction {
    COSINE,
    L2,
    INNER_PRODUCT
  }

  static final int NUM_CENTROIDS = 256;

  private final int numDims;
  private final int numSubQuantizer;
  private final int subVectorLength;
  private final float[][][] centroids;
  private final DistanceFunction distanceFunction;

  private ProductQuantizer(
      int numDims, int numSubQuantizer, float[][][] centroids, DistanceFunction distanceFunction) {
    this.numDims = numDims;
    this.numSubQuantizer = numSubQuantizer;
    this.subVectorLength = numDims / numSubQuantizer;
    this.centroids = centroids;
    this.distanceFunction = distanceFunction;
  }

  public static ProductQuantizer create(
      RandomAccessVectorValues.Floats reader,
      int numSubQuantizer,
      DistanceFunction distanceFunction,
      long seed)
      throws IOException {
    int subVectorLength = reader.dimension() / numSubQuantizer;
    float[][][] centroids = new float[numSubQuantizer][][];
    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startOffset = i * subVectorLength;
      int endOffset = Math.min(startOffset + subVectorLength, reader.dimension());
      KMeans kmeans = new KMeans(reader, startOffset, endOffset, NUM_CENTROIDS, seed);
      centroids[i] = kmeans.computeCentroids();
    }
    return new ProductQuantizer(reader.dimension(), numSubQuantizer, centroids, distanceFunction);
  }

  public byte[] encode(float[] vector) {
    byte[] pqCode = new byte[numSubQuantizer];

    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startIndex = i * subVectorLength;
      int endIndex = Math.min(startIndex + subVectorLength, numDims);
      float[] subVector = ArrayUtil.copyOfSubArray(vector, startIndex, endIndex);
      pqCode[i] = computeNearestProductIndex(subVector, i);
    }
    return pqCode;
  }

  public DistanceRunner createDistanceRunner(float[] qVector) {
    float[] distances = new float[numSubQuantizer * NUM_CENTROIDS];
    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startIndex = i * subVectorLength;
      int endIndex = startIndex + subVectorLength;
      float[] subVector = ArrayUtil.copyOfSubArray(qVector, startIndex, endIndex);
      for (int j = 0; j < NUM_CENTROIDS; j++) {
        float dist;
        switch (distanceFunction) {
          case COSINE -> dist = VectorUtil.cosine(centroids[i][j], subVector);
          case L2 -> dist = 1f - VectorUtil.squareDistance(centroids[i][j], subVector);
          case INNER_PRODUCT -> dist = VectorUtil.dotProduct(centroids[i][j], subVector);
          default -> throw new IllegalArgumentException("not implemented");
        }
        distances[i * NUM_CENTROIDS + j] = dist;
      }
    }
    return new DistanceRunner(distances);
  }

  /**
   * DistanceRunner is a helper class to compute the distance between a query vector and a PQ code.
   */
  public static class DistanceRunner {
    public final float[] distanceTable;

    private DistanceRunner(float[] distances) {
      this.distanceTable = distances;
    }

    public float distance(byte[] pqCode) {
      if (pqCode.length >= 8) {
        return distanceUnrolled(pqCode);
      }
      return distanceSimple(pqCode);
    }

    private float distanceSimple(byte[] pqCode) {
      float score = 0f;
      for (int i = 0; i < pqCode.length; i++) {
        score += distanceTable[i * NUM_CENTROIDS + ((int) pqCode[i] & 0xFF)];
      }
      return score;
    }

    private float distanceUnrolled(byte[] pqCode) {
      float res = 0f;
      int i = 0;
      for (i = 0; i < pqCode.length % 8; i++) {
        res += distanceTable[i * NUM_CENTROIDS + ((int) pqCode[i] & 0xFF)];
      }
      if (pqCode.length < 8) {
        return res;
      }
      for (; i + 7 < pqCode.length; i += 8) {
        res += distanceTable[i * NUM_CENTROIDS + ((int) pqCode[i] & 0xFF)];
        res += distanceTable[(i + 1) * NUM_CENTROIDS + ((int) pqCode[(i + 1)] & 0xFF)];
        res += distanceTable[(i + 2) * NUM_CENTROIDS + ((int) pqCode[(i + 2)] & 0xFF)];
        res += distanceTable[(i + 3) * NUM_CENTROIDS + ((int) pqCode[(i + 3)] & 0xFF)];
        res += distanceTable[(i + 4) * NUM_CENTROIDS + ((int) pqCode[(i + 4)] & 0xFF)];
        res += distanceTable[(i + 5) * NUM_CENTROIDS + ((int) pqCode[(i + 5)] & 0xFF)];
        res += distanceTable[(i + 6) * NUM_CENTROIDS + ((int) pqCode[(i + 6)] & 0xFF)];
        res += distanceTable[(i + 7) * NUM_CENTROIDS + ((int) pqCode[(i + 7)] & 0xFF)];
      }
      return res;
    }
  }

  private byte computeNearestProductIndex(float[] subVector, int subIndex) {
    int centroidIndex = -1;
    float bestDistance = Float.NEGATIVE_INFINITY;
    for (int c = 0; c < NUM_CENTROIDS; c++) {
      float dist = 1f - VectorUtil.squareDistance(centroids[subIndex][c], subVector);
      if (dist > bestDistance) {
        bestDistance = dist;
        centroidIndex = c;
      }
    }
    return (byte) centroidIndex;
  }
}
