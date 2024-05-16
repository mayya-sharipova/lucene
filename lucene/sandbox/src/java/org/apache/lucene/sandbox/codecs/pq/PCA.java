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

import java.util.Arrays;

/** PCA implementation */
public class PCA {

  public static Pair pca(float[][] docs) {
    int dim = docs[0].length;

    float[][] centeredDocs = centerData(docs);
    double[][] cov = covarianceMatrix(centeredDocs);

    // Compute the eigenvectors and eigenvalues of the covariance matrix
    // using the QR algorithm.
    double[] eigVals = new double[dim];
    double[][] eigVecs = new double[dim][dim];
    qrAlgorithm(cov, eigVals, eigVecs);

    Integer[] sortedEigValIdx = new Integer[dim];
    for (int i = 0; i < dim; i++) {
      sortedEigValIdx[i] = i;
    }

    Arrays.sort(sortedEigValIdx, (i, j) -> Double.compare(eigVals[j], eigVals[i]));

    double[] sortedEigVals = new double[dim];
    double[][] sortedEigVecs = new double[dim][dim];

    for (int i = 0; i < dim; i++) {
      int k = sortedEigValIdx[i];
      sortedEigVals[i] = eigVals[k];
      for (int j = 0; j < dim; j++) {
        sortedEigVecs[i][j] = eigVecs[k][j];
      }
    }
    return new Pair(sortedEigVals, sortedEigVecs);
  }

  private static float[][] centerData(float[][] docs) {
    int numDocs = docs.length;
    int dim = docs[0].length;
    float[] mean = new float[dim];
    float[][] centeredDocs = new float[numDocs][dim];
    for (int i = 0; i < numDocs; i++) {
      for (int j = 0; j < dim; j++) {
        mean[j] += docs[i][j];
        centeredDocs[i][j] = docs[i][j];
      }
    }
    for (int j = 0; j < dim; j++) {
      mean[j] /= numDocs;
    }
    for (int i = 0; i < numDocs; i++) {
      for (int j = 0; j < dim; j++) {
        centeredDocs[i][j] -= mean[j];
      }
    }
    return centeredDocs;
  }

  private static double[][] covarianceMatrix(float[][] data) {
    int numDocs = data.length;
    int dim = data[0].length;
    double[][] cov = new double[dim][dim];
    for (int i = 0; i < numDocs; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          cov[j][k] += data[i][j] * data[i][k];
        }
      }
    }
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        cov[j][k] /= numDocs;
      }
    }
    return cov;
  }

  private static void qrAlgorithm(double[][] matrix, double[] eigVals, double[][] eigVecs) {
    int n = matrix.length;
    double[][] q = new double[n][n];
    double[][] r = new double[n][n];
    double[][] a = matrix;
    for (int i = 0; i < n; i++) {
      Arrays.fill(eigVecs[i], 0.0);
      eigVecs[i][i] = 1.0;
    }
    double eps = 1e-10;
    while (true) {
      qrDecomposition(a, q, r);
      double[][] aNew = matrixMultiply(r, q);
      double maxOffDiag = 0.0;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (i != j) {
            maxOffDiag = Math.max(maxOffDiag, Math.abs(aNew[i][j]));
          }
        }
      }
      if (maxOffDiag < eps) {
        break;
      }
      a = aNew;
      double[][] vNew = matrixMultiply(eigVecs, q);
      eigVecs = vNew;
    }
    for (int i = 0; i < n; i++) {
      eigVals[i] = a[i][i];
    }
  }

  private static void qrDecomposition(double[][] matrix, double[][] q, double[][] r) {
    int n = matrix.length;
    double[][] a = matrix;
    for (int i = 0; i < n; i++) {
      Arrays.fill(q[i], 0.0);
      q[i][i] = 1.0;
    }
    for (int i = 0; i < n; i++) {
      double norm = 0.0;
      for (int j = 0; j < n; j++) {
        norm += a[j][i] * a[j][i];
      }
      norm = Math.sqrt(norm);
      for (int j = 0; j < n; j++) {
        r[j][i] = a[j][i] / norm;
      }
      for (int j = 0; j < i; j++) {
        double dotProduct = 0.0;
        for (int k = 0; k < n; k++) {
          dotProduct += a[k][j] * r[k][i];
        }
        for (int k = 0; k < n; k++) {
          r[k][i] -= dotProduct * r[k][j];
        }
      }
      for (int j = 0; j < n; j++) {
        double dotProduct = 0.0;
        for (int k = 0; k < n; k++) {
          dotProduct += a[k][j] * r[k][i];
        }
        for (int k = 0; k < n; k++) {
          q[k][i] = r[k][j] * dotProduct;
        }
      }
    }
  }

  private static double[][] matrixMultiply(double[][] a, double[][] b) {
    int m = a.length;
    int n = a[0].length;
    int p = b[0].length;
    double[][] c = new double[m][p];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < p; j++) {
        for (int k = 0; k < n; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return c;
  }

  record Pair(double[] sortedEigVals, double[][] sortedEigVecs) {}
}
