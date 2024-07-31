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

import static org.apache.lucene.sandbox.codecs.pq.ProductQuantizer.BOOK_SIZE;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/** A class contains various utility functions and classes for running PQ */
public class PQUtils {

  static class PQScorerSupplier implements RandomVectorScorerSupplier {
    private final RandomAccessVectorValues.Bytes quantizedVectors;
    private final float[][][] codebooks;
    private final VectorSimilarityFunction vectorSimFunction;
    private final float[][][] distances;

    PQScorerSupplier(
        RandomAccessVectorValues.Bytes quantizedVectors,
        VectorSimilarityFunction vectorSimFunction,
        float[][][] codebooks) {
      this.quantizedVectors = quantizedVectors;
      this.codebooks = codebooks;
      this.vectorSimFunction = vectorSimFunction;

      // pre-calculate distances between all centroids of each book
      int numBooks = codebooks.length;
      this.distances = new float[numBooks][BOOK_SIZE][BOOK_SIZE];
      for (int b = 0; b < numBooks; b++) {
        for (int c1 = 0; c1 < BOOK_SIZE; c1++) {
          for (int c2 = c1; c2 < BOOK_SIZE; c2++) {
            float dist = vectorSimFunction.compare(codebooks[b][c1], codebooks[b][c2]);
            distances[b][c1][c2] = dist;
            distances[b][c2][c1] = dist;
          }
        }
      }
    }

    @Override
    public RandomVectorScorer scorer(int ord) throws IOException {
      return new PQScorer(quantizedVectors, distances, quantizedVectors.vectorValue(ord));
    }

    @Override
    public RandomVectorScorerSupplier copy() throws IOException {
      throw new IllegalStateException("Not implemented");
    }

    public RandomVectorScorer scorer(byte[] query) throws IOException {
      return new PQScorer(quantizedVectors, distances, query);
    }

    public RandomVectorScorer queryScorer(float[] query, float[] coarseCentroid) {
      return new QueryPQScorer(
          quantizedVectors, vectorSimFunction, coarseCentroid, codebooks, query);
    }
  }

  static class PQScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    private final RandomAccessVectorValues.Bytes values;
    private final float[][][] distances;
    private final byte[] queryPQCode;

    public PQScorer(RandomAccessVectorValues.Bytes values, float[][][] distances, byte[] query) {
      super(values);
      this.values = values;
      this.distances = distances;
      this.queryPQCode = query;
    }

    @Override
    public float score(int node) throws IOException {
      byte[] pqCode = values.vectorValue(node);
      if (pqCode.length >= 8) {
        return distanceUnrolled(pqCode);
      }
      return distanceSimple(pqCode);
    }

    private float distanceSimple(byte[] pqCode) {
      float score = 0f;
      for (int b = 0; b < pqCode.length; b++) {
        int docIdx = (int) pqCode[b] & 0xFF;
        int queryIdx = (int) queryPQCode[b] & 0xFF;
        score += distances[b][queryIdx][docIdx];
      }
      return score;
    }

    private float distanceUnrolled(byte[] pqCode) {
      float res = 0f;
      int i;
      for (i = 0; i < pqCode.length % 8; i++) {
        int docIdx = (int) pqCode[i] & 0xFF;
        int queryIdx = (int) queryPQCode[i] & 0xFF;
        res += distances[i][queryIdx][docIdx];
      }
      if (pqCode.length < 8) {
        return res;
      }
      for (; i + 7 < pqCode.length; i += 8) {
        res += distances[i][(int) queryPQCode[i] & 0xFF][((int) pqCode[i] & 0xFF)];
        res += distances[i + 1][(int) queryPQCode[i + 1] & 0xFF][((int) pqCode[i + 1] & 0xFF)];
        res += distances[i + 2][(int) queryPQCode[i + 2] & 0xFF][((int) pqCode[i + 2] & 0xFF)];
        res += distances[i + 3][(int) queryPQCode[i + 3] & 0xFF][((int) pqCode[i + 3] & 0xFF)];
        res += distances[i + 4][(int) queryPQCode[i + 4] & 0xFF][((int) pqCode[i + 4] & 0xFF)];
        res += distances[i + 5][(int) queryPQCode[i + 5] & 0xFF][((int) pqCode[i + 5] & 0xFF)];
        res += distances[i + 6][(int) queryPQCode[i + 6] & 0xFF][((int) pqCode[i + 6] & 0xFF)];
        res += distances[i + 7][(int) queryPQCode[i + 7] & 0xFF][((int) pqCode[i + 7] & 0xFF)];
      }
      return res;
    }
  }

  static class QueryPQScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    private final RandomAccessVectorValues.Bytes values;
    private final VectorSimilarityFunction vectorSimFunction;
    private final float[][][] codebooks;
    private final float[][] querySubVectors;

    public QueryPQScorer(
        RandomAccessVectorValues.Bytes values,
        VectorSimilarityFunction vectorSimFunction,
        float[] coarseCentroid,
        float[][][] codebooks,
        float[] query) {
      super(values);
      this.values = values;
      this.vectorSimFunction = vectorSimFunction;
      this.codebooks = codebooks;
      int numBooks = codebooks.length;
      int bookDim = codebooks[0][0].length;
      this.querySubVectors = new float[numBooks][bookDim];
      for (int b = 0; b < numBooks; b++) {
        int offset = b * bookDim;
        for (int dim = 0; dim < bookDim; dim++) {
          querySubVectors[b][dim] = query[offset + dim] - coarseCentroid[offset + dim];
        }
      }
    }

    @Override
    public float score(int node) throws IOException {
      byte[] pqCode = values.vectorValue(node);
      float score = 0f;
      for (int b = 0; b < pqCode.length; b++) {
        score +=
            vectorSimFunction.compare(codebooks[b][(int) pqCode[b] & 0xFF], querySubVectors[b]);
      }
      return score;
    }
  }

  static class QuantizedVectors implements RandomAccessVectorValues.Bytes {
    private final byte[][] docCodes;
    private final int dim;

    QuantizedVectors(byte[][] docCodes) {
      this.docCodes = docCodes;
      this.dim = docCodes[0].length;
    }

    @Override
    public int size() {
      return docCodes.length;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public Bytes copy() throws IOException {
      return new QuantizedVectors(docCodes);
    }

    @Override
    public byte[] vectorValue(int targetOrd) {
      return docCodes[targetOrd];
    }
  }

  static class QuantizedDocs extends DocIdSetIterator {
    private final int[] docs;
    private final byte[][] docCodes;
    private int idx = -1;
    private int doc = -1;

    QuantizedDocs(int[] docs, byte[][] docCodes) {
      this.docs = docs;
      this.docCodes = docCodes;
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      idx++;
      if (idx >= docs.length) {
        doc = NO_MORE_DOCS;
      } else {
        doc = docs[idx];
      }
      return doc;
    }

    @Override
    public int advance(int target) throws IOException {
      idx = target;
      if (idx >= docs.length) {
        doc = NO_MORE_DOCS;
      } else {
        doc = docs[idx];
      }
      return doc;
    }

    @Override
    public long cost() {
      return docs.length;
    }

    public byte[] pqCode() {
      if (doc == -1 || doc == NO_MORE_DOCS) {
        throw new IllegalStateException("Iterator needs to be positioned on a valid doc");
      }
      return docCodes[doc];
    }
  }
}
