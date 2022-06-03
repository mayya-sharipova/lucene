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
package org.apache.lucene.index;

import java.io.IOException;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;

/**
 * This class provides access to per-document floating point vector values indexed as {@link
 * KnnVectorField}.
 *
 * @lucene.experimental
 */
public abstract class VectorValues extends DocIdSetIterator {

  /** The maximum length of a vector */
  public static final int MAX_DIMENSIONS = 1024;

  /** Sole constructor */
  protected VectorValues() {}

  /** Return the dimension of the vectors */
  public abstract int dimension();

  /**
   * The total number of vectors in this iterator.
   *  If there are multiple vectors per document, this number
   *  can be larger than the number of documents having a value for this field.
   *
   * @return the number of vectors returned by this iterator
   */
  public abstract long size();

  /**
   * Retrieves the number of values for the current document. This must always be greater than zero.
  */
  public abstract int docValueCount();

  /**
   * Iterates and returns the next vector value in the current document. Do not call this more
   * than {@link#docValueCount} times for the document. It is illegal to call this method
   * when the iterator is not positioned: before advancing, or after failing to advance.
   * The returned array may be shared across calls, re-used, and modified as the iterator advances.
   *
   * @return the next vector value for the current document ID
  */
  public abstract float[] nextVectorValue() throws IOException;


  /**
   * Iterates and returns the next binary encoded vector value for the current document.
   * These are the bytes corresponding to the float array return by {@link #nextVectorValue}.
   * Do not call this more than {@link#docValueCount} times for the document.
   * It is illegal to call this method when the iterator is not positioned: before advancing,
   * or after failing to advance.
   *
   * The returned storage may be shared across calls, re-used, and modified as the iterator advances.
   *
   * @return the next binary encoded vector value for the current document ID
  */
  public BytesRef nextBinaryValue() throws IOException {
    throw new UnsupportedOperationException();
  }

  /**
   * Represents the lack of vector values. It is returned by providers that do not support
   * VectorValues.
   */
  public static final VectorValues EMPTY =
      new VectorValues() {

        @Override
        public long size() {
          return 0;
        }

        @Override
        public int docValueCount() {
          return 0;
        }

        @Override
        public int dimension() {
          return 0;
        }

        @Override
        public float[] nextVectorValue() {
          throw new IllegalStateException(
              "Attempt to get vectors from EMPTY values (which was not advanced)");
        }

        @Override
        public int docID() {
          throw new IllegalStateException("VectorValues is EMPTY, and not positioned on a doc");
        }

        @Override
        public int nextDoc() {
          return NO_MORE_DOCS;
        }

        @Override
        public int advance(int target) {
          return NO_MORE_DOCS;
        }

        @Override
        public long cost() {
          return 0;
        }
      };
}
