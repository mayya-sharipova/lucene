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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;

/**
 * Buffers up pending vector value(s) per doc, then flushes when segment flushes.
 *
 * @lucene.experimental
 */
class VectorValuesWriter {

  private final FieldInfo fieldInfo;
  private final Counter iwBytesUsed;
  //TODO a better data structure than List
  private final List<float[]> vectors = new ArrayList<>();
  private PackedLongValues.Builder pendingCounts; // count of values per doc
  private final DocsWithFieldSet docsWithField;

  private int lastDocID = -1;
  private int lastDocValuesCount = 0;

  private long bytesUsed;

  VectorValuesWriter(FieldInfo fieldInfo, Counter iwBytesUsed) {
    this.fieldInfo = fieldInfo;
    this.iwBytesUsed = iwBytesUsed;
    this.docsWithField = new DocsWithFieldSet();
    this.bytesUsed = docsWithField.ramBytesUsed();
    if (iwBytesUsed != null) {
      iwBytesUsed.addAndGet(bytesUsed);
    }
  }

  /**
   * Adds a value for the given document. Only a single value may be added.
   *
   * @param docID the value is added to this document
   * @param vectorValue the value to add
   * @throws IllegalArgumentException if a value has already been added to the given document
   */
  public void addValue(int docID, float[] vectorValue) {
    assert docID >= lastDocID;
    if (vectorValue.length != fieldInfo.getVectorDimension()) {
      throw new IllegalArgumentException(
          "Attempt to index a vector of dimension "
              + vectorValue.length
              + " but \""
              + fieldInfo.name
              + "\" has dimension "
              + fieldInfo.getVectorDimension());
    }
    if (docID != lastDocID) {
      finishLastDoc();
      lastDocID = docID;
    }
    vectors.add(ArrayUtil.copyOfSubArray(vectorValue, 0, vectorValue.length));
    lastDocValuesCount++;
    updateBytesUsed();
  }

  private void finishLastDoc() {
    if (lastDocID == -1) {
      return;
    }
    // record the number of values for this doc
    if (pendingCounts != null) {
      pendingCounts.add(lastDocValuesCount);
    } else if (lastDocValuesCount != 1) {
      // initialize pendingCounts at the first occurrence of doc with multiple values
      pendingCounts = PackedLongValues.deltaPackedBuilder(PackedInts.COMPACT);
      for (int i = 0; i < docsWithField.cardinality(); ++i) {
        pendingCounts.add(1);
      }
      pendingCounts.add(lastDocValuesCount);
    }
    lastDocValuesCount = 0;

    docsWithField.add(lastDocID);
  }

  private void updateBytesUsed() {
    final long newBytesUsed =
        docsWithField.ramBytesUsed()
            + vectors.size()
                * (RamUsageEstimator.NUM_BYTES_OBJECT_REF
                    + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
            + vectors.size() * vectors.get(0).length * Float.BYTES
            + (pendingCounts == null ? 0 : pendingCounts.ramBytesUsed());
    if (iwBytesUsed != null) {
      iwBytesUsed.addAndGet(newBytesUsed - bytesUsed);
    }
    bytesUsed = newBytesUsed;
  }

  /**
   * Flush this field's values to storage, sorting the values in accordance with sortMap
   *
   * @param sortMap specifies the order of documents being flushed, or null if they are to be
   *     flushed in docid order
   * @param knnVectorsWriter the Codec's vector writer that handles the actual encoding and I/O
   * @throws IOException if there is an error writing the field and its values
   */
  public void flush(Sorter.DocMap sortMap, KnnVectorsWriter knnVectorsWriter) throws IOException {
    KnnVectorsReader knnVectorsReader =
        new KnnVectorsReader() {
          @Override
          public long ramBytesUsed() {
            return 0;
          }

          @Override
          public void close() {
            throw new UnsupportedOperationException();
          }

          @Override
          public void checkIntegrity() {
            throw new UnsupportedOperationException();
          }

          @Override
          public VectorValues getVectorValues(String field) throws IOException {
            if (pendingCounts == null) {
              VectorValues vectorValues = new BufferedVectorValues(docsWithField, vectors,
                  fieldInfo.getVectorDimension());
              if (sortMap == null) {
                return vectorValues;
              } else {
                return new SortingVectorValues(vectorValues, sortMap);
              }
            } else {
              PackedLongValues valueCounts = pendingCounts.build();
              VectorValues vectorValues = new MultipleBufferedVectorValues(docsWithField, valueCounts,
                  vectors, fieldInfo.getVectorDimension());
              if (sortMap == null) {
                return vectorValues;
              } else {
                return new MultipleSortingVectorValues(vectorValues, sortMap);
              }
            }
          }

          @Override
          public TopDocs search(
              String field, float[] target, int k, Bits acceptDocs, int visitedLimit) {
            throw new UnsupportedOperationException();
          }
        };

    knnVectorsWriter.writeField(fieldInfo, knnVectorsReader);
  }

  static class MultipleSortingVectorValues extends VectorValues
      implements RandomAccessVectorValuesProducer {

    private final VectorValues values;
    private final RandomAccessVectorValues randomAccess;
    // map newDocID -> valueCounts
    private final int[] newValueCounts;
    // map newDocID -> offset in randomAccess from which to read values
    private final long[] valuesOffsets;
    private final long[] ordMap;

    private int docId = -1;
    private int numValues = -1;
    private long currentOffset = -1;
    private long currentEndOffset;

    MultipleSortingVectorValues(VectorValues values, Sorter.DocMap sortMap) throws IOException {
      this.values = values;
      newValueCounts = new int[sortMap.size()];
      valuesOffsets = new long[sortMap.size()];
      randomAccess = ((RandomAccessVectorValuesProducer) values).randomAccess();

      long valuesOffset = 0;
      int docID;
      while ((docID = values.nextDoc()) != NO_MORE_DOCS) {
        int newDocID = sortMap.oldToNew(docID);
        int numValues = values.docValueCount();
        newValueCounts[newDocID] = numValues;
        valuesOffsets[newDocID] = valuesOffset;
        for (int i = 0; i < numValues; i++) {
          values.nextVectorValue();
          valuesOffset++;
        }
      }

      // set up ordMap to map from new dense ordinal to old dense ordinal
      ordMap = new long[(int) values.size()];
      int ord = 0;
      for (int newDocId = 0; newDocId < newValueCounts.length; newDocId++) {
        int numValues = newValueCounts[newDocId];
        for (int j = 0; j < numValues; j++) {
          ordMap[ord++] = valuesOffsets[newDocId] + j;
        }
      }
    }

    @Override
    public int docID() {
      return docId;
    }

    @Override
    public int nextDoc() throws IOException {
      do {
        docId++;
        if (docId >= newValueCounts.length) {
          return docId = NO_MORE_DOCS;
        }
      } while (newValueCounts[docId] == 0);
      numValues = newValueCounts[docId];
      currentOffset = valuesOffsets[docId];
      currentEndOffset = currentOffset + numValues;
      return docId;
    }

    @Override
    public int docValueCount() {
      return numValues;
    }

    @Override
    public BytesRef nextBinaryValue() throws IOException {
      if (currentOffset == currentEndOffset) {
        throw new AssertionError();
      } else {
        return randomAccess.binaryValue(currentOffset++);
      }
    }

    @Override
    public float[] nextVectorValue() throws IOException {
      if (currentOffset == currentEndOffset) {
        throw new AssertionError();
      } else {
        return randomAccess.vectorValue(currentOffset++);
      }
    }

    @Override
    public int dimension() {
      return values.dimension();
    }

    @Override
    public long size() {
      return values.size();
    }

    @Override
    public int advance(int target) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return values.cost();
    }

    @Override
    public RandomAccessVectorValues randomAccess() throws IOException {

      // Must make a new delegate randomAccess so that we have our own distinct float[]
      final RandomAccessVectorValues delegateRA =
          ((RandomAccessVectorValuesProducer) MultipleSortingVectorValues.this.values).randomAccess();

      return new RandomAccessVectorValues() {

        @Override
        public long size() {
          return delegateRA.size();
        }

        @Override
        public int dimension() {
          return delegateRA.dimension();
        }

        @Override
        public float[] vectorValue(long targetOrd) throws IOException {
          return delegateRA.vectorValue(ordMap[(int) targetOrd]);
        }

        @Override
        public BytesRef binaryValue(long targetOrd) {
          throw new UnsupportedOperationException();
        }
      };
    }
  }

  private static class BufferedVectorValues extends VectorValues
          implements RandomAccessVectorValues, RandomAccessVectorValuesProducer {

    final DocsWithFieldSet docsWithField;

    // These are always the vectors of a VectorValuesWriter, which are copied when added to it
    final List<float[]> vectors;
    final int dimension;

    final ByteBuffer buffer;
    final BytesRef binaryValue;
    final ByteBuffer raBuffer;
    final BytesRef raBinaryValue;

    DocIdSetIterator docsWithFieldIter;
    int ord = -1;

    BufferedVectorValues(DocsWithFieldSet docsWithField, List<float[]> vectors, int dimension) {
      this.docsWithField = docsWithField;
      this.vectors = vectors;
      this.dimension = dimension;
      buffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      binaryValue = new BytesRef(buffer.array());
      raBuffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      raBinaryValue = new BytesRef(raBuffer.array());
      docsWithFieldIter = docsWithField.iterator();
    }

    @Override
    public RandomAccessVectorValues randomAccess() {
      return new BufferedVectorValues(docsWithField, vectors, dimension);
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public long size() {
      return vectors.size();
    }

    @Override
    public int docValueCount() {
      return 1;
    }

    @Override
    public BytesRef nextBinaryValue() {
      buffer.asFloatBuffer().put(nextVectorValue());
      return binaryValue;
    }

    @Override
    public BytesRef binaryValue(long targetOrd) {
      raBuffer.asFloatBuffer().put(vectors.get((int) targetOrd));
      return raBinaryValue;
    }

    @Override
    public float[] nextVectorValue() {
      return vectors.get(ord);
    }

    @Override
    public float[] vectorValue(long targetOrd) {
      return vectors.get((int) targetOrd);
    }

    @Override
    public int docID() {
      return docsWithFieldIter.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      int docID = docsWithFieldIter.nextDoc();
      if (docID != NO_MORE_DOCS) {
        ++ord;
      }
      return docID;
    }

    @Override
    public int advance(int target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return docsWithFieldIter.cost();
    }
  }

  static class SortingVectorValues extends VectorValues
          implements RandomAccessVectorValuesProducer {

    private final VectorValues delegate;
    private final RandomAccessVectorValues randomAccess;
    private final int[] docIdOffsets;
    private final int[] ordMap;
    private int docId = -1;

    SortingVectorValues(VectorValues delegate, Sorter.DocMap sortMap) throws IOException {
      this.delegate = delegate;
      randomAccess = ((RandomAccessVectorValuesProducer) delegate).randomAccess();
      docIdOffsets = new int[sortMap.size()];

      int offset = 1; // 0 means no vector for this (field, document)
      int docID;
      while ((docID = delegate.nextDoc()) != NO_MORE_DOCS) {
        int newDocID = sortMap.oldToNew(docID);
        docIdOffsets[newDocID] = offset++;
      }

      // set up ordMap to map from new dense ordinal to old dense ordinal
      ordMap = new int[offset - 1];
      int ord = 0;
      for (int docIdOffset : docIdOffsets) {
        if (docIdOffset != 0) {
          ordMap[ord++] = docIdOffset - 1;
        }
      }
      assert ord == ordMap.length;
    }

    @Override
    public int docID() {
      return docId;
    }

    @Override
    public int nextDoc() throws IOException {
      while (docId < docIdOffsets.length - 1) {
        ++docId;
        if (docIdOffsets[docId] != 0) {
          return docId;
        }
      }
      docId = NO_MORE_DOCS;
      return docId;
    }

    @Override
    public int docValueCount() {
      return 0;
    }

    @Override
    public BytesRef nextBinaryValue() throws IOException {
      return randomAccess.binaryValue(docIdOffsets[docId] - 1);
    }

    @Override
    public float[] nextVectorValue() throws IOException {
      return randomAccess.vectorValue(docIdOffsets[docId] - 1);
    }

    @Override
    public int dimension() {
      return delegate.dimension();
    }

    @Override
    public long size() {
      return delegate.size();
    }

    @Override
    public int advance(int target) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return delegate.cost();
    }

    @Override
    public RandomAccessVectorValues randomAccess() throws IOException {

      // Must make a new delegate randomAccess so that we have our own distinct float[]
      final RandomAccessVectorValues delegateRA =
              ((RandomAccessVectorValuesProducer) SortingVectorValues.this.delegate).randomAccess();

      return new RandomAccessVectorValues() {

        @Override
        public long size() {
          return delegateRA.size();
        }

        @Override
        public int dimension() {
          return delegateRA.dimension();
        }

        @Override
        public float[] vectorValue(long targetOrd) throws IOException {
          return delegateRA.vectorValue(ordMap[(int) targetOrd]);
        }

        @Override
        public BytesRef binaryValue(long targetOrd) {
          throw new UnsupportedOperationException();
        }
      };
    }
  }

  private static class MultipleBufferedVectorValues extends VectorValues
      implements RandomAccessVectorValues, RandomAccessVectorValuesProducer {

    // These are always the vectors of a VectorValuesWriter, which are copied when added to it
    final List<float[]> vectors;
    final int dimension;
    final DocsWithFieldSet docsWithField;
    final PackedLongValues valueCounts;
    final Iterator<float[]> valuesIter;
    final PackedLongValues.Iterator valueCountsIter;
    final DocIdSetIterator docsWithFieldIter;

    final ByteBuffer buffer;
    final BytesRef binaryValue;
    final ByteBuffer raBuffer;
    final BytesRef raBinaryValue;

    private int valueCount;
    private int valueUpto;

    MultipleBufferedVectorValues(DocsWithFieldSet docsWithField, PackedLongValues valueCounts,
          List<float[]> vectors, int dimension) {
      this.docsWithField = docsWithField;
      this.valueCounts = valueCounts;
      this.vectors = vectors;
      this.dimension = dimension;
      valuesIter = vectors.iterator();
      valueCountsIter = valueCounts.iterator();
      docsWithFieldIter = docsWithField.iterator();

      buffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      binaryValue = new BytesRef(buffer.array());
      raBuffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      raBinaryValue = new BytesRef(raBuffer.array());

    }

    @Override
    public RandomAccessVectorValues randomAccess() {
      return new MultipleBufferedVectorValues(docsWithField, valueCounts, vectors, dimension);
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public long size() {
      return vectors.size();
    }

    @Override
    public int docValueCount() {
      return valueCount;
    }

    @Override
    public BytesRef nextBinaryValue() {
      buffer.asFloatBuffer().put(nextVectorValue());
      return binaryValue;
    }

    @Override
    public float[] nextVectorValue() {
      if (valueUpto == valueCount) {
        throw new IllegalStateException();
      }
      valueUpto++;
      return valuesIter.next();
    }

    @Override
    public BytesRef binaryValue(long targetOrd) {
      raBuffer.asFloatBuffer().put(vectors.get((int) targetOrd));
      return raBinaryValue;
    }

    @Override
    public float[] vectorValue(long targetOrd) {
      return vectors.get((int) targetOrd);
    }

    @Override
    public int docID() {
      return docsWithFieldIter.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      for (int i = valueUpto; i < valueCount; i++) {
        valuesIter.next();
      }

      int docID = docsWithFieldIter.nextDoc();
      if (docID != NO_MORE_DOCS) {
        valueCount = Math.toIntExact(valueCountsIter.next());
        valueUpto = 0;
      }
      return docID;
    }

    @Override
    public int advance(int target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return docsWithFieldIter.cost();
    }
  }
}
