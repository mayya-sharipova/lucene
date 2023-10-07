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
package org.apache.lucene.misc.store;

import java.io.IOException;
import java.nio.file.Path;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FlushInfo;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MergeInfo;
import org.apache.lucene.tests.store.BaseDirectoryTestCase;
import org.apache.lucene.util.VectorUtil;

public class TestByteWritesTrackingDirectoryWrapper extends BaseDirectoryTestCase {

  public void testEmptyDir() throws Exception {
    ByteWritesTrackingDirectoryWrapper dir =
        new ByteWritesTrackingDirectoryWrapper(new ByteBuffersDirectory());
    assertEquals(0, dir.getFlushedBytes());
    assertEquals(0, dir.getMergedBytes());
  }

  public void testRandomOutput() throws Exception {
    ByteWritesTrackingDirectoryWrapper dir =
        new ByteWritesTrackingDirectoryWrapper(new ByteBuffersDirectory());

    int expectedFlushBytes = random().nextInt(100);
    int expectedMergeBytes = random().nextInt(100);

    IndexOutput output =
        dir.createOutput("write", new IOContext(new FlushInfo(10, expectedFlushBytes)));
    byte[] flushBytesArr = new byte[expectedFlushBytes];
    for (int i = 0; i < expectedFlushBytes; i++) {
      flushBytesArr[i] = (byte) random().nextInt(127);
    }
    output.writeBytes(flushBytesArr, flushBytesArr.length);
    assertEquals(0, dir.getFlushedBytes());
    assertEquals(0, dir.getMergedBytes());
    output.close();

    // now merge bytes
    output =
        dir.createOutput("merge", new IOContext(new MergeInfo(10, expectedMergeBytes, false, 2)));
    byte[] mergeBytesArr = new byte[expectedMergeBytes];
    for (int i = 0; i < expectedMergeBytes; i++) {
      mergeBytesArr[i] = (byte) random().nextInt(127);
    }
    output.writeBytes(mergeBytesArr, mergeBytesArr.length);
    assertEquals(expectedFlushBytes, dir.getFlushedBytes());
    assertEquals(0, dir.getMergedBytes());
    output.close();

    assertEquals(expectedFlushBytes, dir.getFlushedBytes());
    assertEquals(expectedMergeBytes, dir.getMergedBytes());
  }

  public void testRandomTempOutput() throws Exception {
    ByteWritesTrackingDirectoryWrapper dir =
        new ByteWritesTrackingDirectoryWrapper(new ByteBuffersDirectory(), true);

    int expectedFlushBytes = random().nextInt(100);
    int expectedMergeBytes = random().nextInt(100);

    IndexOutput output =
        dir.createTempOutput("temp", "write", new IOContext(new FlushInfo(10, expectedFlushBytes)));
    byte[] flushBytesArr = new byte[expectedFlushBytes];
    for (int i = 0; i < expectedFlushBytes; i++) {
      flushBytesArr[i] = (byte) random().nextInt(127);
    }
    output.writeBytes(flushBytesArr, flushBytesArr.length);
    assertEquals(0, dir.getFlushedBytes());
    assertEquals(0, dir.getMergedBytes());
    output.close();

    // now merge bytes
    output =
        dir.createTempOutput(
            "temp", "merge", new IOContext(new MergeInfo(10, expectedMergeBytes, false, 2)));
    byte[] mergeBytesArr = new byte[expectedMergeBytes];
    for (int i = 0; i < expectedMergeBytes; i++) {
      mergeBytesArr[i] = (byte) random().nextInt(127);
    }
    output.writeBytes(mergeBytesArr, mergeBytesArr.length);
    assertEquals(expectedFlushBytes, dir.getFlushedBytes());
    assertEquals(0, dir.getMergedBytes());
    output.close();

    assertEquals(expectedFlushBytes, dir.getFlushedBytes());
    assertEquals(expectedMergeBytes, dir.getMergedBytes());
  }

  public void testVectors() throws IOException {
    final int numDocs = 10_000;
    final int dim = 100;
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    IndexWriterConfig iwc = new IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.CREATE);

    try (ByteWritesTrackingDirectoryWrapper dir =
        new ByteWritesTrackingDirectoryWrapper(newDirectory(), false)) {
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          float[] v = randomVector(dim);
          doc.add(new KnnFloatVectorField("knn_vector", v, similarityFunction));
          w.addDocument(doc);
          if (i % 300 == 0) {
            w.flush();
          }
          if (i % 1000 == 0) {
            w.forceMerge(1);
          }
        }
      }
      System.out.println(
          dir.getFlushedBytes()
              + " "
              + dir.getMergedBytes()
              + " "
              + dir.getMergedBytes() * 1.0 / dir.getFlushedBytes());
    }
  }

  private float[] randomVector(int dim) {
    assert dim > 0;
    float[] v = new float[dim];
    double squareSum = 0.0;
    // keep generating until we don't get a zero-length vector
    while (squareSum == 0.0) {
      squareSum = 0.0;
      for (int i = 0; i < dim; i++) {
        v[i] = random().nextFloat();
        squareSum += v[i] * v[i];
      }
    }
    VectorUtil.l2normalize(v);
    return v;
  }

  @Override
  protected Directory getDirectory(Path path) throws IOException {
    return new ByteWritesTrackingDirectoryWrapper(new ByteBuffersDirectory());
  }
}
