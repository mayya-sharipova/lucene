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

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;
import java.util.Random;

/**
 * A reader of vector values that samples a subset of the vectors.
 */
public class SampleReader implements RandomAccessVectorValues.Floats {
    private final RandomAccessVectorValues.Floats origin;
    private final int numSamples;
    private final int[] samples;

    SampleReader(RandomAccessVectorValues.Floats origin, int numSamples, long seed) {
      this.origin = origin;
      this.numSamples = numSamples;
      this.samples = reservoirSample(origin.size(), numSamples, seed);
    }

    @Override
    public int size() {
      return numSamples;
    }

    @Override
    public int dimension() {
      return origin.dimension();
    }

    @Override
    public Floats copy() throws IOException {
      throw new IllegalStateException("Not supported");
    }

    @Override
    public IndexInput getSlice() {
      throw new IllegalStateException("Not supported");
    }

    @Override
    public float[] vectorValue(int targetOrd) throws IOException {
      return origin.vectorValue(samples[targetOrd]);
    }

    @Override
    public int getVectorByteLength() {
      return origin.getVectorByteLength();
    }

    @Override
    public int ordToDoc(int ord) {
      throw new IllegalStateException("Not supported");
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      throw new IllegalStateException("Not supported");
    }


  /**
   * Sample k elements from n elements according to reservoir sampling algorithm.
   * @param n number of elements
   * @param k number of samples
   * @param seed random seed
   * @return array of k samples
   */
  public static int[] reservoirSample(int n, int k, long seed) {
    Random rnd = new Random(seed);
    int[] reservoir = new int[k];
    for (int i = 0; i < k; i++) {
      reservoir[i] = i;
    }
    for (int i = k; i < n; i++) {
      int j = rnd.nextInt(i + 1);
      if (j < k) {
        reservoir[j] = i;
      }
    }
    return reservoir;
  }
}
