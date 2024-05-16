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

import static org.apache.lucene.sandbox.codecs.pq.Benchmark.FILE_BYTESIZE;
import static org.apache.lucene.sandbox.codecs.pq.Benchmark.FILE_VECTOR_OFFSET;
import static org.apache.lucene.sandbox.codecs.pq.SampleReader.createSampleReader;
import static org.apache.lucene.util.hnsw.HnswGraphBuilder.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.util.hnsw.HnswGraphBuilder.DEFAULT_MAX_CONN;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.codecs.pq.PQUtils.PQScorerSupplier;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/** ProductQuantizer is a quantization algorithm that quantizes a vector into a byte array. */
@SuppressForbidden(reason = "System.out required: command line tool")
public class ProductQuantizer {
  // The number of docs to sample to compute the coarse clustering.
  static final int COARSE_CLUSTERING_SAMPLE_SIZE = 128 * 1024;
  // The expected number of docs for each cluster when computing the coarse clustering.
  // This is used to compute the number of clusters to use.
  static final int COARSE_CLUSTERING_DOCS_PER_CLUSTER = 100_000_000; // 256 * 1024;
  // The number of iterations to run k-means for when computing the coarse clustering.
  static final int COARSE_CLUSTERING_KMEANS_ITR = 10; // 10;
  // The number of random restarts of clustering to use when computing the coarse clustering.
  static final int COARSE_CLUSTERING_KMEANS_RESTARTS = 5; // 5;
  // The number of centroids in each codebook
  static final int BOOK_SIZE = 256;
  // The number of random restarts of clustering to use when constructing the codebooks.
  static final int BOOK_CONSTRUCTION_K_MEANS_RESTARTS = 5; // 5;
  // The number of iterations to run k-means for when constructing the codebooks.
  static final int BOOK_CONSTRUCTION_K_MEANS_ITR = 4;

  // The number of closest coarse centroids to consider during query
  static final int W = 5;

  private final float[][] coarseCentroids;
  private final int[][] docsByCentroid;
  private final int numBooks;
  private final int bookDim;
  private final float[][][][] codebooks;
  private final float[][][] norms;
  private final byte[][] docCodes;
  private final VectorSimilarityFunction vectorSimFunction;
  private final OnHeapHnswGraph hnswGraph;
  private final PQScorerSupplier scorerSupplier;

  private ProductQuantizer(
      int numBooks,
      int bookDim,
      float[][] coarseCentroids,
      int[][] docsByCentroid,
      byte[][] docCodes,
      float[][][][] codebooks,
      float[][][] norms,
      VectorSimilarityFunction vectorSimFunction,
      OnHeapHnswGraph hnswGraph,
      PQScorerSupplier scorerSupplier) {
    this.numBooks = numBooks;
    this.bookDim = bookDim;
    this.coarseCentroids = coarseCentroids;
    this.docsByCentroid = docsByCentroid;
    this.codebooks = codebooks;
    this.norms = norms;
    this.docCodes = docCodes;
    this.vectorSimFunction = vectorSimFunction;
    this.hnswGraph = hnswGraph;
    this.scorerSupplier = scorerSupplier;
  }

  public static ProductQuantizer create(
      RandomAccessVectorValues.Floats vectors,
      int numBooks,
      VectorSimilarityFunction vectorSimFunction,
      long seed,
      boolean useHnsw,
      float anisotropicThreshold)
      throws IOException {

    // **** 1. Coarse clustering documents
    long start = System.nanoTime();
    final int dims = vectors.dimension();
    if (dims % numBooks != 0) {
      throw new IllegalArgumentException(
          "The number of dimensions must be divisible by the number of books");
    }

    // for each document which coarse centroid it belongs to,
    // we expect < 32767 coarse centroids
    short[] docCentroids = new short[vectors.size()];
    float[][] coarseCentroids =
        coarseClustering(
            vectors, vectorSimFunction == VectorSimilarityFunction.COSINE, docCentroids, seed);

    int numCoarseCentroids = coarseCentroids.length;
    int[][] docsByCentroids = breakDocsByCentroids(numCoarseCentroids, docCentroids);
    long elapsed = System.nanoTime() - start;
    System.out.format(
        "Coarse clustering took: %d ms, sample size: %d docs, centroids:%d %n",
        TimeUnit.NANOSECONDS.toMillis(elapsed), COARSE_CLUSTERING_SAMPLE_SIZE, numCoarseCentroids);

    // **** 2. Train sub-quantizers for each coarse centroid separately
    start = System.nanoTime();
    int bookDim = dims / numBooks;
    int numSamplesPerCluster = vectors.dimension() * 64;
    float[][][][] codebooks = new float[numCoarseCentroids][numBooks][][];
    for (int c = 0; c < numCoarseCentroids; c++) {
      // sample from the docs belonging the cluster c
      int[] samples =
          SampleReader.reservoirSampleFromArray(docsByCentroids[c], numSamplesPerCluster, seed);

      for (int b = 0; b < numBooks; b++) {
        RandomAccessVectorValues.Floats subVectors =
            initializeSubspaceReader(vectors, samples, b, bookDim, coarseCentroids[c]);
        KMeans kmeans = new KMeans(subVectors, BOOK_SIZE, seed);
        codebooks[c][b] =
            kmeans.computeCentroids(
                BOOK_CONSTRUCTION_K_MEANS_RESTARTS, BOOK_CONSTRUCTION_K_MEANS_ITR, null);
      }
    }
    float[][][] norms = null;
    // TODO: investigate the degraded performance with norms table
//    if (vectorSimFunction == VectorSimilarityFunction.COSINE) {
//      norms = new float[numCoarseCentroids][numBooks * BOOK_SIZE][2];
//      // Build a table of the norms of each codebook centre and the dot product
//      // of each cluster centre with each codebook centre.
//      float[] ci;
//      float[] coarseCentroidProj;
//      for (int c = 0; c < numCoarseCentroids; c++) {
//        for (int b = 0; b < numBooks; b++) {
//          coarseCentroidProj =
//              Arrays.copyOfRange(coarseCentroids[c], b * bookDim, (b + 1) * bookDim);
//          int offset = b * BOOK_SIZE;
//          for (int i = 0; i < BOOK_SIZE; i++) {
//            ci = codebooks[c][b][i];
//            norms[c][offset + i][0] = VectorUtil.dotProduct(ci, coarseCentroidProj);
//            norms[c][offset + i][1] = VectorUtil.dotProduct(ci, ci);
//          }
//        }
//      }
//    }

    elapsed = System.nanoTime() - start;
    System.out.format(
        "Product quantizer took:  %d ms, sample size:%d, books: %d %n",
        TimeUnit.NANOSECONDS.toMillis(elapsed), numSamplesPerCluster, numBooks);

    //  **** 3. encode vector values
    start = System.nanoTime();
    final byte[][] docCodes =
        anisotropicThreshold > 0
            ? encodeAnisotropicDocs(
                vectors, coarseCentroids, codebooks, docCentroids, numBooks, anisotropicThreshold)
            : encodeDocs(vectors, coarseCentroids, codebooks, docCentroids, numBooks);
    elapsed = System.nanoTime() - start;
    System.out.format(
        "Encoding all docs took: %d ms, docs: %d %n",
        TimeUnit.NANOSECONDS.toMillis(elapsed), vectors.size());

    //  **** 4. Build graph if needed
    start = System.nanoTime();
    OnHeapHnswGraph hnswGraph = null;
    PQScorerSupplier scorerSupplier = null;
    if (useHnsw & numCoarseCentroids == 1) {
      PQUtils.QuantizedVectors quantizedVectors = new PQUtils.QuantizedVectors(docCodes);
      scorerSupplier =
          new PQUtils.PQScorerSupplier(quantizedVectors, vectorSimFunction, codebooks[0]);
      HnswGraphBuilder builder =
          HnswGraphBuilder.create(
              scorerSupplier,
              DEFAULT_MAX_CONN,
              DEFAULT_BEAM_WIDTH,
              HnswGraphBuilder.randSeed,
              vectors.size());
      hnswGraph = builder.build(vectors.size());
      elapsed = System.nanoTime() - start;
      System.out.format("Graph build took: %d ms%n", TimeUnit.NANOSECONDS.toMillis(elapsed));
    }
    return new ProductQuantizer(
        numBooks,
        bookDim,
        coarseCentroids,
        docsByCentroids,
        docCodes,
        codebooks,
        norms,
        vectorSimFunction,
        hnswGraph,
        scorerSupplier);
  }

  private static float[][] coarseClustering(
      RandomAccessVectorValues.Floats vectors, boolean normalized, short[] docCentroids, long seed)
      throws IOException {
    // for cosine distance, use spherical k-means; normalize centers
    Consumer<float[][]> ifSphericalKMeansNormalize =
        (centers) -> {
          if (normalized) {
            for (int i = 0; i < centers.length; i++) {
              VectorUtil.l2normalize(centers[i]);
            }
          }
        };

    float[][] coarseCentroids;
    int numClusters = Math.max(1, vectors.size() / COARSE_CLUSTERING_DOCS_PER_CLUSTER);
    if (numClusters > 1) {
      RandomAccessVectorValues.Floats sampleVectors =
          createSampleReader(vectors, COARSE_CLUSTERING_SAMPLE_SIZE, seed);
      KMeans kmeans = new KMeans(sampleVectors, numClusters, seed);
      coarseCentroids =
          kmeans.computeCentroids(
              COARSE_CLUSTERING_KMEANS_RESTARTS,
              COARSE_CLUSTERING_KMEANS_ITR,
              ifSphericalKMeansNormalize);
    } else {
      coarseCentroids = new float[1][vectors.dimension()];
    }

    // Assign each document to the nearest centroid and update the centres.
    KMeans.runKMeansStep(vectors, coarseCentroids, docCentroids, true, ifSphericalKMeansNormalize);
    return coarseCentroids;
  }

  private static int[][] breakDocsByCentroids(int numCentroids, short[] docCentroids) {
    int[][] docsByCentroids = new int[numCentroids][];
    if (numCentroids == 1) {
      docsByCentroids[0] = new int[docCentroids.length];
      for (int doc = 0; doc < docCentroids.length; doc++) {
        docsByCentroids[0][doc] = doc;
      }
      return docsByCentroids;
    }

    int[] centroidSizes = new int[numCentroids];
    for (int c : docCentroids) {
      centroidSizes[c]++;
    }
    for (int i = 0; i < numCentroids; i++) {
      docsByCentroids[i] = new int[centroidSizes[i]];
    }
    for (int c = 0; c < numCentroids; c++) {
      centroidSizes[c] = 0;
    }
    for (int doc = 0; doc < docCentroids.length; doc++) {
      short c = docCentroids[doc];
      docsByCentroids[c][centroidSizes[c]] = doc;
      centroidSizes[c]++;
    }
    return docsByCentroids;
  }

  private static byte[][] encodeDocs(
      RandomAccessVectorValues.Floats vectors,
      float[][] coarseCentroids,
      float[][][][] codebooks,
      short[] docCentroids,
      int numBooks)
      throws IOException {
    final byte[][] docCodes = new byte[vectors.size()][];
    int dims = vectors.dimension();
    int bookDim = dims / numBooks;
    float[] subVector = new float[bookDim];
    for (int doc = 0; doc < vectors.size(); doc++) {
      float[] coarseCentroid = coarseCentroids[docCentroids[doc]];
      float[] residual = vectors.vectorValue(doc);
      for (int dim = 0; dim < dims; dim++) {
        residual[dim] -= coarseCentroid[dim];
      }
      docCodes[doc] = new byte[numBooks];
      for (int b = 0; b < numBooks; b++) {
        int startIndex = b * bookDim;
        System.arraycopy(residual, startIndex, subVector, 0, bookDim);
        docCodes[doc][b] = encode(subVector, codebooks[docCentroids[doc]][b]);
      }
    }
    return docCodes;
  }

  private static byte encode(float[] subVector, float[][] subCodebook) {
    int centroidIndex = -1;
    float minDistance = Float.MAX_VALUE;
    for (int c = 0; c < BOOK_SIZE; c++) {
      float dist = VectorUtil.squareDistance(subCodebook[c], subVector);
      if (dist < minDistance) {
        minDistance = dist;
        centroidIndex = c;
      }
    }
    return (byte) centroidIndex;
  }

  private static byte[][] encodeAnisotropicDocs(
      RandomAccessVectorValues.Floats vectors,
      float[][] coarseCentroids,
      float[][][][] codebooks,
      short[] docCentroids,
      int numBooks,
      float anisotropicThreshold)
      throws IOException {
    final byte[][] docCodes = new byte[vectors.size()][];
    int dims = vectors.dimension();
    int bookDim = dims / numBooks;

    float scale = anisotropicThreshold * anisotropicThreshold;
    scale = scale / (1.0F - scale) * (dims - 1.0F);

    float[] subVector = new float[bookDim];
    for (int doc = 0; doc < vectors.size(); doc++) {
      float[] coarseCentroid = coarseCentroids[docCentroids[doc]];
      float[] residual = vectors.vectorValue(doc);
      for (int dim = 0; dim < dims; dim++) {
        residual[dim] -= coarseCentroid[dim];
      }
      float norm2 = VectorUtil.dotProduct(residual, residual);
      docCodes[doc] = new byte[numBooks];
      for (int b = 0; b < numBooks; b++) {
        int startIndex = b * bookDim;
        System.arraycopy(residual, startIndex, subVector, 0, bookDim);
        docCodes[doc][b] =
            encodeAnisotropic(subVector, codebooks[docCentroids[doc]][b], scale, norm2);
      }
    }
    return docCodes;
  }

  private static byte encodeAnisotropic(
      float[] subVector, float[][] subCodebook, float scale, float norm2) {
    int centroidIndex = -1;
    float minDistance = Float.MAX_VALUE;
    for (int c = 0; c < BOOK_SIZE; c++) {
      float parallelDist = 0;
      for (int dim = 0; dim < subVector.length; dim++) {
        parallelDist += subVector[dim] * (subCodebook[c][dim] - subVector[dim]);
      }
      float orthogonalDist = VectorUtil.squareDistance(subCodebook[c], subVector);
      float dist = orthogonalDist + (scale - 1.0F) * parallelDist * parallelDist / norm2;
      if (dist < minDistance) {
        minDistance = dist;
        centroidIndex = c;
      }
    }
    return (byte) centroidIndex;
  }

  /**
   * Initialize the sub-quantizer reader for a given sub-quantizer index.
   *
   * <p>For the last sub-space if vector dimensions are not divisible by subSpaceDim, fill the
   * remaining dimensions with zeros.
   */
  private static RandomAccessVectorValues.Floats initializeSubspaceReader(
      RandomAccessVectorValues.Floats vectors,
      int[] samples,
      int subspaceIdx,
      int bookDim,
      float[] coarseCentroid)
      throws IOException {
    int subspaceOffset = subspaceIdx * bookDim;

    RandomAccessVectorValues.Floats subVectorsReader =
        new VectorsReaderWithOffset(
            vectors.getSlice(),
            vectors.size(),
            bookDim,
            FILE_BYTESIZE,
            FILE_VECTOR_OFFSET + subspaceOffset * Float.BYTES);
    final float[][] subVectors = new float[samples.length][bookDim];
    for (int i = 0; i < samples.length; i++) {
      float[] vector = subVectorsReader.vectorValue(samples[i]);
      // compute residual from the coarse centroid
      for (int dim = 0; dim < bookDim; dim++) {
        subVectors[i][dim] = vector[dim] - coarseCentroid[subspaceOffset + dim];
      }
    }

    return new RandomAccessVectorValues.Floats() {
      @Override
      public Floats copy() {
        throw new UnsupportedOperationException();
      }

      @Override
      public float[] vectorValue(int targetOrd) {
        return subVectors[targetOrd];
      }

      @Override
      public int size() {
        return samples.length;
      }

      @Override
      public int dimension() {
        return bookDim;
      }
    };
  }

  public int[] getTopDocs(float[] query, int topK) throws IOException {
    NeighborQueue queue = new NeighborQueue(topK, false);
    if (hnswGraph != null) {
      // quantize query
      assert coarseCentroids.length == 1 : "number of coarse centroids must be 1";
      byte[] quantizedQuery = new byte[numBooks];
      float[] residual = query;
      for (int dim = 0; dim < query.length; dim++) {
        residual[dim] -= coarseCentroids[0][dim];
      }
      float[] subVector = new float[bookDim];
      for (int b = 0; b < numBooks; b++) {
        int startIndex = b * bookDim;
        System.arraycopy(residual, startIndex, subVector, 0, bookDim);
        quantizedQuery[b] = encode(subVector, codebooks[0][b]);
      }

      RandomVectorScorer scorer = scorerSupplier.scorer(quantizedQuery);
      KnnCollector knnCollector =
          HnswGraphSearcher.search(scorer, topK, hnswGraph, null, Integer.MAX_VALUE);
      ScoreDoc[] scoreDocs = knnCollector.topDocs().scoreDocs;
      int[] topDocs = new int[scoreDocs.length];
      for (int i = 0; i < scoreDocs.length; i++) {
        topDocs[i] = scoreDocs[i].doc;
      }
      return topDocs;
    }

    ProductQuantizer.DistanceRunner[] runners = createDistanceRunners(query);
    for (ProductQuantizer.DistanceRunner runner : runners) {
      PQUtils.QuantizedDocs docs =
          new PQUtils.QuantizedDocs(docsByCentroid[runner.getCoarseCentroid()], docCodes);
      for (int doc = docs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = docs.nextDoc()) {
        float res = runner.similarity(docs.pqCode());
        queue.insertWithOverflow(doc, res);
      }
    }
    int[] topDocs = new int[topK];
    for (int k = topK - 1; k >= 0; k--) {
      topDocs[k] = queue.topNode();
      queue.pop();
    }
    return topDocs;
  }

  public DistanceRunner[] createDistanceRunners(float[] qVector) {
    // TODO: substitute cosine with dot_product if norms table is used
    // Find the closest W coarse centroids to the query vector
    int[] topCoarseCentroids;
    float[] topSims;
    if (coarseCentroids.length == 1) {
      topCoarseCentroids = new int[] {0};
      topSims = new float[] {0f};
    } else {
      NeighborQueue q = new NeighborQueue(W, false);
      for (int c = 0; c < coarseCentroids.length; c++) {
        float sim = vectorSimFunction.compare(coarseCentroids[c], qVector);
        q.insertWithOverflow(c, sim);
      }
      final int w = q.size();
      topCoarseCentroids = new int[w];
      topSims = new float[w];
      for (int k = w - 1; k >= 0; k--) {
        topCoarseCentroids[k] = q.topNode();
        topSims[k] = q.topScore();
        q.pop();
      }
    }

    // For each of the w closest coarse centroids,
    // compute the distances to its sub-quantizer centroids
    float[] querySubVector = new float[bookDim];
    DistanceRunner[] distanceRunners = new DistanceRunner[topCoarseCentroids.length];
    for (int i = 0; i < topCoarseCentroids.length; i++) {
      int cIdx = topCoarseCentroids[i];
      float[] coarseCentroid = coarseCentroids[cIdx];
      float[] simTable = new float[numBooks * BOOK_SIZE];
      for (int b = 0; b < numBooks; b++) {
        // prepare the sub-vector for the sub-quantizer: compute residual
        int offset = b * bookDim;
        for (int dim = 0; dim < bookDim; dim++) {
          querySubVector[dim] = qVector[offset + dim] - coarseCentroid[offset + dim];
        }
        offset = b * BOOK_SIZE;
        for (int j = 0; j < BOOK_SIZE; j++) {
          simTable[offset + j] = vectorSimFunction.compare(codebooks[cIdx][b][j], querySubVector);
        }
      }
      distanceRunners[i] =
          new DistanceRunner(cIdx, topSims[i], simTable, norms == null ? null : norms[cIdx]);
    }
    return distanceRunners;
  }

  /**
   * DistanceRunner is a helper class to compute the distance between a query vector and a PQ code.
   */
  private static class DistanceRunner {
    private final int coarseCentroid;
    // query similarity with a coarse centroid
    private final float coarseCentroidSim;
    public final float[] simTable;
    private final float[][] normsTable;

    private DistanceRunner(
        int coarseCentroid, float coarseCentroidSim, float[] simTable, float[][] normsTable) {
      this.coarseCentroid = coarseCentroid;
      this.coarseCentroidSim = coarseCentroidSim;
      this.simTable = simTable;
      this.normsTable = normsTable;
    }

    public int getCoarseCentroid() {
      return coarseCentroid;
    }

    public float similarity(byte[] pqCode) {
      float sim = coarseCentroidSim;
      if (pqCode.length >= 8) {
        sim += similarityUnrolled(pqCode);
      } else {
        sim += similaritySimple(pqCode);
      }
      // If the index is normalized then we need to normalize the distance
      // by the norm of the document vector. The norm of the document vector
      // is |c + r| = (|c|^2 + 2 c^t r + |r|^2)^(1/2). By construction the
      // centres are normalized so this simplifies to (1 + 2 c^t r + |r|)^(1/2).
      // We can look up the dot product between the centre and the residual
      // and the norm of the residual in the normsTable.
      if (normsTable != null) {
        if (pqCode.length >= 800) {
          sim = normalizeSimUnrolled(sim, pqCode);
        } else {
          sim = normalizeSimSimple(sim, pqCode);
        }
      }
      return sim;
    }

    private float similaritySimple(byte[] pqCode) {
      float score = 0f;
      for (int i = 0; i < pqCode.length; i++) {
        score += simTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)];
      }
      return score;
    }

    private float similarityUnrolled(byte[] pqCode) {
      float res = 0f;
      int i;
      for (i = 0; i < pqCode.length % 8; i++) {
        res += simTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)];
      }
      if (pqCode.length < 8) {
        return res;
      }
      for (; i + 7 < pqCode.length; i += 8) {
        res += simTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)];
        res += simTable[(i + 1) * BOOK_SIZE + ((int) pqCode[(i + 1)] & 0xFF)];
        res += simTable[(i + 2) * BOOK_SIZE + ((int) pqCode[(i + 2)] & 0xFF)];
        res += simTable[(i + 3) * BOOK_SIZE + ((int) pqCode[(i + 3)] & 0xFF)];
        res += simTable[(i + 4) * BOOK_SIZE + ((int) pqCode[(i + 4)] & 0xFF)];
        res += simTable[(i + 5) * BOOK_SIZE + ((int) pqCode[(i + 5)] & 0xFF)];
        res += simTable[(i + 6) * BOOK_SIZE + ((int) pqCode[(i + 6)] & 0xFF)];
        res += simTable[(i + 7) * BOOK_SIZE + ((int) pqCode[(i + 7)] & 0xFF)];
      }
      return res;
    }

    private float normalizeSimSimple(float sim, byte[] pqCode) {
      float cdotr = 0;
      float rnorm2 = 0;
      for (int i = 0; i < pqCode.length; i++) {
        int idx = i * BOOK_SIZE + (int) pqCode[i] & 0xFF;
        cdotr += normsTable[idx][0];
        rnorm2 += normsTable[idx][1];
      }
      sim = sim / (float) Math.sqrt(1f + 2f * cdotr + rnorm2);
      return sim;
    }

    private float normalizeSimUnrolled(float sim, byte[] pqCode) {
      float cdotr = 0;
      float rnorm2 = 0;
      int i;
      for (i = 0; i < pqCode.length % 8; i++) {
        cdotr += normsTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)][0];
        rnorm2 += normsTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)][1];
      }
      for (; i + 7 < pqCode.length; i += 8) {
        cdotr += normsTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)][0];
        rnorm2 += normsTable[i * BOOK_SIZE + ((int) pqCode[i] & 0xFF)][1];
        cdotr += normsTable[(i + 1) * BOOK_SIZE + ((int) pqCode[i + 1] & 0xFF)][0];
        rnorm2 += normsTable[(i + 1) * BOOK_SIZE + ((int) pqCode[i + 1] & 0xFF)][1];
        cdotr += normsTable[(i + 2) * BOOK_SIZE + ((int) pqCode[i + 2] & 0xFF)][0];
        rnorm2 += normsTable[(i + 2) * BOOK_SIZE + ((int) pqCode[i + 2] & 0xFF)][1];
        cdotr += normsTable[(i + 3) * BOOK_SIZE + ((int) pqCode[i + 3] & 0xFF)][0];
        rnorm2 += normsTable[(i + 3) * BOOK_SIZE + ((int) pqCode[i + 3] & 0xFF)][1];
        cdotr += normsTable[(i + 4) * BOOK_SIZE + ((int) pqCode[i + 4] & 0xFF)][0];
        rnorm2 += normsTable[(i + 4) * BOOK_SIZE + ((int) pqCode[i + 4] & 0xFF)][1];
        cdotr += normsTable[(i + 5) * BOOK_SIZE + ((int) pqCode[i + 5] & 0xFF)][0];
        rnorm2 += normsTable[(i + 5) * BOOK_SIZE + ((int) pqCode[i + 5] & 0xFF)][1];
        cdotr += normsTable[(i + 6) * BOOK_SIZE + ((int) pqCode[i + 6] & 0xFF)][0];
        rnorm2 += normsTable[(i + 6) * BOOK_SIZE + ((int) pqCode[i + 6] & 0xFF)][1];
        cdotr += normsTable[(i + 7) * BOOK_SIZE + ((int) pqCode[i + 7] & 0xFF)][0];
        rnorm2 += normsTable[(i + 7) * BOOK_SIZE + ((int) pqCode[i + 7] & 0xFF)][1];
      }
      sim = sim / (float) Math.sqrt(1f + 2f * cdotr + rnorm2);
      return sim;
    }
  }
}
