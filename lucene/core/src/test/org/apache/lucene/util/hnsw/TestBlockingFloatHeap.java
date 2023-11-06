package org.apache.lucene.util.hnsw;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;

import java.util.concurrent.CountDownLatch;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestBlockingFloatHeap extends LuceneTestCase {

  public void testBasicOperations() {
    BlockingFloatHeap heap = new BlockingFloatHeap(3);
    heap.offer(2);
    heap.offer(4);
    heap.offer(1);
    heap.offer(3);
    assertEquals(3, heap.size());
    assertEquals(2, heap.peek(), 0);

    assertEquals(2, heap.poll(), 0);
    assertEquals(3, heap.poll(), 0);
    assertEquals(4, heap.poll(), 0);
    assertEquals(0, heap.size(), 0);
  }

  public void testBasicOperations2() {
    int size = atLeast(10);
    BlockingFloatHeap heap = new BlockingFloatHeap(size);
    double sum = 0, sum2 = 0;

    for (int i = 0; i < size; i++) {
      float next = random().nextFloat(100f);
      sum += next;
      heap.offer(next);
    }

    float last = Float.NEGATIVE_INFINITY;
    for (long i = 0; i < size; i++) {
      float next = heap.poll();
      assertTrue(next >= last);
      last = next;
      sum2 += last;
    }
    assertEquals(sum, sum2, 0.01);
  }

  public void testMultipleThreads() throws Exception {
    Thread[] threads = new Thread[randomIntBetween(3, 20)];
    final CountDownLatch latch = new CountDownLatch(1);
    BlockingFloatHeap globalHeap = new BlockingFloatHeap(1);

    for (int i = 0; i < threads.length; i++) {
      threads[i] =
          new Thread(
            () -> {
              try {
                latch.await();
                int numIterations = randomIntBetween(10, 100);
                float bottomValue = 0;

                while (numIterations-- > 0) {
                  bottomValue += randomIntBetween(0, 5);
                  globalHeap.offer(bottomValue);
                  Thread.sleep(randomIntBetween(0, 50));

                  float globalBottomValue = globalHeap.peek();
                  assertTrue(globalBottomValue >= bottomValue);
                  bottomValue = globalBottomValue;
                }
              } catch (Exception e) {
                throw new RuntimeException(e);
              }
            });
      threads[i].start();
    }

    latch.countDown();
    for (Thread t : threads) {
      t.join();
    }
  }
}
