package org.apache.lucene.util.hnsw;

import java.util.concurrent.locks.ReentrantLock;

/**
 * A blocking bounded min heap that stores floats. The top element is the lowest value of the heap.
 *
 * <p>A primitive priority queue that maintains a partial ordering of its elements such that the
 * least element can always be found in constant time. Implementation is based on {@link
 * org.apache.lucene.util.LongHeap}
 *
 * @lucene.internal
 */
public final class BlockingFloatHeap {
  private final int maxSize;
  private final float[] heap;
  private final ReentrantLock lock;
  private int size;

  public BlockingFloatHeap(int maxSize) {
    this.maxSize = maxSize;
    this.heap = new float[maxSize + 1];
    this.lock = new ReentrantLock();
    this.size = 0;
  }

  /**
   * Inserts a value into this heap.
   *
   * <p>If the number of values would exceed the heap's maxSize, the least value is discarded
   *
   * @param value the value to add
   * @return {@code true} if the value was added to this heap {@code false} if the value was not
   *     added because the heap was full and the value is smaller than the top of the heap
   */
  public boolean offer(float value) {
    boolean valueAdded = true;

    lock.lock();
    try {
      if (size < maxSize) {
        push(value);
      } else {
        if (value >= heap[1]) {
          updateTop(value);
        } else {
          valueAdded = false;
        }
      }
    } finally {
      lock.unlock();
    }
    return valueAdded;
  }

  /**
   * Removes and returns the head of the heap
   *
   * @return the head of the heap, the smallest value
   * @throws IllegalStateException if the heap is empty
   */
  public float poll() {
    if (size > 0) {
      float result;

      lock.lock();
      try {
        result = heap[1]; // save first value
        heap[1] = heap[size]; // move last to first
        size--;
        downHeap(1); // adjust heap
      } finally {
        lock.unlock();
      }
      return result;
    } else {
      throw new IllegalStateException("The heap is empty");
    }
  }

  /**
   * Retrieves, but does not remove, the head of this heap.
   *
   * @return the head of the heap, the smallest value
   */
  public float peek() {
    lock.lock();
    try {
      return heap[1];
    } finally {
      lock.unlock();
    }
  }

  /**
   * Returns the number of elements in this heap.
   *
   * @return the number of elements in this heap
   */
  public int size() {
    lock.lock();
    try {
      return size;
    } finally {
      lock.unlock();
    }
  }

  private void push(float element) {
    size++;
    heap[size] = element;
    upHeap(size);
  }

  private float updateTop(float value) {
    heap[1] = value;
    downHeap(1);
    return heap[1];
  }

  private void downHeap(int i) {
    float value = heap[i]; // save top value
    int j = i << 1; // find smaller child
    int k = j + 1;
    if (k <= size && heap[k] < heap[j]) {
      j = k;
    }
    while (j <= size && heap[j] < value) {
      heap[i] = heap[j]; // shift up child
      i = j;
      j = i << 1;
      k = j + 1;
      if (k <= size && heap[k] < heap[j]) {
        j = k;
      }
    }
    heap[i] = value; // install saved value
  }

  private void upHeap(int origPos) {
    int i = origPos;
    float value = heap[i]; // save bottom value
    int j = i >>> 1;
    while (j > 0 && value < heap[j]) {
      heap[i] = heap[j]; // shift parents down
      i = j;
      j = j >>> 1;
    }
    heap[i] = value; // install saved value
  }
}
