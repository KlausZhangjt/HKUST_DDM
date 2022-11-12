# source from https://www.youtube.com/watch?v=2EbHSCvGFM0&t=7s
import numpy as np
import time

def VectorAdd(a, b, c):
    for i in range(a.size):
      c[i] = a[i] + b[i]

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)

    start = time.time()
    VectorAdd(A, B, C)
    vector_add_time = time.time() - start

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))

    print(f"VectorAdd0 took for {vector_add_time} seconds")

if __name__=='__main__':
    main()
