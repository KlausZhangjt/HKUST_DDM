# source from https://www.youtube.com/watch?v=2EbHSCvGFM0&t=7s
import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))

    print(f"VectorAdd took for {vector_add_time} seconds")

if __name__=='__main__':
    main()
