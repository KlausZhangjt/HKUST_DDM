import numpy as np
import os
import time
from multiprocessing import Pool
from multiprocessing import Process


# %% question 2
def integral(i):
    h, ans = 1/10**4, 0
    x = np.linspace(1250*i*h, 1250*(i+1)*h, 1250, endpoint=False)
    ys = list(map(lambda y: h * 4/(1+y**2), x))
    return np.sum(ys)


start = time.time()
res = Pool(8).map(integral, range(8))
print("Estimated value of pi:", np.sum(res))
print("Time cost: ", time.time()-start)
