import numpy as np
import os
import time
from multiprocessing import Pool
from multiprocessing import Process


# %% question 3
def calculate_pi(n):
    np.random.seed(77)
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = (x ** 2 + y ** 2 <= 1)
    return np.sum(z)


N = 10**5
procs = os.cpu_count()
lis = [N//procs] * procs
start = time.time()
res = Pool(processes=procs).map(func=calculate_pi, iterable=lis)
pi_estimate = sum(res) * 4 / N
print("Estimated value of pi:", pi_estimate)
print("Time cost: ", time.time()-start)
