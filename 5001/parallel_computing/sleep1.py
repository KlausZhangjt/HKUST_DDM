import time
import multiprocessing

start = time.perf_counter()

def func():
  time.sleep(1)
  print('slept 1 second...')

p1 = multiprocessing.Process(target=func, args=())
p2 = multiprocessing.Process(target=func, args=())

finish = time.perf_counter()

print(f'Finished in {round(finish-start,2)} second(s)')
