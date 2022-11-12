import time
import multiprocessing

start = time.perf_counter()

def func(t):
  time.sleep(t)
  print(f'slept {t} second...')

if __name__ == '__main__':

  processes = []

  for x in range(1,10):
    p = multiprocessing.Process(target=func, args=(x/10,))
    processes.append(p)
    p.start()


  finish = time.perf_counter()

  print(f'Finished in {round(finish-start,2)} second(s)')
