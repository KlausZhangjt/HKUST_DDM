from multiprocessing import Process, Value, Lock
from random import random

def count_in(x, lock):
  a = random()
  b = random()
  if a*a + b*b < 1:
    with lock:
      x.value += 1

if __name__ == '__main__':
  x = Value('d',0)
  lock = Lock()
  processes = []

  for i in range(1000):
    p = Process(target=count_in, args=(x,lock,))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

  print(f"pi = {x.value/250}")
