import multiprocessing
from multiprocessing import Process
import os

def do_something():
  print('Hello')
  print(f"World from {os.getpid()}")
  
if __name__ == '__main__':
  p1 = multiprocessing.Process(target=do_something)
  p2 = multiprocessing.Process(target=do_something)

  p1.start()
  p2.start()

  p1.join()
  p2.join()

