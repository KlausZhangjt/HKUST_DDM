import multiprocessing
from multiprocessing import Process
import os

def do_something(id):
  print('Hello')
  print("World from",id)
  
if __name__ == '__main__':
  p1 = multiprocessing.Process(target=do_something, args=(1,))
  p2 = multiprocessing.Process(target=do_something, args=(2,))

  p2.start()
  p2.join()

  p1.start()
  p1.join()

