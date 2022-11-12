from multiprocessing import Process, Value
import time

def add_100(number):
  for i in range(100):
    time.sleep(0.01)
    number.value += 1

if __name__ == "__main__":
  shared_num = Value('i', 0)
  print(f"Number at beginning is {shared_num.value}")

  p1 = Process(target=add_100, args=(shared_num,))
  p2 = Process(target=add_100, args=(shared_num,))

  p1.start()
  p2.start()

  p1.join()
  p2.join()

  print(f"Number at end is {shared_num.value}")
