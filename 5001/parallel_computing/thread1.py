import threading
import time

def cal_square(numbers):
  print("calculate square numbers")
  for n in numbers:
    time.sleep(0.2)
    print('square:',n*n)

def cal_cube(numbers):
  print("calculate cube numbers")
  for n in numbers:
    time.sleep(0.2)
    print('cube:',n*n*n)

if __name__ == '__main__':
  array = [2,4,6,8,10]

  t = time.time()
  t1 = threading.Thread(target=cal_square,args=(array,))
  t2 = threading.Thread(target=cal_cube,args=(array,))

  t1.start()
  t2.start()

  t1.join()
  t2.join()

  print('time elapsed:',time.time()-t)
