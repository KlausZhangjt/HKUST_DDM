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

array = [2,4,6,8,10]

t = time.time()
cal_square(array)
cal_cube(array)

print('time elapsed:',time.time()-t)
