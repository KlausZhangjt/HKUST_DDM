from multiprocessing import Process, Pool
from math import sqrt, floor

def isprime(number):
  cap = floor(sqrt(number))+1
  for i in range(2,cap):
    if number % i == 0:
      return ''
      break
  
  return str(number)

if __name__ == '__main__':
  input = range(2,100)
  p = Pool()
  result = p.map(isprime, input)
  p.close()
  p.join()

  y = [];
  for x in result:
    if x:
      t = int(x)
      y.append(t)

  print(y[:])
