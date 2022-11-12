import multiprocessing

result=[]

def cal_square(numbers):
  global result
  for n in numbers:
    print('square '+str(n*n))
    result.append(n*n)

if __name__ == "__main__":
  array = [2,3,5,7]
  p1 = multiprocessing.Process(target=cal_square, args=(array,))

  p1.start()
  p1.join()

  print('result '+str(result))
  print('done')
