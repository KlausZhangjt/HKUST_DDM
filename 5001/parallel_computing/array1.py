from multiprocessing import Process, Array

def cal_square(numbers,result):
  for id,n in enumerate(numbers):
    print('square '+str(n*n))
    result[id] = n*n

if __name__ == "__main__":
  array = [2,3,5,7]
  result = Array('d',array)
  p1 = Process(target=cal_square, args=(array,result,))

  p1.start()
  p1.join()

  print('result ',result[:])
  print('done')
