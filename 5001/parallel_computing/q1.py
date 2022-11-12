from multiprocessing import Process, Queue

def cal_square(numbers, q):
  for n in numbers:
    q.put(n*n)

#  print('inside process ',str(result))

if __name__ == "__main__":
  numbers = [2,3,5,7]
  q = Queue()
  p = Process(target=cal_square, args=(numbers,q,))

  p.start()
  p.join()

  while q.empty() is False:
    print(q.get())

