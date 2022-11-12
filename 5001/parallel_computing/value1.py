from multiprocessing import Process,Value

def cal_square(v, v2):
  c = v.value
  v2.value = c*c
  v.value = 3.14

if __name__ == "__main__":
  v = Value('d',0)
  v2= Value('d',1)
  print(f"before: {v.value} {v2.value}")

  p = Process(target=cal_square, args=(v,v2,))

  p.start()
  p.join()

  print(f"after: {v.value} {v2.value}")
