import time

start = time.perf_counter()

def func():
  time.sleep(1)
  print('slept 1 second...')

func()

finish = time.perf_counter()

print(f'Finished in {round(finish-start,2)} second(s)')
