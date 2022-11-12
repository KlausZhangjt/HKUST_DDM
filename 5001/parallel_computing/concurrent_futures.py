# based on https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/MultiProcessing/multiprocessing-demo.py
import concurrent.futures
import time

def func1():
  print('done')

def func2(t):
  time.sleep(t)
  print(f'slept {t} second(s)')

def func3(t):
  time.sleep(t)
  return 'done sleeping...'

def func4(t):
  time.sleep(t)
  return(f'done sleeping {t} second(s)')

def func5(t):
  time.sleep(t)
  return(f'slept {t} second(s)')

if __name__ == '__main__':
  with concurrent.futures.ProcessPoolExecutor () as executor:
    f1 = executor.submit(func1)
    f2 = executor.submit(func2,1)

    result3 = [executor.submit(func3, 0.25) for _ in range(5)]
    for f in concurrent.futures.as_completed(result3):
      print(f.result())

    secs = [0.3,0.2,0.1]
    result4 = [executor.submit(func4, sec) for sec in secs]
    for f in concurrent.futures.as_completed(result4):
      print(f.result())

    result5 = executor.map(func2, secs)
