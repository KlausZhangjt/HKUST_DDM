from timeit import timeit

# naive approach, not caching
def fib_1(n):
    return fib_1(n-1) + fib_1(n-2) if n > 2 else 1

memo = {}
def fib_2(n):
    if n not in memo:
        memo[n] = fib_2(n-1) + fib_2(n-2) if n > 2 else 1
    return memo[n]

memo = {1:1, 2:1}
def fib_2p(n):
    if n not in memo:
        memo[n] = fib_2p(n-1) + fib_2p(n-2)
    return memo[n]

def fib_2_entry(n):
    memo = {} # clear global cache
    return fib_2(n)

def fib_2p_entry(n):
    memo = {} # clear global cache
    return fib_2p(n)

from functools import lru_cache
@lru_cache(maxsize=None)
def fib_3(n):
    return fib_3(n-1) + fib_3(n-2) if n > 2 else 1

def fib_3_entry(n):
    fib_3.cache_clear()
    return fib_3(n)

def fib_4(n):
    fib = {1:1, 2:1}
    for i in range(3, n+1):
        fib[i] = fib[i-1] + fib[i-2]
    return fib[n]

def fib_5(n):
    fib = [1 for i in range(n+1)]
    for i in range(3, n+1):
        fib[i] = fib[i-1] + fib[i-2]
    return fib[n]

print("Timings of fib_1:")
print(timeit('fib_1(32)', globals=globals(), number=1))

print("Timings of fib_2_entry (x1000):")
print(timeit('fib_2_entry(32)', globals=globals(), number=10000))

print("Timings of fib_2p_entry (x1000):")
print(timeit('fib_2p_entry(32)', globals=globals(), number=10000))

print("Timings of fib_3_entry (x1000):")
print(timeit('fib_3_entry(32)', globals=globals(), number=10000))
print(fib_3.cache_info())

print("Timings of fib_4 (x1000):")
print(timeit('fib_4(32)', globals=globals(), number=10000))

print("Timings of fib_5 (x1000):")
print(timeit('fib_5(32)', globals=globals(), number=10000))
