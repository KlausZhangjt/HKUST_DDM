zero = lambda f: lambda x: x
succ = lambda n: lambda f: lambda x: f(n(f)(x))
one = succ(zero)
two = succ(one)
...



print(one)
print(two)