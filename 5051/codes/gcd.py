def GCD(a, b):
    return a if b == 0 else GCD(b, a % b)

print(GCD(1071, 462))

