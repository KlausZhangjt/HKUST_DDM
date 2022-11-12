
def permute_string(str):
    permute_array(list(str))

def permute_array(a, swapping = 0):
    if swapping == len(a) - 1:
        print("".join(a))
        return
    for i in range(swapping, len(a)):
        a[swapping], a[i] = a[i], a[swapping]
        permute_array(a, swapping+1)
        a[i], a[swapping] = a[swapping], a[i]

permute_string("abc")
