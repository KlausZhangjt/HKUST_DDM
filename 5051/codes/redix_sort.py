# Here we assume base 10 when bursting numbers into integers.

def counting_sort(array):
    max_elem = max(array)
    counts = [0 for i in range(max_elem + 1)]
    for elem in array:
        counts[elem] += 1
    return [i for i in range(len(counts)) for cnt in range(counts[i])]


def counting_sort_by(array, max_rank = None, rank = lambda x: x):
    "Counting sort wrt its non-negative integer-valued rank(elem)."
    if max_rank is None:
        max_rank = 0
        for elem in array:
            if rank(elem) > max_rank:
                max_rank = rank(elem)
    # One cannot do counts = [[]] * (max_rank + 1), otherwise all [] have the same reference
    counts = [[] for cnt in range(max_rank+1)]
    for elem in array:
        (counts[rank(elem)]).append(elem)
    return [elem for sublist in counts for elem in sublist]

def integer_digits(num):
    "Example: integer_digits(3142) == [3,1,4,2]"
    digits = []
    while num > 0:
        digits.append(num % 10)
        num = num // 10
    return digits[::-1]

def from_digits(digits):
    num = 0
    for d in digits:
        num = num * 10
        num += d
    return num

def dig(rd, d):
    "Example: dig([3,1,4,2],0) == 2; dig([3,1,4,2],1) == 4;  dig([3,1,4,2],4) == 0"
    return 0 if d >= len(rd) else rd[-(d+1)]

def radix_LSD_sort(array):
    rd_array = []
    max_n_digits = 0
    for num in array:
        bursted = integer_digits(num)
        rd_array.append(bursted)
        if max_n_digits < len(bursted):
            max_n_digits = len(bursted)
    for d in range(max_n_digits):
        rd_array = counting_sort_by(rd_array, 9, lambda rd: dig(rd, d))
    return [from_digits(digits) for digits in rd_array]

if __name__ == "__main__":
    test = counting_sort([2,4,3,0,2,4,6,3,2,3,2,1,2,1,0])
    print(test)
    test = counting_sort_by([2,4,3,0,2,4,6,3,2,3,2,1,2,1,0])
    print(test)
    test = counting_sort_by([2,4,3,0,2,4,6,3,2,3,2,1,2,1,0], max_rank=10)
    print(test)
    print(integer_digits(31415926535))
    test = radix_LSD_sort([12, 1048576, 31, 1073741824, 9, 16384, 18, 246, 65535, 0, 23, 1, 2048])
    print(test)