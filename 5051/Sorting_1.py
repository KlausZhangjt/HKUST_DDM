# Sorting Algo

# array = [5, 4, 3, 2, 1]
array = [9, 2, 25, 4, 8, 7, 2, 1, 15, 4, 29]


# %% Insertion Sort
def insertion_sort(array):
    n = len(array)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if array[j] < array[j - 1]:
                array[j - 1], array[j] = array[j], array[j - 1]
            else:
                break
    return array


# %% Bubble sort
def bubble_sort(array):
    n = len(array)
    for i in range(n - 1):
        for j in range(n - 1, i, -1):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
    return array


# %% Quick sort
def quicksort(array):
    if len(array) < 2:
        return array

    pivot = array[0]
    left = [i for i in array[1:] if i <= pivot]
    right = [i for i in array[1:] if i > pivot]
    return quicksort(left) + [pivot] + quicksort(right)


# %% Merge sort
def merge_sort(array):
    if len(array) < 2:
        return array

    mid = (len(array) + 1) // 2
    sub1 = merge_sort(array[:mid])
    sub2 = merge_sort(array[mid:])
    return ordered_merge(sub1, sub2)


def ordered_merge(a1, a2):
    cnt1, cnt2 = 0, 0
    result = []
    while cnt1 < len(a1) and cnt2 < len(a2):
        if a1[cnt1] <= a2[cnt2]:
            result.append(a1[cnt1])
            cnt1 += 1
        else:
            result.append(a2[cnt2])
            cnt2 += 1
    result += a1[cnt1:]
    result += a2[cnt2:]
    return result


# %% Heap sort
class Heap:
    def __init__(self, array):
        self.heap = array
        self.length = len(array)
        self.build_max_heap()

    def left(self, idx):
        pos = 2 * idx + 1
        return pos if pos < self.length else None

    def right(self, idx):
        pos = 2 * idx + 2
        return pos if pos < self.length else None

    def parent(self, idx):
        return (idx - 1) // 2 if idx > 0 else None

    def build_max_heap(self):
        last_to_heapify = self.parent(self.length - 1)
        # lower limit of loop is 0
        for i in range(last_to_heapify, -1, -1):
            self.max_heapify(i)

    def _greater_child(self, i):
        left, right = self.left(i), self.right(i)
        if left is None and right is None:
            return None
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            return left if self.heap[left] > self.heap[right] else right

    def max_heapify(self, i):
        greater_child = self._greater_child(i)
        if greater_child is not None and self.heap[greater_child] > self.heap[i]:
            self.heap[i], self.heap[greater_child] = self.heap[greater_child], self.heap[i]
            self.max_heapify(greater_child)

    def sort(self):
        while self.length > 1:
            self.heap[0], self.heap[self.length - 1] = self.heap[self.length - 1], self.heap[0]
            self.length -= 1
            self.max_heapify(0)
        return self.heap


def heap_sort(array):
    my_heap = Heap(array)
    return my_heap.sort()


# %% Radix sort
# Here we assume base 10 when bursting numbers into integers.
def counting_sort(array):
    max_elem = max(array)
    counts = [0 for i in range(max_elem + 1)]
    for elem in array:
        counts[elem] += 1
    return [i for i in range(len(counts)) for cnt in range(counts[i])]


def counting_sort_by(array, max_rank=None, rank=lambda x: x):
    """Counting sort wrt its non-negative integer-valued rank(elem)."""
    if max_rank is None:
        max_rank = 0
        for elem in array:
            if rank(elem) > max_rank:
                max_rank = rank(elem)
    # One cannot do counts = [[]] * (max_rank + 1), otherwise all [] have the same reference
    counts = [[] for cnt in range(max_rank + 1)]
    for elem in array:
        (counts[rank(elem)]).append(elem)
    return [elem for sublist in counts for elem in sublist]


def integer_digits(num):
    """Example: integer_digits(3142) == [3,1,4,2]"""
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
    """Example: dig([3,1,4,2],0) == 2; dig([3,1,4,2],1) == 4;  dig([3,1,4,2],4) == 0"""
    return 0 if d >= len(rd) else rd[-(d + 1)]


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
    test = counting_sort([2, 4, 3, 0, 2, 4, 6, 3, 2, 3, 2, 1, 2, 1, 0])
    print(test)
    test = counting_sort_by([2, 4, 3, 0, 2, 4, 6, 3, 2, 3, 2, 1, 2, 1, 0])
    print(test)
    test = counting_sort_by([2, 4, 3, 0, 2, 4, 6, 3, 2, 3, 2, 1, 2, 1, 0], max_rank=10)
    print(test)
    print(integer_digits(31415926535))
    test = radix_LSD_sort([12, 1048576, 31, 1073741824, 9, 16384, 18, 246, 65535, 0, 23, 1, 2048])
    print(test)


# %% Practice 1: Natural sort
def natural_sort(array):
    run_list = cut_into_runs(array)
    result = merge_runs(run_list)
    return result


def cut_into_runs(array):
    runs = []
    indx = 0
    for i in range(1, len(array)):
        if array[i] < array[i - 1]:
            runs.append(array[indx:i])
            indx = i
    runs.append(array[indx:])
    return runs


def merge_runs(run_list):
    if len(run_list) == 1:
        return run_list[0]

    runs = []
    for i in range(1, len(run_list), 2):
        runs.append(ordered_merge(run_list[i - 1], run_list[i]))

    if len(run_list) % 2:
        runs.append(run_list[-1])

    return merge_runs(runs)


def ordered_merge(a1, a2):
    cnt1, cnt2 = 0, 0
    result = []

    while cnt1 < len(a1) and cnt2 < len(a2):
        if a1[cnt1] <= a2[cnt2]:
            result.append(a1[cnt1])
            cnt1 += 1
        else:
            result.append(a2[cnt2])
            cnt2 += 1
    result += a1[cnt1:]
    result += a2[cnt2:]

    return result


array1 = [6, 7, 8, 3, 4, 1, 11, 12, 13, 2]
print(cut_into_runs(array1))
print(natural_sort(array1))


# %% Practice 2: Insertion sort in linked list
class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node


class SinglyLinkedList:
    def __init__(self, head=None):
        self.head = head

    def print_list(self):
        print_node = self.head
        while print_node is not None:
            print(print_node.data)
            print_node = print_node.next

    def insertion_sort(self):
        prev = self.head
        while prev.next:
            curr = prev.next
            if curr.data < self.head.data:
                prev.next = curr.next
                curr.next = self.head
                self.head = curr  # declare current node to be the new head of the list
            else:
                checker = self.head  # declare a checker that runs from the beginning to prev
                while checker is not prev:
                    if curr.data < checker.next.data:
                        prev.next = curr.next
                        curr.next = checker.next
                        checker.next = curr
                        break  # escape the checker loop
                    checker = checker.next
                if checker is prev:
                    prev = prev.next


my_array = [9, 2, 25, 4, 8, 7, 2, 1, 15, 4, 29]

# convert into linked list
node_list = [Node(i) for i in my_array]
my_SLL = SinglyLinkedList(node_list[0])
for i, node in enumerate(node_list[:-1]):
    node.next = node_list[i + 1]

my_SLL.insertion_sort()
my_SLL.print_list()

