class Heap():
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
            return left if self.heap[left]>self.heap[right] else right

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

if __name__ == "__main__":
    my_heap = Heap([2,5,9,4,2,4,10,8,2,1])
    print(my_heap.heap)
    print(my_heap.sort())
