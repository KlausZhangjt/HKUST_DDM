class Node():
    def __init__(self, data):
        self.next = None
        self.data = data

class LinkedList():

    def __init__(self, data_list = []):
        self.head = None
        for d in data_list:
            cur = Node(d)
            if self.head:
                prev.next = cur
            else:
                self.head = cur
            prev = cur

    def __init___old_version(self, data_list = []):
        if len(data_list) == 0:
            self.head = None
            return
        self.head = Node(data_list[0])
        prev = self.head
        for d in data_list[1:]:
            prev.next = Node(d)
            prev = prev.next

    def __str__(self, node = 0):
        if node == 0:
            node = self.head
        if node:
            return "{} -> {}".format(node.data, self.__str__(node.next))
        else:
            return "None"

    def revert(self):
        old_prev = None
        node = self.head
        while node:
            old_next = node.next
            node.next = old_prev
            old_prev = node
            node = old_next
        self.head = old_prev

    def double(self):
        if self.head is None: return
        new_head = Node(self.head.data)
        new_prev = new_head
        old_prev = self.head
        while old_prev.next:
            new_prev.next = Node(old_prev.next.data)
            old_prev = old_prev.next
            new_prev = new_prev.next
        old_prev.next = new_head



if __name__ == "__main__":

    ln = LinkedList()
    print(ln)  
    ln.revert()
    print(ln)
    ln.double()
    print(ln)
    ln = LinkedList([0,1,2])
    print(ln)  
    ln.revert()
    print(ln)
    ln.double()
    print(ln)