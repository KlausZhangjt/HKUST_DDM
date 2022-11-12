class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

# It's more organized to use a class for linked list. Here just for teaching purpose we use individual functions.

def construct2(data_list):
    head = None
    prev = None
    for data in data_list:
        new_node = Node(data)
        if not head:
            head = new_node
        if prev:
            prev.next = new_node
        prev = new_node
    return head

def construct(data_list):
    next = None
    for data in reversed(data_list):
        next = Node(data, next)
    return next

def print_linked_list(head):
    node = head
    while node:
        print(node.data)
        node = node.next

def revert(head):
    prev = head
    cur = head.next
    head.next = None
    while cur:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    return prev

ll = construct2(["a","b","c"])
ll = construct2(["a"])
print_linked_list(ll)
print_linked_list(revert(ll))