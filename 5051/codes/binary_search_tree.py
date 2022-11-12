import binary_tree

class Node(binary_tree.Node):
    def left_height(self):
        return -1 if self.left is None else self.left.height
    def right_height(self):
        return -1 if self.right is None else self.right.height
    def update_height(self):
        self.height = max(self.left_height(), self.right_height()) + 1
    def balance(self):
        "-2, -1: left heavy, 1, 2: right heavy"
        return self.right_height() - self.left_height()

class BinarySearchTree(binary_tree.BinaryTree):
    def __init__(self, data_array = []):
        self.root = None
        for data in data_array:
            self.insert(Node(data))
    
    def insert(self, new_node, node = 0):
        if not self.root:
            self.root = new_node
            return new_node
        if node == 0:
            node = self.root
        if new_node.data < node.data:
            if node.left:
                self.insert(new_node, node.left)
            else:
                new_node.parent = node
                node.left = new_node
        else:
            if node.right:
                self.insert(new_node, node.right)
            else:
                new_node.parent = node
                node.right = new_node
        node.update_height()

    def sort(self):
        return self.inorder()

def BST_sort(array):
    my_tree = BinarySearchTree(array)
    return my_tree.sort()

if __name__ == "__main__":
    tree = BinarySearchTree([0,3,7,5,2,1,9,10])
    tree.insert(Node(-1))
    print(tree)
    print(tree.sort())
    tree = BinarySearchTree(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"])
    print(tree)
    print(tree.sort())
