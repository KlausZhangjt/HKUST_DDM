class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.parent = None
        self.data = data
        self.height = 0

class BinaryTree:

    def __init__(self):
        self.root = None

    def __str__(self, node = 0, depth = 0, direction_label = ""):
        "The tree structure in string form, to be used in str(my_node) or print(my_node)."
        if node == 0:
            node = self.root
        if node:
            height_info = "(H"+str(node.height)+")" if node.height > 0 else ""
            return depth * "\t" + direction_label + height_info + str(node.data) + "\n" + \
                self.__str__(node.left, depth+1, "L:") + self.__str__(node.right, depth+1, "R:")
        else:
            return ""

    def inorder(self, node = 0, result = None):
        if result is None:
            result = []
        if node == 0:
            node = self.root
        if node:
            self.inorder(node.left, result)
            result.append(node.data)
            self.inorder(node.right, result)
        return result


if __name__ == "__main__":
    tree = BinaryTree()
    tree.root = Node("0")
    tree.root.left = Node("1")
    tree.root.right = Node("2")
    tree.root.left.right = Node("3")
    print(tree)
    print(tree.inorder())

    tree = BinaryTree()
    tree.root = Node("0")
    print(tree)
    print(tree.inorder())



