def quickselect(array, k):
    pivot = array[0]
    left = [i for i in array[1:] if i <= pivot]
    right = [i for i in array[1:] if i > pivot]
    if k > len(left):
        return quickselect(right, k - len(left) - 1)
    elif k < len(left):
        return quickselect(left, k)
    else:
        return pivot

if __name__ == "__main__":
    print(quickselect([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0],3))
    print(quickselect([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0],4))
    print(quickselect(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"], 3))