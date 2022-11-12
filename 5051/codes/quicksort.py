def quicksort(array):
    if len(array)<2: return array
    pivot = array[0]
    left = [i for i in array[1:] if i <= pivot]
    right = [i for i in array[1:] if i > pivot]
    return quicksort(left) + [pivot] + quicksort(right)

if __name__ == "__main__":
    print(quicksort([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0]))
    print(quicksort(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"]))

