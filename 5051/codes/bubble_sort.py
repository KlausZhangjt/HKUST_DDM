
def bubble_sort_1(array):
    n = len(array)
    for i in range(n):
        for j in range(n-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array


def bubble_sort_2(array):
    n = len(array)
    while True:
        swapped = False
        for i in range(n-1):
            if array[i] > array[i+1]:
                array[i], array[i+1] = array[i+1], array[i]
                swapped = True
        if swapped == False:
            break
    return array

if __name__ == "__main__":
    print(bubble_sort_1([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0]))
    print(bubble_sort_1(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"]))