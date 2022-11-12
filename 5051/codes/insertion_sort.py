def insertion_sort(array):
    n = len(array)
    for i in range(1, n): # The new card: (1, 2, ..., n-1)
        for j in range(i, 0, -1): # From the new card to old cards
            if array[j] < array[j-1]:
                array[j], array[j-1] = array[j-1], array[j]
            else:
                break
    return array

if __name__ == "__main__":
    print(insertion_sort([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0]))
    print(insertion_sort(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"]))

