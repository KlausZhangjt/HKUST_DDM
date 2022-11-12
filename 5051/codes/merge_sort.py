def merge_sort(array):
    if len(array) <= 1:
        return array
    mid = (len(array) + 1) // 2
    sub1 = merge_sort(array[: mid])
    sub2 = merge_sort(array[mid :])
    return ordered_merge(sub1, sub2)

def ordered_merge(a1, a2):
    cnt1, cnt2 = 0, 0
    result = []
    while cnt1 < len(a1) and cnt2 < len(a2):
        if a1[cnt1] < a2[cnt2]:
            result.append(a1[cnt1])
            cnt1 += 1
        else:
            result.append(a2[cnt2])
            cnt2 += 1
    result += a1[cnt1 :]
    result += a2[cnt2 :]
    return result

if __name__ == "__main__":
    print(merge_sort([2,5,7,3,2,5,7,9,6,7,9,2,3,5,9,3,0]))
    print(merge_sort(["Zhao", "Qian", "Sun","Li", "Zhou", "Wu", "Zheng", "Wang"]))

