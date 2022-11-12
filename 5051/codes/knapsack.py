class item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value


from functools import lru_cache
def knapsack_1(S, item_array):
    items = [item(i[0], i[1]) for i in item_array]

    @lru_cache(maxsize=None)
    def DP(S, k):
        if k == -1: return 0
        if S - items[k].weight < 0: return DP(S, k-1)
        return max(DP(S, k-1), DP(S-items[k].weight, k-1) + items[k].value)
    print_solution(S, items, DP)


def knapsack_2(S, item_array):
    memo = {}
    items = [item(i[0], i[1]) for i in item_array]
    def DP(S, k):
        if k == -1: return 0
        if S - items[k].weight < 0: 
            memo[(S, k)] = DP(S, k-1)
            return memo[(S, k)]
        if (S, k) not in memo:
            memo[(S, k)] = max(DP(S, k-1), DP(S-items[k].weight, k-1) + items[k].value)
        return memo[(S, k)]
    print_solution(S, items, DP)


def knapsack_3(S, item_array):
    memo = {}
    items = [item(i[0], i[1]) for i in item_array]
    k = len(items)
    for ls in range(S+1):
        for lk in range(-1, k):
            if lk == -1:
                memo[(ls, lk)] = 0
                continue
            if ls - items[lk].weight < 0:
                memo[(ls, lk)] = memo[(ls, lk-1)]
                continue
            memo[(ls, lk)] = max(memo[(ls, lk-1)], memo[(ls-items[lk].weight, lk-1)] + items[lk].value)

    print()
    for ls in range(S+1):
        print([memo[(ls, i)] for i in range(-1, k)])


    print()

    print_solution(S, items, lambda ls, lk: memo[(ls, lk)])

def print_solution(S, items, DP):
    print("Total value = ", DP(S, len(items)-1))
    remaining = S
    picked = []
    for k in reversed(range(len(items))):
        if DP(remaining, k) != DP(remaining, k-1):
            picked.append(k)
            remaining -= items[k].weight
    print(picked)

knapsack = knapsack_3
knapsack(8, [[1, 15], [5, 10], [3, 9], [4, 5]])

# knapsack(3, [[2,2],[1,1],[2,2]])
# knapsack(165, [[23,92],[31,57],[29,49],[44,68],[53,60],[38,43],[63,67],[85,84],[89,87],[82,72]])

# item100 = [[4, 4], [5, 1], [5, 5], [3, 2], [1, 2], [5, 3], [2, 3], [5, 2], [5, 4], [5, 4], [3, 4], [5, 1], [1, 4], [1, 2], [1, 2], [1, 1], [3, 4], [4, 3], [4, 1], [2, 5], [1, 1], [4, 2], [3, 2], [3, 1], [2, 1], [5, 2], [1, 2], [1, 4], [5, 2], [4, 4], [3, 3], [5, 4], [2, 5], [1, 1], [4, 5], [4, 2], [3, 1], [3, 2], [5, 4], [5, 2], [2, 2], [5, 2], [5, 2], [5, 3], [2, 4], [3, 2], [4, 2], [2, 1], [4, 2], [5, 4], [2, 5], [5, 1], [4, 4], [5, 1], [2, 5], [5, 1], [3, 2], [4, 5], [3, 3], [5, 5], [2, 2], [1, 3], [1, 1], [1, 5], [1, 5], [1, 4], [5, 5], [3, 3], [2, 3], [2, 1], [5, 2], [1, 1], [4, 2], [1, 4], [1, 2], [2, 1], [1, 3], [3, 3], [2, 1], [5, 5], [2, 1], [2, 3], [4, 5], [3, 5], [3, 3], [2, 2], [5, 1], [3, 1], [1, 5], [4, 4], [1, 2], [4, 3], [1, 2], [3, 4], [1, 3], [5, 1], [4, 2], [1, 3], [3, 5], [4, 4]]

# knapsack(250, item100)

