def run(fn, S, T):
    return fn(S, T, len(S)-1, len(T)-1)

# LCS: Longest Common Subsequence

def LCS_1(S, T, n, m):
    if m<0 or n<0: return ""
    if S[n] == T[m]:
        return LCS_1(S, T, n-1, m-1) + S[n]
    elif len(LCS_1(S, T, n-1, m)) > len(LCS_1(S, T, n, m-1)):
        return LCS_1(S, T, n-1, m)
    else: 
        return LCS_1(S, T, n, m-1)

from functools import lru_cache
@lru_cache(maxsize=None)
def LCS_2(S, T, n, m):
    if m<0 or n<0: return ""
    if S[n] == T[m]:
        return LCS_2(S, T, n-1, m-1) + S[n]
    elif len(LCS_2(S, T, n-1, m)) > len(LCS_2(S, T, n, m-1)):
        return LCS_2(S, T, n-1, m)
    else: 
        return LCS_2(S, T, n, m-1)    

memo = {}
def LCS_3(S, T, n, m):
    if m<0 or n<0: return ""
    if (n, m) in memo: return memo[(n, m)]
    if S[n] == T[m]:
        result =  LCS_3(S, T, n-1, m-1) + S[n]
    elif len(LCS_3(S, T, n-1, m)) > len(LCS_3(S, T, n, m-1)):
        result =  LCS_3(S, T, n-1, m)
    else: 
        result =  LCS_3(S, T, n, m-1)
    memo[(n, m)] = result
    return result

def LCS_4(S, T, n, m):
    memo = {}
    for i in range(-1, len(S)):
        for j in range(-1, len(T)):
            if i == -1 or j == -1: 
                memo[(i, j)] = ""
                continue
            if S[i] == T[j]:
                memo[(i, j)] = memo[(i-1, j-1)] + S[i]
            elif len(memo[(i-1, j)]) > len(memo[(i, j-1)]):
                memo[(i, j)] = memo[(i-1, j)]
            else:
                memo[(i, j)] = memo[(i, j-1)]
    return memo[len(S)-1, len(T)-1]


# SCS: Shortest Common Supersequence

@lru_cache(maxsize=None)
def SCS_1(X, Y, n, m):
    if m == -1: return X[:n+1]
    if n == -1: return Y[:m+1]
    if X[n] == Y[m]: return SCS_1(X, Y, n-1, m-1) + X[n]
    if len(SCS_1(X, Y, n-1, m)) < len(SCS_1(X, Y, n, m-1)):
        return SCS_1(X, Y, n-1, m) + X[n]
    else:
        return SCS_1(X, Y, n, m-1) + Y[m]


# Tests

S = "ABAZDC"
T = "BACBAD"

S = "SDLTQLWSLSDLTQLWSLSDLTQLWSLSDLTQLWSL"
T = "TLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDR"

S = "SDL TQL WSL"
T = "SQL server on Windows Subsystem for Linux"

# print(run(LCS_1, S, T))
print(run(LCS_2, S, T))
print(run(LCS_3, S, T))
print(run(LCS_4, S, T))

X = "SDLTQL"
Y = "DL666"

# X = "ABCBDAB"
# Y = "BDCABA"

# X = "SDLTQLWSLSDLTQLWSLSDLTQLWSLSDLTQLWSL"
# Y = "TLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDRTLDR"

print(run(SCS_1, X, Y))