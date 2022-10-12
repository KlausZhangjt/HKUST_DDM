#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZHANG Juntao

5002 Assignment-2
"""
# %% packages required
import numpy as np
import time
import os
import shutil
import math
import matplotlib.pyplot as plt
np.random.seed(20908272)

# %%question 1: Russian roulette

def A_win_1_bullet(num_game):
    num_pos = 10
    B_round = [1, 2, 3, 4, 5]
    B_lose = 0
    for i in range(num_game):
        gun = np.zeros(num_pos, bool)
        gun[np.random.randint(num_pos)] = True
        if any(gun[B_round]):
            B_lose += 1
    return B_lose/num_game


def A_win_1_bullet_fast(num_game):
    num_pos = 10
    bullets = np.random.randint(num_pos, size=num_game)
    B_lose = np.sum(bullets >= 1) - np.sum(bullets > 5)
    return B_lose/num_game


def A_win_3_bullet(num_game):
    num_pos = 10
    B_round = [1, 2, 3, 4, 5]
    B_lose = 0
    
    for i in range(num_game):
        gun = np.zeros(num_pos)
        b1 = np.random.randint(num_pos)
        gun[b1] = 1
        
        b2 = b1
        while b2 == b1:
            b2 = np.random.randint(num_pos)
        gun[b2] = 1
        
        b3 = b1
        while b3 == b1 or b3 == b2:
            b3 = np.random.randint(num_pos)
        gun[b3] = 1
        
        for n in range(num_pos):
            if gun[n] == 1:
                if n in B_round:
                    B_lose += 1
                break
    return B_lose/num_game


def A_win_3_bullet_fast(num_game):
    t = num_game
    B_lose = 0
    d = np.random.binomial(t, 3/10)
    t -= d
    for i in range(1, 6):
        d = np.random.binomial(t, 3/(10-i))
        B_lose += d
        t = t-d
    return B_lose / num_game


# 1_bullet, 1_bullet_fast, 3_bullet, 3_bullet_fast
T1, T2, T3, T4 = 0, 0, 0, 0  # cumulative running time
P1, P2, P3, P4 = [], [], [], []  # records of winning probability

np.random.seed(777)
num_test = 10
print('test', end=' ')
for n in range(num_test):
    print(n, end=' ')
    num_game = 10 ** 5

    start_time = time.time()
    P1.append(A_win_1_bullet(num_game))
    T1 += (time.time() - start_time)

    start_time = time.time()
    P2.append(A_win_1_bullet_fast(num_game))
    T2 += (time.time() - start_time)

    start_time = time.time()
    P3.append(A_win_3_bullet(num_game))
    T3 += (time.time() - start_time)

    start_time = time.time()
    P4.append(A_win_3_bullet_fast(num_game))
    T4 += (time.time() - start_time)

print()
print('T2/T1 =', T2 / T1)
print('T4/T3 =', T4 / T3)
print(min(P1), '<= P1 <=', max(P1))
print(min(P2), '<= P2 <=', max(P2))
print(min(P3), '<= P3 <=', max(P3))
print(min(P4), '<= P4 <=', max(P4))


# %% question 2: Supermarket

class Simulation:
    def __init__(self, n, k, src):
        self.n = n
        self.k = k
        self.src = src
        self.tmp_r1 = 0
        self.tmp_r2 = 0
        self.tmp_c1 = 0
        self.tmp_c2 = 0

    def finding_time(self, addition=False):
        r1, c1, r2, c2 = self.get_initial_pos()
        t = 0
        self.tmp_r1, self.tmp_c1, self.tmp_r2, self.tmp_c2 = self.n+2, self.n+2, self.n+2, self.n+2
        while r1 != r2 or c1 != c2:
            if addition is False:
                if r1 == r2:
                    return t + abs(c1 - c2) / 2 * 5
                elif c1 == c2:
                    return t + abs(r1 - r2) / 2 * 5
            elif addition is True:
                if r1 == r2 and abs(c1-c2) == 1:
                    return t+5
                elif c1 == c2 and abs(r1-r2) == 1:
                    return t+5
            t += 5
            r1, c1 = self.update_step_A(r1, c1)
            if self.src == 2:
                r2, c2 = self.update_step_B(r2, c2)
            elif self.src == 1:
                continue
        return t

    def get_initial_pos(self):
        if self.k == -1:
            return 0, 0, self.n, self.n
        else:
            al, bo = self.n//2-self.k,  self.n-self.n//2+self.k
            return al, al, bo, bo

    def update_step_A(self, rx, cx):
        lisA = []
        for a, b in ((rx + 1, cx), (rx - 1, cx), (rx, cx + 1), (rx, cx - 1)):
            if 0 <= a <= self.n and 0 <= b <= self.n and (a != self.tmp_r1 or b != self.tmp_c1):
                lisA.append([a, b])
        d1 = np.random.randint(len(lisA))
        self.tmp_r1, self.tmp_c1 = rx, cx
        return lisA[d1][0], lisA[d1][1]

    def update_step_B(self, rx, cx):
        lisB = []
        for a, b in ((rx + 1, cx), (rx - 1, cx), (rx, cx + 1), (rx, cx - 1)):
            if 0 <= a <= self.n and 0 <= b <= self.n and (a != self.tmp_r2 or b != self.tmp_c2):
                lisB.append([a, b])
        d2 = np.random.randint(len(lisB))
        self.tmp_r2, self.tmp_c2 = rx, cx
        return lisB[d2][0], lisB[d2][1]


def get_AvgTime(left, right, trail=1000, add=False, x_axis_k=False):
    length = right-left+1
    src_1, src_2 = [[0]*trail for _ in range(length)], [[0]*trail for _ in range(length)]
    time_src_1, time_src_2 = [], []
    xs = [k for k in range(left, right+1)]
    for nu in range(length):
        for ti in range(trail):
            if x_axis_k is False:
                S1 = Simulation(n=nu+2, k=-1, src=1)
                S2 = Simulation(n=nu+2, k=-1, src=2)
            elif x_axis_k is True:
                S1 = Simulation(n=19, k=nu, src=1)
                S2 = Simulation(n=19, k=nu, src=2)
            src_1[nu][ti] = S1.finding_time(add)
            src_2[nu][ti] = S2.finding_time(add)
        time_src_1.append(np.mean(src_1[nu]))
        time_src_2.append(np.mean(src_2[nu]))
    return xs, time_src_1, time_src_2


# %% question 2a
x, s1, s2 = get_AvgTime(2, 20)
plt.figure(figsize=(10, 7))
plt.plot(x, s1, marker='.', label='src1: only Alice walks')
plt.plot(x, s2, marker='.', label='src2: Alice and Bob both walk')
x_ticks = np.linspace(2, 20, 19)
plt.xticks(x_ticks)
plt.title('Question 2a: Average finding time over 1000 trials')
plt.xlabel('n')
plt.ylabel('time')
plt.legend()
plt.grid(ls='--')
plt.show()

# %% question 2b
x, s1, s2 = get_AvgTime(0, 9, x_axis_k=True)
plt.figure(figsize=(10, 7))
plt.plot(x, s1, marker='.', label='src1: only Alice walks')
plt.plot(x, s2, marker='.', label='src2: Alice and Bob both walk')
x_ticks = np.linspace(0, 9, 10)
plt.xticks(x_ticks)
plt.title('Question 2b: Average finding time over 1000 trials')
plt.xlabel('k')
plt.ylabel('time')
plt.legend()
plt.grid(ls='--')
plt.show()

# %% question 2c_a
x, s1, s2 = get_AvgTime(2, 20, add=True)
plt.figure(figsize=(10, 7))
plt.plot(x, s1, marker='.', label='src1: only Alice walks')
plt.plot(x, s2, marker='.', label='src2: Alice and Bob both walk')
x_ticks = np.linspace(2, 20, 19)
plt.xticks(x_ticks)
plt.title('Question 2c_a: Average finding time over 1000 trials')
plt.xlabel('n')
plt.ylabel('time')
plt.legend()
plt.grid(ls='--')
plt.show()

# %% question 2c_b
x, s1, s2 = get_AvgTime(0, 9, add=True, x_axis_k=True)
plt.figure(figsize=(10, 7))
plt.plot(x, s1, marker='.', label='src1: only Alice walks')
plt.plot(x, s2, marker='.', label='src2: Alice and Bob both walk')
x_ticks = np.linspace(0, 9, 10)
plt.xticks(x_ticks)
plt.title('Question 2c_b: Average finding time over 1000 trials')
plt.xlabel('k')
plt.ylabel('time')
plt.legend()
plt.grid(ls='--')
plt.show()


# %%question3: File organization

def organize(path):
    dic = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
           'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
    file_names = os.listdir(path)
    years = []
    for i in range(len(file_names)):
        if file_names[i][7:11] not in years:
            years.append(file_names[i][7:11])
    for j in range(len(years)):
        os.mkdir(os.path.join(path, str(years[j])))
    for file in file_names:
        old = os.path.join(path, file)
        new_path = os.path.join(path, file[7:11])
        new = os.path.join(new_path, file)
        shutil.move(old, new)
        replaced = file.replace(file[3:6], dic[file[3:6]])
        new_name = replaced[3:5]+replaced[0:2]+replaced[10:]
        os.rename(new, os.path.join(new_path, new_name))


organize('/Users/klaus_zhangjt/Desktop/assignment2_materials/question3')


# %%question4: Frequency analysis

def get_key(dic, value):
    res = []
    for alpha in dic:
        if dic[alpha] == value:
            res.append(alpha)
    res.sort()
    return res


def f_analysis(sentence,  f_key):
    print(sentence)
    print('=== Frequency table ===')
    sentence = sentence.lower()
    dic = {}
    for item in sentence:
        if item.isalpha():
            dic[item] = sentence.count(item)
    count = []
    for n in dic.values():
        if n not in count:
            count.append(n)
    count.sort(reverse=True)
    dic_true = {}
    for i in count:
        items = get_key(dic, i)
        dic_true[i] = items
        s = '， '.join(items)
        print(i, '    ', s, sep='')
    if f_key is True:
        return dic_true
    else:
        return dic


S = 'Alice and Bob both take MSDM5002 at HKUST.'
D = f_analysis(S, f_key=True)

# D = f_analysis(S, f_key=False)


# %% question 5: Bracket checking

def bracket_check(s):
    round_b, quare_b, curly_b = 0, 0, 0
    count = 0
    for i in s:
        if i == '(':
            round_b += 1
            count += 1
        if i == '[':
            quare_b += 1
            count += 1
        if i == '{':
            curly_b += 1
            count += 1
        round_b -= 1 if i == ')' else 0
        quare_b -= 1 if i == ']' else 0
        curly_b -= 1 if i == '}' else 0
        if round_b < 0 or quare_b < 0 or curly_b < 0:
            return -1
    if round_b == 0 and quare_b == 0 and curly_b == 0:
        return count
    else:
        return -1


bracket_check('{m(s[d]m(5}0[c)02]')

# bracket_check('p(y[th[on]{c(our)s}e])')


# %% question 6: Band matrix

def bandwidth(A):
    def find_width(A):
        m, n = len(A), len(A[0])
        k, width = min(m, n), 0
        while width < n:
            fail = 0
            for r in range(k):
                if r + width + 1 < n:
                    fail += 0 if A[r][r + width + 1:].any() == 0 else 1
                else:
                    break
            if fail == 0:
                return width
            width += 1
        return width

    # print('(', find_width(A), ', ', find_width(A.T), ')', sep='')
    return find_width(A),  find_width(A.T)


X = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
Y = np.array([[1, 0], [1, 1], [1, 1]])

bandwidth(X)

# bandwidth(Y)

# %% question 7:  Coupon collector

class Box:
    def __init__(self, N):
        self.coupon = np.random.randint(N)

    def open(self):
        return self.coupon


def box_number(N, k):
    get = set()
    count = 0
    while len(get) < k:
        get.add(Box(N).open())
        count += 1
    return count


start = time.perf_counter()
duration = 0
res1, res2 = [[] for _ in range(10)], [[] for _ in range(10)]
while duration <= 0.1:
    for j in range(1, 11):
        res1[j-1].append(box_number(10, j))
        res2[j-1].append(box_number(j, j))
    duration = time.perf_counter()-start

num1 = [np.mean(res1[i]) for i in range(10)]
num2 = [np.mean(res2[i]) for i in range(10)]
k = [_ for _ in range(1, 11)]

# %% question 7a
plt.figure(figsize=(10, 7))
plt.plot(k, num1, marker='.')
x_ticks = np.linspace(1, 10, 10)
plt.xticks(x_ticks)
y_ticks = np.linspace(0, 30, 16)
plt.yticks(y_ticks)
plt.title('Q7a: Expected number of boxes')
plt.xlabel('k')
plt.ylabel('number')
plt.grid(ls='--')
plt.show()

# %% question 7b
plt.figure(figsize=(10, 7))
plt.plot(k, num2, marker='.')
x_ticks = np.linspace(1, 10, 10)
plt.xticks(x_ticks)
y_ticks = np.linspace(0, 30, 16)
plt.yticks(y_ticks)
plt.title('Q7b: Expected number of boxes')
plt.xlabel('N')
plt.ylabel('number')
plt.grid(ls='--')
plt.show()


# %% question 8: Tic-tac-toe
class TicTac:
    def __init__(self):
        self.res_pos = 9
        self.plate = [[0] * 3 for _ in range(3)]
        self.Aw = 0
        self.Bw = 0
        self.draw = 0
        self.ans = set()

    def tictactoe(self, ai=False):
        def update(a=1):  # a=1 represent update Alice
            if ai is False:
                return self.pos()
            else:
                if a == 1:
                    return self.A_pos_AI()
                else:
                    return self.B_pos_AI()
        res = [0]*4
        self.Aw, self.Bw, self.draw, competition, dur = 0, 0, 0, 0, 0
        start = time.perf_counter()
        while dur < 1:
            self.res_pos = 9
            self.plate = [[0] * 3 for _ in range(3)]
            if np.random.randint(2) == 1:
                x1, y1 = update(a=1)
                self.plate[x1][y1] = 1
                while True:
                    x2, y2 = update(a=0)
                    self.plate[x2][y2] = 2
                    if self.who_win() is True:
                        break
                    x1, y1 = update(a=1)
                    self.plate[x1][y1] = 1
                    if self.who_win() is True:
                        break
            else:
                x2, y2 = update(a=0)
                self.plate[x2][y2] = 2
                while True:
                    x1, y1 = update(a=1)
                    self.plate[x1][y1] = 1
                    if self.who_win() is True:
                        break
                    x2, y2 = update(a=0)
                    self.plate[x2][y2] = 2
                    if self.who_win() is True:
                        break
            competition += 1
            dur = time.perf_counter()-start
        res[0], res[1], res[2], res[3] = self.Aw/competition, self.Bw/competition, self.draw/competition, competition
        print('AI status:', ai)
        print('Alice’s winning probability:', res[0])
        print('Bob’s winning probability:', res[1])
        print('The probability of drawing:', res[2])
        print('The number of competitions:', res[3])
        print('Time cost:', dur)
        return res, dur

    def pos(self):
        while self.res_pos > 0:
            i, j = np.random.randint(3), np.random.randint(3)
            if self.plate[i][j] == 0:
                self.res_pos -= 1
                return i, j

    def get_cur_result(self):
        self.ans = set()
        for i in range(3):
            self.ans.add(self.plate[i][0] ** 2 + self.plate[i][1] ** 2 + self.plate[i][2] ** 2)
            self.ans.add(self.plate[0][i] ** 2 + self.plate[1][i] ** 2 + self.plate[2][i] ** 2)
        self.ans.add(self.plate[0][0] ** 2 + self.plate[1][1] ** 2 + self.plate[2][2] ** 2)
        self.ans.add(self.plate[0][2] ** 2 + self.plate[1][1] ** 2 + self.plate[2][0] ** 2)
        return None

    def who_win(self):
        self.get_cur_result()
        if 3 in self.ans and self.res_pos >= 0:
            self.Aw += 1
            return True
        elif 12 in self.ans and self.res_pos >= 0:
            self.Bw += 1
            return True
        elif self.res_pos == 0 and (3 not in self.ans) and (12 not in self.ans):
            self.draw += 1
            return True
        else:
            return False

    def win_pos(self, p):  # p = 2 or 8
        for i in range(3):
            if self.plate[i][0]**2 + self.plate[i][1]**2 + self.plate[i][2]**2 == p:
                return i, self.plate[i].index(0)
            if self.plate[0][i]**2 + self.plate[1][i]**2 + self.plate[2][i]**2 == p:
                d1 = [self.plate[0][i], self.plate[1][i], self.plate[2][i]]
                return d1.index(0), i

        if self.plate[0][0]**2 + self.plate[1][1]**2 + self.plate[2][2]**2 == p:
            d2 = [self.plate[0][0], self.plate[1][1], self.plate[2][2]]
            return d2.index(0), d2.index(0)

        if self.plate[0][2]**2 + self.plate[1][1]**2 + self.plate[2][0]**2 == p:
            d3, d4 = [self.plate[0][2], self.plate[1][1], self.plate[2][0]**2], [[0, 2], [1, 1], [2, 0]]
            return d4[d3.index(0)][0], d4[d3.index(0)][1]

    def A_pos_AI(self):
        self.get_cur_result()
        if 2 in self.ans:
            self.res_pos -= 1
            return self.win_pos(2)
        elif 8 in self.ans:
            self.res_pos -= 1
            return self.win_pos(8)
        else:
            return self.pos()

    def B_pos_AI(self):
        if self.plate[1][1] == 0:
            self.res_pos -= 1
            return 1, 1
        else:
            return self.pos()


t = TicTac()
t.tictactoe(ai=False)
t.tictactoe(ai=True)


# %% question 9:

square = lambda x: x * x


def timeit_custom(repeat, loops):
    times = []
    for i in range(repeat):
        start = time.perf_counter()
        for j in range(loops):
            square(1)
        end = time.perf_counter()
        times += [(end - start) / loops] * loops

    print(round(np.mean(times) * 10 ** 9, 0), 'ns', u"\u00B1", round(math.sqrt(np.var(times)) * 10 ** 9, 4),
          'ns per loop', '(mean', u"\u00B1",  'std. dev. of', repeat, 'runs,', loops, 'loops each)')
    return np.mean(times), math.sqrt(np.var(times))


timeit_custom(7, 10000000)


# %% question 10:
def quadratic(a,  b,  c):
    e = b**2-4*a*c
    if e < 0:
        e = abs(complex(e))
        j = complex(0, 1)
        ans1 = (-b + j * math.sqrt(e)) / (2 * a)
        ans2 = (-b - j * math.sqrt(e)) / (2 * a)
    else:
        ans1 = (-b + math.sqrt(e)) / (2 * a)
        ans2 = (-b - math.sqrt(e)) / (2 * a)

    print('(', ans1, ', ', ans2, ')', sep='')
    return None


quadratic(1, -3, 2)
quadratic(4, -4, 1)
quadratic(1, -2, 2)
quadratic(1, 0, 4)

