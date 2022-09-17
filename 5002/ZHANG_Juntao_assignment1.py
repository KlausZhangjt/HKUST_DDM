# -*- coding: utf-8 -*-
"""
ZHANG Juntao

5002 Assignment-1
"""
# %% (question 1)


def get_input(question, max_input_times, default_value):
    input_right = 0
    count_num = 0
    while input_right == 0:
        h_tmp = input(question)
        try:
            val: float = float(h_tmp)
            input_right = 1
        except ValueError:
            count_num += 1
            print("Your input should be a number")
        else:
            if val < 0:
                count_num += 1
                print("Your input should be a POSITIVE number")
                input_right = 0

        if count_num > max_input_times:
            val = default_value
            print("You are so stupid! I have to stop you and set the initial height as ", default_value, "meters.")
            break
    return val


h = get_input('Enter the height: ', 2, 10)

t = get_input('Enter the time: ', 1, 3.14)

h, t


#%% (question 2)

import datetime

TODAY = str(datetime.date.today())


def my_copyright4(name, email, date=TODAY):
    len_name = len(name)
    a = len_name + 37
    if a % 2 == 0:
        b_left = (a - 30) // 2
        b_right = (a - 30) // 2
    else:
        b_left = (a - 31) // 2
        b_right = (a - 30) - b_left

    print("*" * a)
    print("***  programmed by", name, "for MSDM5002  ***")
    print("***  ", " " * b_left, "date:", date, " " * b_right, "  ***")
    print("***", "-" * (a - 6), "***", sep='')

    sen = "You can use it as you like, but there might be many bugs. If you find some bugs, please send them to "
    o_len = len(sen)
    sen += '"'
    sen += email
    sen += '"'
    tor = a - 10
    i, j = tor - 1, 0
    while j < len(sen) - tor + 1:
        if (o_len - j) > tor:
            while sen[i] != ' ':
                i -= 1

            print("***  ", sen[j:i], ' ' * (tor - i + j), "  ***", sep='')
            j = i + 1
            i = j + tor
        else:
            print("***  ", sen[j:j + tor], "  ***", sep='')
            j += tor
    print("***  ", sen[j:], ' ' * (tor - len(sen[j:])), "  ***", sep='')

    print("*" * a)


my_copyright4('IA', 'ia@ust.hk')  
  
my_copyright4('Alice & Bob', 'alice@wonder.land', '2022-12-31')



#%% (question 3)


nc = ["a", "an", "and", "as", "at", "but", "by", "for",
      "in", "nor", "of", "on", "or", "the", "to", "up"]


def capitalize():

    sen = input("Enter a sentence: ")
    sen1 = sen.split(' ')
    for i in range(len(sen1)):
        if sen1[i] not in nc:
            sen1[i] = sen1[i][:1].upper() + sen1[i][1:]
    sen1 = ' '.join(sen1)

    sen2 = sen1.split('-')
    for j in range(len(sen2)):
        if sen2[j] not in nc:
            sen2[j] = sen2[j][:1].upper() + sen2[j][1:]
    sen2 = '-'.join(sen2)

    print(sen2)


capitalize()

# test sentence:
# Welcome to the world of data-driven modeling for master students offered by UST on a great and green campus on a hill. 


#%% (question 4)

lis = []

while True:
    name = str(input("Enter a student’s name (Enter q/Q to stop): "))
    lis.append(name)
    if name == 'q' or name == 'Q':
        break

for i in range(len(lis)-1):
    print("Hello ", lis[i], ",", " welcome to the course 5002.", sep='')



#%% (question 5)

from random import randint

heights = [randint(4, 10) for peak in range(3)]

long = 2 * sum(heights) - 5
h = max(heights)
mat = [[' ' for i in range(long)] for j in range(h)]
m, n = h - 1, 0
poi = [heights[0]-1, 2*heights[0]-2, 2*heights[0]+heights[1]-3,
       2*heights[0]+2*heights[1]-4, 2*heights[0]+2*heights[1]+heights[2]-5]

while n < long:
    mat[m][n] = '#'
    if n < poi[0] or poi[1] <= n < poi[2] or poi[3] <= n < poi[4]:
        m -= 1
    elif poi[0] <= n < poi[1] or poi[2] <= n < poi[3]:
        m += 1
    else:
        m += 1
    n += 1

for i in range(h):
    print(''.join(mat[i]))



#%% (question 6)

from random import randint

target = randint(2, 99)


def gauss():
    count = 0
    left, right = 1, 100
    while True:
        s1, s2 = str(left), str(right)
        question = "Guess my number: " + s1 + " to " + s2 + "! "
        ans = int(input(question))

        if ans >= right or ans <= left:
            count += 1
            print("Number out of range! Try again!")
        else:
            if ans == target:
                count += 1
                print("Bingo! You got it in", count, "guesses!")
                break

            elif ans < target:
                left = ans
                count += 1
                continue

            elif ans > target:
                right = ans
                count += 1
                continue
  
            
gauss()

