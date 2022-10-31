# %% question 1: Chess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(20908272)


# %% (a)
class Chess:
    def __init__(self):
        print('')

    def draw(self):
        plt.figure(figsize=(5, 5))
        light_brown = (255 / 255, 206 / 255, 158 / 255)
        deep_brown = (209 / 255, 139 / 255, 71 / 255)
        my_camp = matplotlib.colors.LinearSegmentedColormap.from_list('my_camp', [light_brown, deep_brown], 5)
        num_x, num_y = 8, 8
        panel = np.zeros([num_x, num_y], float)
        for nx in range(num_x):
            for ny in range(num_y):
                if (nx + ny) % 2 == 0:
                    panel[nx, ny] = 1
        plt.imshow(panel, cmap=my_camp)
        dic1 = {-0.375: '\u2656', 0.625: '\u2658', 1.625: '\u2657', 2.625: '\u2655',
                3.625: '\u2654', 4.625: '\u2657', 5.625: '\u2658', 6.625: '\u2656'}
        dic2 = {-0.375: '\u265C', 0.625: '\u265E', 1.625: '\u265D', 2.625: '\u265B',
                3.625: '\u265A', 4.625: '\u265D', 5.625: '\u265E', 6.625: '\u265C'}
        for i in range(len(dic1)):
            plt.text(-0.375 + i, -0.375, dic1[-0.375 + i], size=30)
            plt.text(-0.375 + i, 6.625, dic2[-0.375 + i], size=30)
        for i in range(8):
            plt.text(-0.375 + i, 0.625, '\u2659', size=30)
            plt.text(-0.375 + i, 5.625, '\u265F', size=30)
        plt.xticks(np.linspace(0, 7, 8), ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), fontsize=20)
        plt.xlim([-0.5, 7.5])
        plt.yticks(np.linspace(0, 7, 8), ('1', '2', '3', '4', '5', '6', '7', '8'), fontsize=20)
        plt.ylim([-0.5, 7.5])
        plt.tick_params(bottom=False, left=False, labeltop=True, labelright=True)
        plt.show()


Chess().draw()


# %%(b)
class Chess:
    def __init__(self, board):
        self.board = board

    def draw(self):
        plt.figure(figsize=(5, 5))
        light_brown = (255 / 255, 206 / 255, 158 / 255)
        deep_brown = (209 / 255, 139 / 255, 71 / 255)
        my_camp = matplotlib.colors.LinearSegmentedColormap.from_list('my_camp', [light_brown, deep_brown], 5)
        num_x, num_y = 8, 8
        panel = np.zeros([num_x, num_y], float)
        for nx in range(num_x):
            for ny in range(num_y):
                if (nx + ny) % 2 == 0:
                    panel[nx, ny] = 1
        plt.imshow(panel, cmap=my_camp)
        if np.all(self.board == 0):
            dic1 = {-0.375: '\u2656', 0.625: '\u2658', 1.625: '\u2657', 2.625: '\u2655',
                    3.625: '\u2654', 4.625: '\u2657', 5.625: '\u2658', 6.625: '\u2656'}
            dic2 = {-0.375: '\u265C', 0.625: '\u265E', 1.625: '\u265D', 2.625: '\u265B',
                    3.625: '\u265A', 4.625: '\u265D', 5.625: '\u265E', 6.625: '\u265C'}
            for i in range(len(dic1)):
                plt.text(-0.375 + i, -0.375, dic1[-0.375 + i], size=30)
                plt.text(-0.375 + i, 6.625, dic2[-0.375 + i], size=30)
            for i in range(8):
                plt.text(-0.375 + i, 0.625, '\u2659', size=30)
                plt.text(-0.375 + i, 5.625, '\u265F', size=30)
        else:
            dic = {-1: '\u265F', -2: '\u265E', -3: '\u265D', -4: '\u265C', -5: '\u265B', -6: '\u265A',
                   1: '\u2659', 2: '\u2658', 3: '\u2657', 4: '\u2656', 5: '\u2655', 6: '\u2654', 0: ''}
            for i in range(len(self.board[0])):
                for j in range(len(self.board)):
                    plt.text(-0.375 + i, 6.625 - j, dic[self.board[j][i]], size=30)

        plt.xticks(np.linspace(0, 7, 8), ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), fontsize=20)
        plt.xlim([-0.5, 7.5])
        plt.yticks(np.linspace(0, 7, 8), ('1', '2', '3', '4', '5', '6', '7', '8'), fontsize=20)
        plt.ylim([-0.5, 7.5])
        plt.tick_params(bottom=False, left=False, labeltop=True, labelright=True)
        plt.show()


DEEPBLUE = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 4],
                     [0, 0, 0, 0, 0, -5, 0, -6],
                     [0, 0, 0, 5, 0, 0, 2, 0],
                     [0, 0, 0, -1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, -1, 1, 1],
                     [0, 0, 0, 0, 0, -2, 0, 6],
                     [0, 0, 0, 0, -4, 0, 0, 0]])

Chess(board=np.zeros((8, 8))).draw()
Chess(board=DEEPBLUE).draw()

# %% question 2: Supermarket


# %% question 3: Artificial landscape
from matplotlib.cm import terrain
from matplotlib.colors import ListedColormap as LC

terrain_above_sea = LC(terrain(np.linspace(.15, 1, 256)))


def generate_landscape():
    """ Generate the altitudes of a random but smooth landscape.
        You do not need to understand how they are generated.

    Returns
    -------
    z: ndarray of float
        A 101*101 array of altitude.
        The sea level has an altitude of 0.

        Using the Cartesian coordinates (x, y),
        the element z[i, j] is the altitude at point (x=j/100, y=i/100).

        Note that x is proportional to j, whereas y is proportional to i.
        For example,
            * z[12, 34] is the altitude at point (x=.34, y=.12)
            * z[56, :] are the altitudes along y=.56
            * z[:, 78] are the altitudes along x=.78
    """
    n, N = 20, 101
    x = y = np.arange(0, n, 1)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros((n, n))
    zz[:, 0], zz[0, :] = np.random.randn(2, n)
    for i in range(1, n):
        for j in range(1, n):
            zz[i, j] = (zz[i - 1, j] + zz[i, j - 1]) / 2 + np.random.randn()

    from scipy import interpolate
    from warnings import filterwarnings
    filterwarnings('ignore')
    x, y = np.linspace(0, n - 1, N), np.linspace(0, n - 1, N)
    return interpolate.bisplev(
        x, y, interpolate.bisplrep(
            xx.flatten(), yy.flatten(), zz.flatten(), s=1))


# %% (a)
np.random.seed(20908272)
plt.figure(figsize=(10, 10))
x = y = np.arange(0, 1.01, 0.01)
X, Y = np.meshgrid(x, y)
Z = generate_landscape()
lis = np.linspace(0, np.max(Z), 13, endpoint=True)
lev = [i for i in lis]
plt.contourf(X, Y, Z, lev, cmap=terrain_above_sea, extend='min')
contour = plt.contour(X, Y, Z, lev, alpha=0.7, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=10, colors='k', fmt='%.3f')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% (b)
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(20908272)
ori = generate_landscape()


def get_gif(c):
    plt.cla()
    d = ori[:, c]
    dmax, dmin, l = np.max(d), np.min(d), len(d)
    p_max, p_min = np.max(ori), np.min(ori)
    x = np.linspace(0, 100, l)
    x0 = list(map(lambda y: max(0, y), d))
    plt.imshow([[p_max, p_max], [0, 0]], cmap=terrain_above_sea, extent=[0, 100, 0, p_max],
               interpolation="bicubic", aspect="auto")
    plt.plot(x, np.zeros(l), color="black", linestyle='dashed')
    plt.fill_between(x, p_min, 0, color="#4293f6")
    plt.fill_between(x, x0, p_max, color="white")
    plt.plot(x, d, color='black')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.ylim([p_min, p_max + 1])
    plt.xlim([0, 100])
    plt.xticks(np.linspace(0, 100, 6), np.round(np.linspace(0, 1, 6), 2))
    plt.text(4, p_max, f"y={c / 100:.2f}", fontsize=10)


figu = plt.figure()
gif = FuncAnimation(fig=figu, func=get_gif, frames=np.arange(0, 101))
gif.save("Q3_2.gif")

# %% question 4:
import numpy as np
import matplotlib.pyplot as plt


def get_prime(n):
    primes = [True] * (n + 1)
    p = 2
    while p ** 2 <= n:
        if primes[p]:
            for i in range(p ** 2, n + 1, p):
                primes[i] = False
        p += 1
    primes = [item for item in range(2, n) if primes[item]]
    return primes


def goldbach(n):
    total = get_prime(n)
    res = []
    for p in total:
        if p <= n // 2 and n - p in total:
            res.append(p)
    return res


def isPrime(x):
    if x <= 1:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
    i += 1
    return True


plt.figure(figsize=(12, 9))
lis_n = [x for x in range(4, 10 ** 6 + 1, 1)]
for n in lis_n:
    y = len(goldbach(n))
    if n % 105 == 0:
        plt.plot(n, y, marker='s', c='grey', markersize=2)
    elif n % 15 == 0:
        plt.plot(n, y, marker='s', c='yellow', markersize=2)
    elif n % 21 == 0:
        plt.plot(n, y, marker='s', c='cyan', markersize=2)
    elif n % 35 == 0:
        plt.plot(n, y, marker='s', c='magenta', markersize=2)
    else:
        if n % 3 == 0:
            plt.plot(n, y, marker='s', c='green', markersize=2)
        if n % 5 == 0:
            plt.plot(n, y, marker='s', c='red', markersize=2)
        if n % 7 == 0:
            plt.plot(n, y, marker='s', c='blue', markersize=2)
        if n % 3 != 0 and n % 5 != 0 and n % 7 != 0:
            if n % 2 == 0 and isPrime(n // 2):
                plt.plot(n, y, marker='s', c='orange', markersize=2)
            else:
                plt.plot(n, y, marker='s', c='black', markersize=2)

ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='x')
plt.xlabel('n')
plt.ylabel('number of ways')
# plt.savefig('Q4.png')
plt.show()

# %% question 5: Image processing
# %pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

I = plt.imread('./lenna.png')
plt.title('original', size=20)
plt.imshow(I)

I_grey = (0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2]) * 255
plt.title('grey', size=20)
plt.imshow(I_grey, cmap='gray')

n = 9
box_filter = np.ones((n, n)) / n ** 2
I_blur = cv2.filter2D(I, -1, box_filter)
plt.title('blur', size=20)
plt.imshow(I_blur)

identity_filter = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]))
Laplacian_filter = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
k = 2
F_filter = identity_filter + k * Laplacian_filter
I_sharp = cv2.filter2D(I, -1, F_filter)
I_sharp[np.where(I_sharp > 1)] = 1
I_sharp[np.where(I_sharp < 0)] = 0
plt.title('sharp', size=20)
plt.imshow(I_sharp)

S_x = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]))
S_y = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))
E_x = cv2.filter2D(I_grey, -1, S_x)
E_y = cv2.filter2D(I_grey, -1, S_y)
I_edge = np.sqrt(np.square(E_x) + np.square(E_y))
plt.title('edge', size=20)
plt.imshow(I_edge, cmap='gray')

plt.figure(figsize=(15, 10))

plt.axes([0.4, 0.4, 0.2, 0.25])
plt.imshow(I)
plt.title('original', size=20)
plt.xticks(())
plt.yticks(())

plt.axes([0.2, 0.55, 0.2, 0.25])
plt.imshow(I_grey, cmap='gray')
plt.title('gray', size=20)
plt.xticks(())
plt.yticks(())

plt.axes([0.2, 0.25, 0.2, 0.25])
plt.imshow(I_blur)
plt.title('blur', size=20)
plt.xticks(())
plt.yticks(())

plt.axes([0.6, 0.55, 0.2, 0.25])
plt.imshow(I_sharp)
plt.title('sharp', size=20)
plt.xticks(())
plt.yticks(())

plt.axes([0.6, 0.25, 0.2, 0.25])
plt.imshow(I_edge, cmap='gray')
plt.title('edge', size=20)
plt.xticks(())
plt.yticks(())

plt.show()


# %% question 6: Minkowski's question mark function
def f(x, k):
    lis = []
    for i in range(k):
        t = np.floor(x)
        if t != x:
            lis.append(t)
            x = 1 / (x - t)
        else:
            break
    return lis


def g(x, k):
    s = 0
    lis = f(x, k)
    if lis is None:
        return x
    for i in range(len(lis)):
        s += lis[0] if i == 0 else 2 * pow(-1, i + 1) / (2 ** sum(lis[1:i + 1]))
    return s


print(g((np.sqrt(3) - 1) / 2, 20), '----', 2 / 7)
print(g((np.sqrt(5) - 1) / 2, 20), '----', 2 / 3)
print(g((np.sqrt(3)) / 2, 20), '----', 84 / 85)
print(g((np.sqrt(2)) / 2, 20), '----', 4 / 5)

x = np.linspace(0, 1.01, 100)
y = list(map(lambda i: g(i, 10), x))
plt.plot(x, y, c='black')
plt.xlabel("x")
plt.ylabel("?(x)")
plt.show()


# %% question 7: Bifurcation diagram
from matplotlib.animation import FuncAnimation


def p(t):
    l, pos = [], []
    for i in range(10 ** 5):
        r = np.random.uniform(0, 4, size=1)
        l.append(r)
        x0 = np.random.uniform(0, 1, size=1)
        for j in range(t):
            x0 = r * x0 * (1 - x0)
        pos.append(x0)
    plt.cla()
    plt.ylim(0, 1)
    plt.xlim(0, 4)
    plt.scatter(l, pos, s=0.7, c='black')
    plt.text(0.2, 0.8, f"t={t}", fontsize=12)
    plt.text(0.2, 0.9, f"$x_(t+1)=rx_t(1-x_t)$", fontsize=12)


fig = plt.figure()
anim = FuncAnimation(fig=fig, func=p, frames=np.arange(0, 51))
anim.save("Q7.gif")
anim
