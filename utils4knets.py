import scipy.io as sio
from scipy.spatial import distance
import numpy as np

# Takes the rows of a matrix and returns the matrix with rows normalized to a length of one.
def normr(x):
    len = np.sqrt(np.sum(x * x, axis=1))
    len = len.reshape(np.shape(len)[0], 1)
    zeroRows = np.argwhere(len == 0)
    zeroRows = zeroRows[:, 0]

    # Turn off warning in case of division by zero
    old_settings = np.seterr(all='ignore')  # seterr to known value
    np.seterr(divide='ignore', invalid='ignore')

    nx = x / len

    np.seterr(**old_settings)  # reset to default

    if np.shape(zeroRows)[0] != 0:
        numColumns = np.shape(x)[1]
        row = np.ones((1, numColumns)) / np.sqrt(numColumns)
        nx[zeroRows, :] = np.tile(row, (np.shape(zeroRows)[0], 1))

    return nx


# Takes the rows of a matrix and returns the matrix with rows normalized to a length of one.
def normc(x):
    len = np.sqrt(np.sum(x * x, axis=0))
    len = len.reshape(1, np.shape(len)[0])
    zeroColumns = np.argwhere(len == 0)
    zeroColumns = zeroColumns[:, 1]

    # Turn off warning in case of division by zero
    old_settings = np.seterr(all='ignore')  # seterr to known value
    np.seterr(divide='ignore', invalid='ignore')

    nx = x / len

    np.seterr(**old_settings)  # reset to default

    if np.shape(zeroColumns)[0] != 0:
        numRows = np.shape(x)[0]
        row = np.ones((numRows, 1)) / np.sqrt(numRows)
        nx[:, zeroColumns] = np.tile(row, (1, np.shape(zeroColumns)[0]))

    return nx


# Break a range of contigues numbers from 1 to N into sets of size step. The last could be smaller or larger than step.
def set2parts(N, step):
    if step > N:
        step = N
    s = np.arange(0, N - step + 1, step)
    e = s + step
    e[-1] = N
    return np.array([s, e])
    # return s, e


# Classic binary search
def bsearch(b, num):
    left = 0
    right = int(np.shape(b)[0])
    index = -1
    while left <= right:
        mid = (left + right) // 2
        if b[mid] == num:
            index = mid
            break
        elif b[mid] > num:
            right = mid - 1
        else:
            left = mid + 1
    return index


# Find all occurences of a value in a vector, based on binary search
def bsfreq(vals, num):
    ind = bsearch(vals, num)
    n = int(np.shape(vals)[0])

    # Count elements
    # on left side.
    count = 1
    left = ind - 1
    while (left >= 0 and
           vals[left] == num):
        count += 1
        left -= 1

    # Count elements on
    # right side.
    right = ind + 1
    while (right < n and
           vals[right] == num):
        count += 1
        right += 1

    # print(vals[left+1:right])
    # print(range(left+1, right))

    return left + 1, right

