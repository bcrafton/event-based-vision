
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

#################################################

x = np.array(range(0, 12)).reshape(1, -1)
x = np.repeat(x, 10, axis=0) * 24
# print (x)
# print (np.shape(x))

y = np.array(range(0, 10)).reshape(-1, 1)
y = np.repeat(y, 12, axis=1) * 24
# print (y)
# print (np.shape(y))

# yx = np.stack((y, x), axis=-1)
# print (yx)

#################################################

out = ''
for i in range(10):
    out += '['
    for j in range(12):
        out += '[%3d, %3d]' % (y[i, j], x[i, j])
        if (j < 11): out += ', '
        else:        out += '],\n'

print (out)

#################################################

