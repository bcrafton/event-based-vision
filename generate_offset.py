
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

#################################################

x = np.array(range(0, 14)).reshape(1, -1)
x = np.repeat(x, 8, axis=0) * 32
# print (x)
# print (np.shape(x))

y = np.array(range(0, 8)).reshape(-1, 1)
y = np.repeat(y, 14, axis=1) * 32
# print (y)
# print (np.shape(y))

# yx = np.stack((y, x), axis=-1)
# print (yx)

#################################################

out = ''
for i in range(8):
    out += '['
    for j in range(14):
        out += '[%3d, %3d]' % (y[i, j], x[i, j])
        if (j < 13): out += ', '
        else:        out += '],\n'

print (out)

#################################################

