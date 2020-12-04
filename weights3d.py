
import numpy as np

x = np.load('1_0.030000_weights.npy', allow_pickle=True).item()

print (x.keys())

weights = {}
for key in x.keys():
    if type(key) == int:
        weights[key + 2] = x[key]
weights[0] = x['0_3d']
weights[1] = x['2_3d']
weights[2] = x['4_3d']

print (weights.keys())

