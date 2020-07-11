
import numpy as np
from calc_map import calc_map

results = np.load('results.npy', allow_pickle=True).item()

# print (np.shape(results['true']), np.shape(results['pred']))
# print (np.std(results['true']), np.std(results['pred']))

calc_map(results['true'], results['pred'])


