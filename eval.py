
import numpy as np
from calc_map import calc_map

results = np.load('results.npy', allow_pickle=True).item()
id = results['id']
true = results['true']
pred = results['pred']

#print (np.shape(true))
#print (np.shape(pred))

###########################3

calc_map(id, true, pred)

###########################

