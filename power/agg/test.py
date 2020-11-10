
import numpy as np
import matplotlib.pyplot as plt

################################################

events = np.load('./data/550.npy', allow_pickle=True).item()
xs = events['x']
ys = events['y']

################################################

addrs = []
for frame in range(12):
    offset = frame * 240 * 304
    for event in range(len(xs[frame])):
        x = xs[frame][event]
        y = ys[frame][event]
        addr = offset + y * 304 + x
        addrs.append(addr)

################################################

from LRUCache import LRUCache

evicts = []
cache = LRUCache(2 ** 14)
for addr in addrs:
    if cache.contains(addr):
        cache.access(addr)
    else:
        evict = cache.add(addr)
        if evict:
            (addr, access) = evict
            evicts.append(access)

print (np.average(evicts))

################################################
