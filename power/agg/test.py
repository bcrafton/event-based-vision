
import numpy as np
import matplotlib.pyplot as plt
import argparse

################################################

parser = argparse.ArgumentParser(description='Process corresponding arguments')
parser.add_argument('events_path',help='Path to events numpy file')
parser.add_argument('-t','--temporal_stacking',type=int,required=False,default=12,help='number of event frames to be stacked temporally before inference')
parser.add_argument('-c','--cache_size',type=int,required=False,default=2**14,help='Size of aggregator cache')

args = parser.parse_args()
print('Event data  : {}'.format(args.events_path))
print('# frames temporally stacked  : {}'.format(args.temporal_stacking))
print('cache size  : {}'.format(args.cache_size))

################################################

# events = np.load('./data/550.npy', allow_pickle=True).item()
events = np.load(args.events_path, allow_pickle=True).item()
xs = events['x']
ys = events['y']

################################################

addrs = []
for frame in range(args.temporal_stacking):
    offset = frame * 240 * 304
    for event in range(len(xs[frame])):
        x = xs[frame][event]
        y = ys[frame][event]
        addr = offset + y * 304 + x
        addrs.append(addr)

################################################

from LRUCache import LRUCache

evicts = []
hits = 0
misses = 0
cache = LRUCache(args.cache_size)
for addr in addrs:
    if cache.contains(addr):
        cache.access(addr)
        hits += 1
    else:
        evict = cache.add(addr)
        if evict:
            (addr, access) = evict
            evicts.append(access)
            misses += 1

print ('Average number of evictions : {}'.format(np.average(evicts)))
print ('hits: {}'.format(hits))
print ('misses: {}'.format(misses))

################################################

def getEnrgy(enrgy_per_dram=10,enrgy_per_sram=1): 
    #femto joules numbers obtained from MDAT paper
    # enrgy_per_sram = 1
    # enrgy_per_dram = 10
    
    # assumed direct indexed cache and neglected the energy of the interconnect
    # on a single miss read sram saw miss read dram and then wrote back to sram
    total_enrgy_misses = misses *(2*enrgy_per_sram + enrgy_per_dram) 
    total_enrgy_hits = hits * enrgy_per_sram
    total_enrgy = total_enrgy_hits + total_enrgy_misses
    
    print("Energy due to SRAM accesses : {} femtojoules".format(total_enrgy_hits))
    print("Energy due to DRAM accesses : {} femtojoules".format(total_enrgy_misses))
    print("Total energy utilization of frame aggregator: {} femtojoules\n".format(total_enrgy))
    print("################################################\n")

    enrgy_dict = {'dram': total_enrgy_misses,'sram':total_enrgy_hits,'total':total_enrgy}

    return enrgy_dict

################################################

getEnrgy()