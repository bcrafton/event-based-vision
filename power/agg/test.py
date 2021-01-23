
import numpy as np
import matplotlib.pyplot as plt
import argparse

################################################

parser = argparse.ArgumentParser(description='Process corresponding arguments')
parser.add_argument('-e','--events_path',required=False,default='./data/550.npy',help='Path to events numpy file')
parser.add_argument('-t','--temporal_stacking',type=int,required=False,default=12,help='number of event frames to be stacked temporally before inference')
parser.add_argument('-c','--cache_size',type=int,required=False,default=2**14,help='Size of aggregator cache')

args = parser.parse_args()
print('Event data  : {}'.format(args.events_path))
print('# frames temporally stacked  : {}'.format(args.temporal_stacking))
print('cache size  : {}'.format(args.cache_size))

################################################

def loadData(event_seq=args.events_path,temporal_stacking=args.temporal_stacking,cache_size=args.cache_size):
	# events = np.load('./data/550.npy', allow_pickle=True).item()
	events = np.load(event_seq, allow_pickle=True).item()
	xs = events['x']
	ys = events['y']
	
	################################################
	
	addrs = []
	for frame in range(temporal_stacking):
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
	cache = LRUCache(cache_size)
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
	print(len(addrs))
	# assert(False)
	print ('Average number of evictions : {}'.format(np.average(evicts)))
	print ('hits: {}'.format(hits))
	print ('misses: {}'.format(misses))
	return hits,misses

################################################

def getEnrgy(hits,misses,enrgy_per_dram=2e-9,enrgy_per_sram=5.2e-13): 
	#femto joules numbers obtained from MDAT paper
	# enrgy_per_sram = 1
	# enrgy_per_dram = 10
	
	# assumed direct indexed cache and neglected the energy of the interconnect
	# on a single miss read sram saw miss read dram and then wrote back to sram
	total_enrgy_misses = misses *(2*enrgy_per_sram + enrgy_per_dram) 
	total_enrgy_hits = hits * enrgy_per_sram
	total_enrgy = total_enrgy_hits + total_enrgy_misses
	no_agg_enrgy = (hits + misses) * enrgy_per_dram   	
 
	print("Energy due to SRAM accesses : {} joules".format(total_enrgy_hits))
	print("Energy due to DRAM accesses : {} joules".format(total_enrgy_misses))
	print("Total energy utilization of frame aggregator: {} joules\n".format(total_enrgy))
	print("Energy assuming there was no aggregator : {} joules".format(no_agg_enrgy))
	print("################################################\n")

	enrgy_dict = {'dram': total_enrgy_misses,'sram':total_enrgy_hits,'total':total_enrgy}

	return enrgy_dict

################################################

# getEnrgy()
