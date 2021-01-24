
import matplotlib.pyplot as plt
from agg.test import getEnrgy, loadData
from cnn.power import get_cnn_enrgy, getModel
import seaborn as sns
import numpy as np

####################################

temporal_list = [12,8,4,1]
N = len(temporal_list)
time_per_frame = 33.33e-3 #for thirty frames per second
events_list = ['./agg/data/550.npy','./agg/data/551.npy','./agg/data/552.npy','./agg/data/553.npy','./agg/data/554.npy']
agg_power = np.zeros(N)
cnn_power = np.zeros(N)
camera_power = 50e-3 #(90.17e-3)/(5.80e-3) #from paper seems very large compared to toher numbers

####################################
for i,t in enumerate(temporal_list) :
	model = getModel(t)	
	cnn_dict = get_cnn_enrgy(model,input_size=(240,288,t))
	cnn_enrgy = cnn_dict['energy']
	print(cnn_enrgy)
	cnn_power[i] = cnn_enrgy#/time_per_frame

for i in range(0,N):
	hits,misses = loadData(event_seq=events_list[i])
	agg_dict = getEnrgy(hits,misses)
	agg_enrgy = agg_dict['total']	
	agg_power[i] = agg_enrgy/time_per_frame

agg_power = np.mean(agg_power)

print(agg_power)
print(cnn_power)

####################################

dram_power = 65e-3 # from ramulator which uses dram power

####################################

ind = np.arange(N)
width = 0.35

####################################

p1 = plt.bar(ind, cnn_power, width,color='k')
p2 = plt.bar(ind, camera_power, width, bottom=cnn_power,color='#252850')
p3 = plt.bar(ind, agg_power, width, bottom=cnn_power+camera_power,color='#f07878')
p4 = plt.bar(ind, dram_power, width, bottom=cnn_power+camera_power+agg_power,color='#d0d2d1')

# plt.show()

yticks = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
plt.yticks(yticks, len(yticks) * [''])
plt.ylim(bottom=0., top=0.30)
plt.grid(True, axis='y', linestyle=(0, (5, 8)), color='black')

xticks = [0, 1, 2, 3]
plt.xticks(xticks, len(xticks) * [''])

plt.gcf().set_size_inches(4., 2.75)
plt.tight_layout(0.)
plt.gcf().savefig('power.png', dpi=500)

####################################














