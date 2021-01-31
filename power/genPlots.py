
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
agg_power = np.zeros(N+1)
cnn_power = np.zeros(N+1)
isp_power = np.zeros(N+1)
cam_list = np.ones(N+1)
agg_list = np.ones(N+1)
camera_power = 50e-3 #(90.17e-3)/(5.80e-3) #from paper seems very large compared to toher numbers
dram_power = 65e-3 # from ramulator which uses dram power
opt_cam =180e-3
isp_pwr=150e-3

####################################
#optical camera
model = getModel(3)	
cnn_dict = get_cnn_enrgy(model,input_size=(240,288,3))
cnn_enrgy = cnn_dict['energy']
print(cnn_enrgy)
cnn_power[0] = cnn_enrgy
isp_power[0] = isp_pwr
cam_list = cam_list*camera_power
cam_list[0] = opt_cam

####################################
for i,t in enumerate(temporal_list) :
	model = getModel(t)	
	cnn_dict = get_cnn_enrgy(model,input_size=(240,288,t))
	cnn_enrgy = cnn_dict['energy']
	print(cnn_enrgy)
	cnn_power[i+1] = cnn_enrgy#/time_per_frame

for i in range(0,N):
	hits,misses = loadData(event_seq=events_list[i])
	agg_dict = getEnrgy(hits,misses)
	agg_enrgy = agg_dict['total']	
	agg_power[i] = agg_enrgy/time_per_frame

agg_power = np.mean(agg_power)
print(agg_power)
print(cnn_power)
agg_list  = agg_list * agg_power
agg_list[0] = 0

####################################

ind = np.arange(N+1)
width = 0.5

####################################

'''
p1 = plt.bar(ind, cam_list,   width,                                                color='#f07878')
p2 = plt.bar(ind, cnn_power,  width, bottom=cam_list,                               color='#888888')
p3 = plt.bar(ind, dram_power, width, bottom=cnn_power+cam_list,                     color='#DDDDDD')
p4 = plt.bar(ind, agg_list,   width, bottom=cnn_power+cam_list+dram_power,          color='black')
p5 = plt.bar(ind, isp_power,  width, bottom=cnn_power+cam_list+agg_list+dram_power, color='#AA99FF')
'''

####################################
'''
p1 = plt.bar(ind, cam_list,   width,                                                color='#f07878')
p2 = plt.bar(ind, agg_list,   width, bottom=cam_list,                               color='black')
p3 = plt.bar(ind, isp_power,  width, bottom=cam_list+agg_list,                      color='black')

p4 = plt.bar(ind, cnn_power,  width, bottom=cam_list+agg_list+isp_power,            color='#888888')
p5 = plt.bar(ind, dram_power, width, bottom=cam_list+agg_list+isp_power+cnn_power,  color='#DDDDDD')
'''
####################################
# '''
p1 = plt.bar(ind, cam_list,   width,                                                color='#f07878')
p2 = plt.bar(ind, agg_list,   width, bottom=cam_list,                               color='#4169e1')
p3 = plt.bar(ind, isp_power,  width, bottom=cam_list+agg_list,                      color='#4169e1')

p4 = plt.bar(ind, cnn_power,  width, bottom=cam_list+agg_list+isp_power,            color='#888888')
p5 = plt.bar(ind, dram_power, width, bottom=cam_list+agg_list+isp_power+cnn_power,  color='#DDDDDD')
# '''
####################################
'''
p1 = plt.bar(ind, cam_list,   width,                                                color='#f07878')
p2 = plt.bar(ind, agg_list,   width, bottom=cam_list,                               color='#4b96ff')
p3 = plt.bar(ind, isp_power,  width, bottom=cam_list+agg_list,                      color='#4b96ff')

p4 = plt.bar(ind, cnn_power,  width, bottom=cam_list+agg_list+isp_power,            color='#888888')
p5 = plt.bar(ind, dram_power, width, bottom=cam_list+agg_list+isp_power+cnn_power,  color='#DDDDDD')
'''
####################################

yticks = [0.00, 0.10, 0.20, 0.30,0.40,0.50,0.60]
plt.yticks(yticks, len(yticks) * [''])
plt.ylim(bottom=0., top=0.625)
plt.grid(True, axis='y', linestyle=(0, (5, 8)), color='k')#(0, (1, 10))

xticks = [0, 1, 2, 3, 4]
plt.xticks(xticks, len(xticks) * [''])

plt.gcf().set_size_inches(4., 2.75)
plt.tight_layout(0.)
plt.gcf().savefig('power.png', dpi=500)

####################################

#calculate percentages
total = np.zeros(N+1)
delta = np.zeros(N+1)
frac = np.zeros(N+1)
cnn_percentages = np.zeros(N+1)
cam_percentages = np.zeros(N+1)
agg_percentages = np.zeros(N+1)
dram_percentages = np.zeros(N+1)
isp_percentages = np.zeros(N+1)

for i in range(len(cnn_power)):
	total[i] = cnn_power[i] + cam_list[i] + agg_list[i] + dram_power+isp_power[i] 
	cnn_percentages[i] = cnn_power[i]/total[i]
	cam_percentages[i] = cam_list[i]/total[i]
	agg_percentages[i] = agg_list[i]/total[i]
	dram_percentages[i] = dram_power/total[i]
	isp_percentages[i] = isp_power[i]/total[i]

for i in range(len(total)):
	delta[i] = (total[0] - total[i]) / total[0]

for i in range(len(total)):
	frac[i] = total[i] / total[0]

print("\n")
print("Percentages for cnn: {}".format(cnn_percentages))
print("Percentages for camera: {}".format(cam_percentages))
print("Percentages for agg: {}".format(agg_percentages))
print("Percentages for dram: {}".format(dram_percentages))
print("Percentages for isp: {}".format(isp_percentages))
print("Deltas percentage for all configurations with relation to an optical camera: {}".format(delta))
print("{}".format(frac))










