import matplotlib.pyplot as plt
from agg.test import getEnrgy, loadData
from cnn.power import get_cnn_enrgy, getModel
import seaborn as sns
import numpy as np

####################################
N = 5
time_per_frame = 33.33e-3 #for thirty frames per second
events_list = ['./agg/data/550.npy','./agg/data/551.npy','./agg/data/552.npy','./agg/data/553.npy','./agg/data/554.npy']
agg_power = np.zeros(N)
cnn_power = np.zeros(N)
camera_power = 0.02 #(90.17e-3)/(5.80e-3) #from paper seems very large compared to toher numbers
model = getModel()
####################################
cnn_dict = get_cnn_enrgy(model,input_size=(240,288,12))

for i in range(0,N):
	hits,misses = loadData(event_seq=events_list[i])
	agg_dict = getEnrgy(hits,misses)
	
	# print('\n\n')
	
	agg_enrgy = agg_dict['total']
	cnn_enrgy = cnn_dict['energy']
	
	agg_power[i] = agg_enrgy/time_per_frame
	cnn_power[i] = (cnn_enrgy/time_per_frame)/25
print(agg_power)
print(cnn_power)

####################################

ind = np.arange(N)
width = 0.35       # the width of the bars: can also be len(x) sequence
sns.set()
def set_sizes(fig_size=(9, 6), font_size=10):
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size+5
    plt.rcParams["legend.fontsize"] = font_size+3
set_sizes((12,8), 15)

####################################

p1 = plt.bar(ind, cnn_power, width)
p2 = plt.bar(ind, camera_power, width,
             bottom=cnn_power)
p3 = plt.bar(ind, agg_power, width,
             bottom=cnn_power)

plt.ylabel('joules/frame')
plt.xlabel('Data Sequences')
plt.title('Power analysis of event based vision system')
plt.xticks(ind, ('550', '551', '552', '553', '554'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0],p3[0]), ('CNN','CAMERA', 'AGG'))

plt.show()

####################################