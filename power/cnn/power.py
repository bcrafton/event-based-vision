
import argparse
import os
import sys

'''
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
'''

####################################

import numpy as np
# import tensorflow as tf
from layers import *
import time
import matplotlib.pyplot as plt
import random

# cacti to get dram power
####################################
'''
# 240, 288
def getModel(temp_stack=12):
	y = model(layers=[
	conv_block((7,7,temp_stack,64), 1), # 240, 288
	
	max_pool(s=3, p=3),
	
	res_block1(64,   64, 1), # 80, 96
	res_block1(64,   64, 1), # 80, 96
	
	max_pool(s=2, p=2),
	
	res_block2(64,   128, 1), # 40, 48
	res_block1(128,  128, 1), # 40, 48
	
	max_pool(s=2, p=2),
	
	res_block2(128,  256, 1), # 20, 24
	res_block1(256,  256, 1), # 20, 24
	
	max_pool(s=2, p=2),
	
	res_block2(256,  512, 1), # 10, 12
	res_block1(512,  512, 1), # 10, 12
	
	max_pool(s=2, p=2),
	
	# res_block2(512,  512, 1), # 5, 6
	# res_block1(512,  512, 1), # 5, 6
	
	dense_block(5*6*512, 2048),
	dense_block(2048, 5*6*12),
	])
	return y
'''
####################################
'''
# https://github.com/bcrafton/ssdfa/blob/master-update3/lib/MobileNet.py
# https://github.com/bcrafton/icsrl-deep-learning/blob/master/image-classification/imagenet224_tf.py

# 240, 288
def getModel(temp_stack=12):
	y = model(layers=[
	conv_block((7,7,temp_stack,64), 1), # 240, 288

	max_pool(s=3, p=3),

	mobile_block( 64,   64, 1), # 80, 96
	mobile_block( 64,  128, 2), # 80, 96

	mobile_block(128,  128, 1), # 40, 48
	mobile_block(128,  256, 2), # 40, 48

	mobile_block(256,  256, 1), # 20, 24
	mobile_block(256,  512, 2), # 20, 24

	mobile_block(512,  512, 1), # 10, 12
	mobile_block(512,  512, 1), # 10, 12
	mobile_block(512,  512, 1), # 10, 12
	mobile_block(512,  512, 1), # 10, 12
	mobile_block(512,  512, 1), # 10, 12

	max_pool(s=2, p=2), 

	dense_block(5*6*512, 2048),
	dense_block(2048, 7*10*12*7),
	])
	return y
'''
####################################

# 240, 288
def getModel(temp_stack=12):
	y = model(layers=[
	conv_block((7,7,temp_stack,32), 1), # 240, 288

	max_pool(s=3, p=3),

	res_block1(32, 64, 1), # 80, 96
	res_block1(64, 64, 1), # 80, 96

	max_pool(s=2, p=2),

	res_block2(64,  128, 1), # 40, 48
	res_block1(128, 128, 1), # 40, 48

	max_pool(s=2, p=2),

	res_block2(128, 256, 1), # 20, 24
	res_block1(256, 256, 1), # 20, 24

	max_pool(s=2, p=2),

	res_block2(256,  512, 1), # 10, 12
	res_block1(512,  512, 1), # 10, 12

	max_pool(s=2, p=2),

	dense_block(5*6*512, 2048),
	dense_block(2048, 5*6*12),
	])
	return y

####################################

def get_cnn_enrgy(model,input_size):
	total_macs = 0
	total_macs,_ = model.forward(input_size)
	total_ops = total_macs * 2
	ops_per_sec = total_ops * 30
	comp_efficiency_tpu = 2e12 # 2 TOPs / Watt
	print('Total Macs : {}'.format(total_macs))
	#(ops/s)/(ops/W)= W/s = j/s/s =j

	total_enrgy = (ops_per_sec ) / comp_efficiency_tpu
	print('Total energy : {}'.format(total_enrgy)) # andrew not sure if this is energy i tihnk its watts


	enrgy_dict = {'macs':total_macs,'energy':total_enrgy}

	return enrgy_dict

####################################
# model = getModel()
# cnn_enrgy_dict = get_cnn_enrgy(model,input_size=(240,288,12))






