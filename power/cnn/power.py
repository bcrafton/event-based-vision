
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
import tensorflow as tf
from layers import *
import time
import matplotlib.pyplot as plt
import random

####################################

# 240, 288
model = model(layers=[
conv_block((7,7,12,64), 1, weights=weights), # 240, 288

max_pool(s=3, p=3),

res_block1(64,   64, 1, weights=weights), # 80, 96
res_block1(64,   64, 1, weights=weights), # 80, 96

max_pool(s=2, p=2),

res_block2(64,   128, 1, weights=weights), # 40, 48
res_block1(128,  128, 1, weights=weights), # 40, 48

max_pool(s=2, p=2),

res_block2(128,  256, 1, weights=weights), # 20, 24
res_block1(256,  256, 1, weights=weights), # 20, 24

max_pool(s=2, p=2),

res_block2(256,  512, 1, weights=weights), # 10, 12
res_block1(512,  512, 1, weights=weights), # 10, 12

max_pool(s=2, p=2),

res_block2(512,  512, 1, weights=weights), # 5, 6
res_block1(512,  512, 1, weights=weights), # 5, 6

dense_block(5*6*512, 2048, weights=weights, dropout=dropout),
dense_block(2048, 5*6*12, weights=weights, relu=False),
])

####################################














