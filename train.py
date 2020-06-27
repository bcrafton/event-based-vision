
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
# parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--name', type=str, default="imagenet_weights")
args = parser.parse_args()

name = '%d_%f.results' % (args.gpu, args.lr)

####################################

import numpy as np
import tensorflow as tf
from layers import *
import time
import matplotlib.pyplot as plt

####################################

# '''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# '''

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[args.gpu]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)
'''

from yolo_loss import yolo_loss
from draw_boxes import draw_box

####################################

# load weights, hope the weight id matches up.
weights = np.load('resnet18.npy', allow_pickle=True).item()

# 240, 288
model = model(layers=[
conv_block((5,5,12,64), 3), # 80, 96

res_block1(64,   64, 1, weights=weights), # 80, 96
res_block1(64,   64, 1, weights=weights), # 80, 96

res_block2(64,   128, 2, weights=weights), # 40, 48
res_block1(128,  128, 1, weights=weights), # 40, 48

res_block2(128,  256, 2, weights=weights), # 20, 24
res_block1(256,  256, 1, weights=weights), # 20, 24

res_block2(256,  512, 2, weights=weights), # 10, 12
res_block1(512,  512, 1, weights=weights), # 10, 12

res_block2(512,  512, 2, weights=None), # 5, 6
res_block1(512,  512, 1, weights=None), # 5, 6

dense_block(5*6*512, 1024, weights=None),
dense_block(1024, 5*6*12, weights=None),
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1.)

def gradients(model, x, coord, obj, no_obj, cat, vld):
    with tf.GradientTape() as tape:
        out = model.train(x)
        out = tf.reshape(out, (8, 5, 6, 12))
        loss = yolo_loss(out, coord, obj, no_obj, cat, vld)
    
    grad = tape.gradient(loss, params)
    return out, loss, grad

####################################

def write(filename, text):
    print (text)
    f = open(filename, "a")
    f.write(text + "\n")
    f.close()

####################################

from load import Loader

nthread = 4
batch_size = 8
nbatch = 100

load = Loader('/home/brian/Desktop/event-based-vision/dataset/data', nbatch, batch_size, nthread)

####################################

total_loss = 0
total = 0

start = time.time()

for i in range(nbatch):
    while load.empty(): pass 
    x, coord, obj, no_obj, cat, vld = load.pop()
    print (i, nbatch)
    
    out, loss, grad = gradients(model, x, coord, obj, no_obj, cat, vld)
    optimizer.apply_gradients(zip(grad, params))

    total_loss += loss.numpy()
    total += batch_size
    
avg_loss = total_loss / total
avg_rate = total / (time.time() - start)
load.join()

print (avg_rate, avg_loss)

####################################














