
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=int, default=1)
# parser.add_argument('--name', type=str, default="imagenet_weights")
args = parser.parse_args()

name = '%d_%f' % (args.gpu, args.lr)

####################################

import numpy as np
import tensorflow as tf
from layers import *
import time
import matplotlib.pyplot as plt

####################################

'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''

# '''
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[args.gpu]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)
# '''

from yolo_loss import yolo_loss
from draw_boxes import draw_box
from calc_map import calc_map
from load import Loader

####################################

if args.train:
    # weights = None # weights = np.load('resnet18.npy', allow_pickle=True).item()
    weights = np.load('models/small_resnet_yolo_abcdef.npy', allow_pickle=True).item()
else:
    weights = np.load('models/small_resnet_yolo_abcdef.npy', allow_pickle=True).item()
    
####################################

# 240, 288
model = model(layers=[
conv_block((7,7,12,64), 1, weights=weights), # 240, 288

max_pool(s=3, p=3),

res_block1(64,   64, 1, weights=weights), # 80, 96
# res_block1(64,   64, 1, weights=weights), # 80, 96

max_pool(s=2, p=2),

res_block2(64,   128, 1, weights=weights), # 40, 48
# res_block1(128,  128, 1, weights=weights), # 40, 48

max_pool(s=2, p=2),

res_block2(128,  256, 1, weights=weights), # 20, 24
# res_block1(256,  256, 1, weights=weights), # 20, 24

max_pool(s=2, p=2),

res_block2(256,  512, 1, weights=weights), # 10, 12
# res_block1(512,  512, 1, weights=weights), # 10, 12

max_pool(s=2, p=2),

# res_block2(512,  512, 1, weights=weights), # 5, 6
# res_block1(512,  512, 1, weights=weights), # 5, 6

# dense_block(5*6*512, 1024, weights=weights),
# dense_block(1024, 5*6*12, weights=weights, relu=False),

dense_block(5*6*512, 5*6*12, weights=weights, relu=False),
])

params = model.get_params()

####################################

if args.train: optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1.)

batch_size_tf = tf.constant(args.batch_size)

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, coord, obj, no_obj, cat, vld):
    with tf.GradientTape() as tape:
        out = model.train(x)
        out = tf.reshape(out, (args.batch_size, 5, 6, 12))
        loss, losses = yolo_loss(batch_size_tf, out, coord, obj, no_obj, cat, vld)
    
    grad = tape.gradient(loss, params)
    return out, loss, losses, grad

####################################

@tf.function(experimental_relax_shapes=False)
def predict(model, x):
    out = model.train(x)
    out = tf.reshape(out, (args.batch_size, 5, 6, 12))
    return out

####################################

def write(filename, text):
    print (text)
    f = open(filename, "a")
    f.write(text + "\n")
    f.close()
    
####################################

def extract_fn(record):
    _feature={
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (-1, 8))
    
    image = tf.io.decode_raw(sample['image_raw'], tf.float32)
    image = tf.cast(image, dtype=tf.float32) # this was tricky ... stored as uint8, not float32.
    image = tf.reshape(image, (1, 240, 288, 12))
    
    return [image, label]

####################################

def collect_filenames(path):
    samples = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == 'placeholder':
                continue
            samples.append(os.path.join(subdir, file))

    return samples
    
####################################

'''
if args.train: N = 1459 # 1458.npy
else:          N = 250
'''

if args.train: epochs = args.epochs
else:          epochs = 1

if args.train: nbatch = 10 # @ batch_size = 16
else:          nbatch = 10 # @ batch_size = 16

nthread = 4

filenames = collect_filenames('/home/brian/Desktop/event-based-vision/dataset/train')

val_dataset = tf.data.TFRecordDataset(filenames)
val_dataset = val_dataset.map(extract_fn)
val_dataset = val_dataset.batch(8)
val_dataset = val_dataset.repeat()

'''
for (image, label) in val_dataset.take(10):
    print (image)
    print (label)
'''

'''
for (image, label) in val_dataset:
    # print (image)
    image_np = image.numpy()
    image_np = image_np[0, 0, :, :, 0]
    # plt.imshow(image_np)
    # plt.show()
    plt.imsave('image.jpg', image_np)
    print (np.max(label))
    assert (False)
'''








