


import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
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
import random

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

####################################
'''
if args.train:
    weights = np.load('models/resnet_yolo.npy', allow_pickle=True).item()
    dropout = True
else:
    weights = np.load('models/resnet_yolo_bn.npy', allow_pickle=True).item()
    dropout = False
'''
####################################
'''
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
dense_block(2048, 5*6*14, weights=weights, relu=False),
])

params = model.get_params()
'''
####################################

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import ReLU
from keras.layers import BatchNormalization
from keras.layers import Add

'''
def model(x):
    x = Conv2D(32, (3, 3), input_shape=(240, 288, 12))(x)
    x = MaxPooling2D(pool_size=(3,3), strides=3)(x)

    x = Conv2D(32, (3, 3), input_shape=(80, 96, 32))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(32, (3, 3), input_shape=(40, 48, 32))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(32, (3, 3), input_shape=(20, 24, 32))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(32, (3, 3), input_shape=(10, 12, 32))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(32, (3, 3), input_shape=(5, 6, 32))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = Dense(units=5*6*32)(x)
    return x
'''

####################################

def conv_block(x, k, f, s, relu=True):
    x = Conv2D(f, (k, k), padding='same', strides=s) (x)
    x = BatchNormalization() (x)
    if relu: x = ReLU() (x)
    return x

####################################

def res_block1(x, f):
    y1 = conv_block(x,  3, f, 1)
    y2 = conv_block(y1, 3, f, 1, relu=False)
    x = Add()([x, y2])
    x = ReLU() (x)
    return x

def res_block2(x, f):
    y1 = conv_block(x,  3, f, 2)
    y2 = conv_block(y1, 3, f, 1, relu=False)
    y3 = conv_block(x,  1, f, 2, relu=False)
    x = Add()([y2, y3])
    x = ReLU() (x)
    return x

####################################

# https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7

####################################

inputs = tf.keras.layers.Input([12, 240, 288, 1])

# 240, 288
x = ConvLSTM2D(16, (7, 7), padding='same', strides=3, return_sequences=True, input_shape=(12, 240, 288, 1)) (inputs)

# 80, 96
x = ConvLSTM2D(32, (3, 3), padding='same', strides=1, return_sequences=False) (x)

# 80, 96
x = res_block2(x, 64)
x = res_block1(x, 64)

# 40, 48
x = res_block2(x, 128)
x = res_block1(x, 128)

# 20, 24
x = res_block2(x, 256)
x = res_block1(x, 256)

# 10, 12
x = res_block2(x, 512)
x = res_block1(x, 512)

# 5, 6
x = Flatten() (x)
x = Dense(units=2048, activation='relu') (x)
x = Dense(units=5*6*14) (x)

####################################

# 240, 288
# model.add( Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(240, 288, 12)) )
# model.add( MaxPooling2D(pool_size=(3, 3), padding='same', strides=3) )

# 80, 96
# x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)
# x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) (x)

# 40, 48
# x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
# x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) (x)

# 20, 24
# x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
# x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) (x)

# 10, 12
# x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
# x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) (x)

# 5, 6
# x = Flatten() (x)
# x = Dense(units=512, activation='relu') (x)
# x = Dense(units=5*6*14) (x)

####################################

model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.Adam(lr=args.lr))
model.summary()

####################################

@tf.function(experimental_relax_shapes=False)
def extract_fn(record):
    _feature={
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (8, 5, 6, 8))

    image = tf.io.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.cast(image, dtype=tf.float32) # this was tricky ... stored as uint8, not float32.
    image = tf.reshape(image, (240, 288, 12))
    
    image = tf.transpose(image, (2, 0, 1))
    image = tf.reshape(image, (12, 240, 288, 1))

    return [image, label]

####################################

def collect_filenames(path):
    samples = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == 'placeholder':
                continue
            samples.append(os.path.join(subdir, file))

    random.shuffle(samples)

    return samples

####################################

if   args.train: filenames = collect_filenames('./dataset/train')
else:            filenames = collect_filenames('./dataset/val')

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
dataset = dataset.map(extract_fn, num_parallel_calls=4)
dataset = dataset.batch(args.batch_size, drop_remainder=True)
# dataset = dataset.repeat()
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

####################################

# super elegant.
# https://keras.io/examples/keras_recipes/tfrecord/

model.fit(dataset, epochs=args.epochs)

####################################

'''
if args.train:
    for epoch in range(args.epochs):

        total = 0
        start = time.time()

        for (x, y) in dataset:
            model.train_on_batch
            
'''

####################################



