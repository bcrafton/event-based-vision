


import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=int, default=1)
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

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[args.gpu]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

from yolo_loss import yolo_loss
from draw_boxes import draw_box
from calc_map import calc_map

####################################

load_weights = np.load('models/resnet_yolo.npy', allow_pickle=True).item()

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
from keras.layers import AveragePooling2D
from keras.layers import LSTM
from keras.layers import TimeDistributed

####################################

layer_id = 0
weights = {}

def conv_block(x, k, f, s, relu=True):
    conv = Conv2D(f, (k, k), padding='same', strides=s, use_bias=False)
    bn = BatchNormalization(epsilon=1e-5)

    x = TimeDistributed(conv) (x)
    x = TimeDistributed(bn) (x)

    global layer_id, weights
    weights[layer_id] = (conv, bn)
    layer_id += 1

    if relu: x = TimeDistributed(ReLU()) (x)
    return x

def res_block1(x, f):
    y1 = conv_block(x,  3, f, 1)
    y2 = conv_block(y1, 3, f, 1, relu=False)
    x = Add()([x, y2])
    x = ReLU() (x)
    return x

def res_block2(x, f):
    y1 = conv_block(x,  3, f, 1)
    y2 = conv_block(y1, 3, f, 1, relu=False)
    y3 = conv_block(x,  1, f, 1, relu=False)
    x = Add()([y2, y3])
    x = ReLU() (x)
    return x

def dense_block(x, n, last=False):
    dense = Dense(units=n)

    global layer_id, weights
    weights[layer_id] = (dense,)
    layer_id += 1

    x = TimeDistributed(dense) (x)
    if not last:
        x = TimeDistributed(ReLU()) (x)
        x = TimeDistributed(Dropout(0.5)) (x)
    return x

inputs = tf.keras.layers.Input([12, 240, 288, 1])

x = conv_block(inputs, 7, 64, 1)
x = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same', strides=3)) (x)

# 80, 96
x = res_block1(x, 64)
x = res_block1(x, 64)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2)) (x)

# 40, 48
x = res_block2(x, 128)
x = res_block1(x, 128)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2)) (x)

# 20, 24
x = res_block2(x, 256)
x = res_block1(x, 256)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2)) (x)

# 10, 12
x = res_block2(x, 512)
x = res_block1(x, 512)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', strides=2)) (x)

# 5, 6
x = res_block2(x, 512)
x = res_block1(x, 512)

x = TimeDistributed(Flatten()) (x)
x = dense_block (x, 2048)

####################################

x = LSTM(1024, return_sequences=False) (x)
x = Dense(5*6*14) (x)

model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1))
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
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

####################################

# super elegant.
# https://keras.io/examples/keras_recipes/tfrecord/

model.fit(dataset, epochs=args.epochs)

####################################

save_weights = {}

for layer in weights.keys():
    if len(weights[layer]) > 1:
        conv, bn = weights[layer]
        (f,) = conv.get_weights()
        (g, b, _, _) = bn.get_weights()
        save_weights[layer] = {'f': f, 'g': g, 'b': b}
    else:
        (dense,) = weights[layer]
        (w, b) = dense.get_weights()
        save_weights[layer] = {'w': w, 'b': b}

np.save('trained_weights', save_weights)

####################################








