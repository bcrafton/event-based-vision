


import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
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

model = Sequential()

# 240, 288
model.add( Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(240, 288, 12)) )
model.add( MaxPooling2D(pool_size=(3, 3), padding='same', strides=3) )

# 80, 96
model.add( Conv2D(32, (3, 3), activation='relu', padding='same') )
model.add( MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) )

# 40, 48
model.add( Conv2D(32, (3, 3), activation='relu', padding='same') )
model.add( MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) )

# 20, 24
model.add( Conv2D(32, (3, 3), activation='relu', padding='same') )
model.add( MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) )

# 10, 12
model.add( Conv2D(32, (3, 3), activation='relu', padding='same') )
model.add( MaxPooling2D(pool_size=(2, 2), padding='same', strides=2) )

# 5, 6
model.add( Flatten() )
model.add( Dense(units=512, activation='relu') )
model.add( Dense(units=5*6*14) )

####################################

model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.Adam(lr=args.lr))

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

model.fit(dataset, epochs=2)

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



