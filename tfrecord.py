
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

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[args.gpu]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)
    
####################################

def extract_fn(record):
    _feature={
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (8, 5, 6, 8))
    
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

for (image, label) in val_dataset:
    # print (image)
    image_np = image.numpy()
    image_np = image_np[0, 0, :, :, 0]
    # plt.imshow(image_np)
    # plt.show()
    plt.imsave('image.jpg', image_np)
    print (np.shape(label))
    assert (False)









