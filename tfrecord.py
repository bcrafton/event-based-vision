
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

#####################################

def collect_filenames(path):
    samples = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == 'placeholder':
                continue
            samples.append(os.path.join(subdir, file))
    return samples

#####################################

def extract(record):
    _feature={
        'id_raw':    tf.io.FixedLenFeature([], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    id = sample['id_raw']

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (8, 5, 6, 8))

    image = tf.io.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.cast(image, dtype=tf.float32) # this was tricky ... stored as uint8, not float32.
    image = tf.reshape(image, (240, 288, 12))

    return [id, image, label]

#####################################

files = collect_filenames('./dataset/val')
tfrecord = tf.data.TFRecordDataset(files)
dataset = tfrecord.map(extract)
for (id, image, label) in dataset:
    image = image.numpy().sum(axis=2)
    plt.imsave('./dataset/images/%d.jpg' % (id), image)
    
#####################################
