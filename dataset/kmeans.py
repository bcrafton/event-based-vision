"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf
import cv2
import argparse

import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

##############################################

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(path, id, frames, label):
    with tf.io.TFRecordWriter('%s/%d.tfrecord' % (path, id)) as writer:
        image_raw = frames.astype(np.uint8).tostring()
        label_raw = label.astype(np.float32).tostring()
        _feature={
                'id_raw':    _int64_feature(id),
                'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)
                }
        _features=tf.train.Features(feature=_feature)
        example = tf.train.Example(features=_features)
        writer.write(example.SerializeToString())

##############################################

def collect_filenames(path):
    filenames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.dat':
                filenames.append(os.path.join(subdir, file))

    filenames = sorted(filenames)
    return filenames

###########################################################

train_path = './src_data/'

records = []
records = records + collect_filenames(train_path)

for record in records:
    print (record)

###########################################################

boxes_np = []
for video_idx in range(len(records)):

    video = PSEELoader(records[video_idx])
    box_video = PSEELoader(records[video_idx].replace('_td.dat', '_box.npy'))

    while not video.done:
        _ = video.load_delta_t(20000)
        boxes = box_video.load_delta_t(20000)
        if len(boxes):
            for box in boxes:
                # t, x, y, w, h
                box[1] = round(box[1] * (448 / 1280))
                box[3] = round(box[3] * (448 / 1280))
                box[2] = round(box[2] * (256 / 720))
                box[4] = round(box[4] * (256 / 720))
                box_np = np.array(list(box))
                boxes_np.append(box_np)
    
###########################################################

boxes_np = np.array(boxes_np)
boxes_np = boxes_np[:, 2:4]

###########################################################

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=9).fit(boxes_np)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=8).fit(boxes_np)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=7).fit(boxes_np)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=6).fit(boxes_np)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=5).fit(boxes_np)
print (kmeans.cluster_centers_)
    
###########################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
