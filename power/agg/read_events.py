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

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    frame = np.zeros((240, 304, 3), dtype=np.uint8)
    
    for video_idx in range(len(td_files)):
        print (video_idx)
    
        video = PSEELoader(td_files[video_idx])
        box_video = PSEELoader(td_files[video_idx].replace('_td.dat', '_bbox.npy'))

        while not video.done:

            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            frame = vis.make_binary_histo(events, img=frame, width=304, height=240)
            plt.imshow(frame)
            plt.show()

###########################################################

train_path = '../../dataset/train_src/'

###########################################################

records = []
for subdir, dirs, files in os.walk(train_path):
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.dat':
            records.append(os.path.join(subdir, file))

records = sorted(records)
for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)

###########################################################
    
    
    
    
    
    
    
