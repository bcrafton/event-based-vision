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
import cv2
import argparse

import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    frame = np.zeros((240, 304, 3), dtype=np.uint8)
    
    for idx in range(len(td_files)):
    
        video = PSEELoader(td_files[idx])
        box_video = PSEELoader(td_files[idx].replace('_td.dat', '_bbox.npy'))
        
        frames = []
        frame_idx = 0
        while not video.done:

            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)

            frame = vis.make_binary_histo(events, img=frame, width=304, height=240)

            assert (np.shape(frame) == (240, 304, 3))
            frame_preprocess = np.copy(frame[:, :, 0])

            frame_preprocess = cv2.resize(frame_preprocess, (288, 240)) # cv2 takes things as {W,H} even when array is sized {H,W}
            frames.append(frame_preprocess)
            
            if len(boxes):
                frames = frames[-12:]
                frames = np.stack(frames, axis=-1)
                for box in boxes:
                    box[1] = round(box[1] * (288 / 304))
                    box[3] = round(box[3] * (288 / 304))
                if np.shape(frames) == (240, 288, 12):
                    # xs.append(frames)
                    # dets.append(np.copy(boxes))
                    sample = {'x': frames, 'y': boxes}
                    np.save('%s/%d_%d' % (path, idx, frame_idx), sample)
                frames = []
                frame_idx += 1

        # print (idx, np.shape(np.array(xs)))
        # dataset = {'x': np.array(xs), 'y': dets}
        # np.save('%s/%d' % (path, idx), dataset)

###########################################################
'''
train_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/train/'
val_path   = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/val/'
'''
###########################################################

train_path = './src_data/'
val_path = ''

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
'''
records = []
for subdir, dirs, files in os.walk(val_path):
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.dat':
            records.append(os.path.join(subdir, file))

records = sorted(records)
for record in records:
    print (record)

play_files_parallel('./val', records, skip=0, delta_t=20000)
'''
###########################################################
    
    
    
    
    
    
    
