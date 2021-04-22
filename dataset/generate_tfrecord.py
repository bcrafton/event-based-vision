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
from scipy import stats
import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

##############################################
'''
def event2frame(events):
    x = events['x']
    y = events['y']
    p = events['p']

    #############

    # (1)
    # tot = np.zeros((240, 304))
    # tot[y, x] += 1
    
    # (2)
    tot = np.zeros((240, 304))
    for (j, i, k) in zip(y, x, p):
        tot[j, i] += 1
    tot = np.reshape(tot, -1)

    # (3)
    # address, count = np.unique(y * 288 + x)

    #############

    frame = np.reshape(tot, -1)
    frame = stats.rankdata(frame, "average")
    frame = np.reshape(frame, (240, 304))

    #############

    # plt.hist(frame.flatten(), bins=100)
    # plt.savefig('hist.png')
    # assert (False)

    #############

    frame = frame - np.min(frame)
    frame = frame / (np.max(frame) + 1e-6)
    frame = 1. - frame

    #############

    frame = cv2.resize(frame, (288, 240)) # cv2 takes things as {W,H} even when array is sized {H,W}

    #############

    return frame
'''
##############################################
def event2frame(events):
    x = events['x']
    y = events['y']
    p = events['p'] * 2 - 1
    
    frame = np.zeros(shape=(240, 304))
    # frame[y, x] += np.ones_like(p)
    for (j, i, k) in zip(y, x, p):
        frame[j, i] += k

    frame = frame - np.min(frame) + 1
    frame = np.log10(frame)

    frame = frame / np.max(frame)
    frame = 1. - frame

    frame = cv2.resize(frame, (288, 240))
    return frame
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
def pick_box(w, h):
    '''
    kmeans = np.array([[ 14.940544,  41.716927],
                       [ 63.379673,  45.207176],
                       [ 84.09129,   63.219963],
                       [142.28989,  126.92114 ],
                       [ 29.110464,  22.325598],
                       [ 44.233715,  32.795734],
                       [106.58469,   83.96504 ],
                       [ 30.385511,  83.227776]])
    '''
    #'''
    kmeans = np.array([[ 47.938934,  35.145702],
                       [ 96.09451,   74.90686 ],
                       [ 29.959908,  22.899212],
                       [ 71.913376,  51.908134],
                       [ 15.042629,  41.93413 ],
                       [ 30.742947,  84.163376],
                       [133.14471,  112.522   ]])
    #'''
    '''
    kmeans = np.array([[ 47.44867,   34.58769 ],
                       [ 96.03882,   74.88576 ],
                       [ 27.881718,  24.065853],
                       [130.79675,  113.888275],
                       [ 71.68791,   51.829678],
                       [ 23.047924,  64.48398 ]])
    '''

    wh = np.array([w, h])
    # wh     = wh     / np.sqrt(np.prod(wh))
    # kmeans = kmeans / np.sqrt(np.prod(kmeans, axis=1, keepdims=True))
    # assert (np.all( np.absolute(1. - np.prod(kmeans, axis=1)) < 1e-9 ))

    i = np.prod(np.minimum(kmeans, wh), axis=1)
    u = np.prod(wh) + np.prod(kmeans, axis=1) - i
    iou = i / np.maximum(np.maximum(1e-10, i), u)
    
    idx = np.argmax(iou)
    return idx, iou[idx], kmeans[idx]

counts = np.zeros(7)
ious = np.zeros(7)
def detection(boxes):
    global counts, ious
    nbox, box_size = np.shape(boxes)
    nbox = min(8, nbox)
    det_np = np.zeros(shape=(8, 7, 10, 12, 8))
    for box_idx in range(nbox):

        _, x, y, w, h, c, _, _ = boxes[box_idx]
        idx, iou, box = pick_box(w, h)
        counts[idx] += 1
        ious[idx] += iou

        x = np.clip(x + 0.5 * w, 0, 288)
        y = np.clip(y + 0.5 * h, 0, 240)

        xc = int(np.clip(x // 24, 0, 11))
        yc = int(np.clip(y // 24, 0, 9))

        x = (x - xc * 24.) / 24. # might want to clip this to zero
        y = (y - yc * 24.) / 24. # might want to clip this to zero
        w = w / box[0]
        h = h / box[1]

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # w = np.clip(w, 0, 1)
        # h = np.clip(h, 0, 1)

        det_np[box_idx, idx, yc, xc, 0:4] = np.array([y, x, h, w])
        det_np[box_idx, idx, yc, xc, 4] = 1.
        det_np[box_idx,   :,  :,  :, 5] = 1.
        det_np[box_idx, idx, yc, xc, 5] = 0.
        det_np[box_idx, idx, yc, xc, 6] = c
        det_np[box_idx,   :,  :,  :, 7] = 1.

    return det_np

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    frame = np.zeros((240, 304, 3), dtype=np.uint8)
    id = 0
    global counts, ious

    for video_idx in range(len(td_files)):
        print (video_idx, counts, np.around(100 * ious / counts))
    
        video = PSEELoader(td_files[video_idx])
        box_video = PSEELoader(td_files[video_idx].replace('_td.dat', '_bbox.npy'))
        
        events_list = []
        while not video.done:

            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            events_list.append(events)
            
            if len(boxes) and len(events_list) >= 12:
                events_list = events_list[-12:]

                frames = [event2frame(es) for es in events_list]
                frames = np.stack(frames, axis=-1)

                boxes_np = []
                for box in boxes:
                    # t, x, y, w, h
                    box[1] = round(box[1] * (288 / 304))
                    box[3] = round(box[3] * (288 / 304))
                    box_np = np.array(list(box))
                    boxes_np.append(box_np)
                boxes_np = np.array(boxes_np)

                ###################################

                det_np = detection(boxes_np)

                ###################################

                centered = np.mean(frames, axis=-1)
                centered = centered - np.min(centered)
                std = np.std(centered)
                if std > 0.05: # look through dumped images and pick #
                    # 1
                    # frames = frames[:, :, -1]
                    # assert (np.shape(frames) == (240, 288))
                    # frames = np.mean(frames, axis=-1)
                    # 4
                    # frames = frames[:, :, -4:]
                    # assert (np.shape(frames) == (240, 288, 4))
                    # 8
                    # frames = frames[:, :, -8:]
                    # assert (np.shape(frames) == (240, 288, 8))
                    # 12
                    # 16
                    write_tfrecord(path, id, frames, det_np)
                events_list = []
                id += 1

###########################################################

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
'''
train_path = './src_data/'

records = []
records = records + collect_filenames(train_path)

for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)
'''
###########################################################
# '''
train_path  = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/train/'
val_path    = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/val/'

records = []
records = records + collect_filenames(train_path)
records = records + collect_filenames(val_path)

for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)
# '''
###########################################################
# '''
test_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/test/'

records = []
records = records + collect_filenames(test_path)

for record in records:
    print (record)

play_files_parallel('./val', records, skip=0, delta_t=20000)
# '''
###########################################################
    
    
    
    
    
    
    
