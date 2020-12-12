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
def pick_box(w, h):
    kmeans = np.array([[ 14.940544,  41.716927],
                       [ 63.379673,  45.207176],
                       [ 84.09129,   63.219963],
                       [142.28989,  126.92114 ],
                       [ 29.110464,  22.325598],
                       [ 44.233715,  32.795734],
                       [106.58469,   83.96504 ],
                       [ 30.385511,  83.227776]])

    wh = np.array([w, h])
    wh     = wh     / np.sqrt(np.prod(wh))
    kmeans = kmeans / np.sqrt(np.prod(kmeans, axis=1, keepdims=True))
    assert (np.all( np.absolute(1. - np.prod(kmeans, axis=1)) < 1e-9 ))

    i = np.prod(np.minimum(kmeans, wh), axis=1)
    u = np.prod(wh) + np.prod(kmeans, axis=1) - i
    iou = i / np.maximum(np.maximum(1e-10, i), u)
    
    idx = np.argmax(iou)
    return idx, kmeans[idx]

def detection(boxes):
    nbox, box_size = np.shape(boxes)
    nbox = min(8, nbox)
    det_np = np.zeros(shape=(8, 8, 5, 6, 8))
    for box_idx in range(nbox):

        _, x, y, w, h, c, _, _ = boxes[box_idx]
        idx, box = pick_box(w, h)

        x = np.clip(x + 0.5 * w, 0, 288)
        y = np.clip(y + 0.5 * h, 0, 240)

        xc = int(np.clip(x // 48, 0, 5))
        yc = int(np.clip(y // 48, 0, 4))

        x = (x - xc * 48.) / 48. # might want to clip this to zero
        y = (y - yc * 48.) / 48. # might want to clip this to zero
        w = np.sqrt(np.log(w / box[0]))
        h = np.sqrt(np.log(h / box[1]))

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # w = np.clip(w, 0, 1)
        # h = np.clip(h, 0, 1)

        det_np[box_idx, idx, yc, xc, 0:4] = np.array([y, x, h, w])
        det_np[box_idx, idx, yc, xc, 4] = 1.
        det_np[box_idx, idx,  :,  :, 5] = 1.
        det_np[box_idx, idx, yc, xc, 5] = 0.
        det_np[box_idx, idx, yc, xc, 6] = c
        det_np[box_idx, idx,  :,  :, 7] = 1.

    return det_np

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    frame = np.zeros((240, 304, 3), dtype=np.uint8)
    id = 0

    for video_idx in range(len(td_files)):
        print (video_idx)
    
        video = PSEELoader(td_files[video_idx])
        box_video = PSEELoader(td_files[video_idx].replace('_td.dat', '_bbox.npy'))
        
        frames = []
        # frame_idx = 0
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
                frames_cp = np.copy(frames)

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

                if np.shape(frames_cp) == (240, 288, 12):
                    img = np.mean(frames_cp, axis=-1)
                    img = img - np.min(img)
                    std = np.std(img)
                    if std > 5:
                        # filename = '%s/%d_%d.tfrecord' % (path, video_idx, frame_idx)
                        write_tfrecord(path, id, frames_cp, det_np)

                frames = []
                # frame_idx += 1
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

train_path  = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/train/'
val_path    = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/val/'
test_a_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/test/test_a/'
test_b_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/test/test_b/'

###########################################################
'''
train_path = './src_data/'
val_path = ''
'''
###########################################################
# '''
records = []
records = records + collect_filenames(train_path)
records = records + collect_filenames(val_path)
records = records + collect_filenames(test_b_path)

for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)
# '''
###########################################################
'''
records = []
records = records + collect_filenames(test_a_path)
# records = records + collect_filenames(test_b_path)

for record in records:
    print (record)

play_files_parallel('./val', records, skip=0, delta_t=20000)
'''
###########################################################
    
    
    
    
    
    
    
