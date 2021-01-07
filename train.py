


import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
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

if args.train:
    weights = np.load('models/resnet_yolo_anchor.npy', allow_pickle=True).item()
    dropout = True
else:
    weights = np.load('models/resnet_yolo_anchor_bn.npy', allow_pickle=True).item()
    dropout = False

####################################

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
dense_block(2048, 7*10*12*7, weights=weights, relu=False),
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1.)

batch_size_tf = tf.constant(args.batch_size)

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, y):
    with tf.GradientTape() as tape:
        out = model.train(x)
        out = tf.reshape(out, (args.batch_size, 7, 10, 12, 7))
        loss, losses = yolo_loss(batch_size_tf, out, y)

    grad = tape.gradient(loss, params)
    return out, loss, losses, grad

####################################

# can't have this decorator since we need to call .numpy() for var and mean.
# @tf.function(experimental_relax_shapes=False)
def collect(model, x):
    out = model.collect(x)
    out = tf.reshape(out, (args.batch_size, 7, 10, 12, 7))
    return out

####################################

@tf.function(experimental_relax_shapes=False)
def predict(model, x):
    out = model.predict(x)
    out = tf.reshape(out, (args.batch_size, 7, 10, 12, 7))
    return out

####################################

def write(filename, text):
    print (text)
    f = open(filename, "a")
    f.write(text + "\n")
    f.close()

####################################

@tf.function(experimental_relax_shapes=False)
def extract_fn(record):
    _feature={
        'id_raw':    tf.io.FixedLenFeature([], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    id = sample['id_raw']

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (8, 7, 10, 12, 8))

    image = tf.io.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.cast(image, dtype=tf.float32) # this was tricky ... stored as uint8, not float32.
    image = tf.reshape(image, (240, 288, 12))

    return [id, image, label]

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

if args.train: filenames = collect_filenames('./dataset/train')
else:          filenames = collect_filenames('./dataset/val')

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
dataset = dataset.map(extract_fn, num_parallel_calls=4)
dataset = dataset.batch(args.batch_size, drop_remainder=True)
# dataset = dataset.repeat()
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

####################################

if args.train:
    for epoch in range(args.epochs):

        total_yx_loss = 0
        total_hw_loss = 0
        total_obj_loss = 0
        total_no_obj_loss = 0
        total_cat_loss = 0

        total_loss = 0 # 1e-6
        total = 0
        start = time.time()

        for (id, x, y) in dataset:

            out, loss, losses, grad = gradients(model, x, y)
            optimizer.apply_gradients(zip(grad, params))

            (yx_loss, hw_loss, obj_loss, no_obj_loss, cat_loss) = losses
            total_yx_loss     += yx_loss.numpy()
            total_hw_loss     += hw_loss.numpy()
            total_obj_loss    += obj_loss.numpy()
            total_no_obj_loss += no_obj_loss.numpy()
            total_cat_loss    += cat_loss.numpy()
            total_loss        += loss.numpy()
            total += 1

            if total % 1000 == 0:
                yx_loss     = int(total_yx_loss     / total_loss * 100)
                hw_loss     = int(total_hw_loss     / total_loss * 100)
                obj_loss    = int(total_obj_loss    / total_loss * 100)
                no_obj_loss = int(total_no_obj_loss / total_loss * 100)
                cat_loss    = int(total_cat_loss    / total_loss * 100)
                avg_loss = total_loss / total
                avg_rate = (total * args.batch_size) / (time.time() - start)
                write(name + '.results', 'total: %d, rate: %f, loss %f (%d %d %d %d %d)' % (total * args.batch_size, avg_rate, avg_loss, yx_loss, hw_loss, obj_loss, no_obj_loss, cat_loss))

        yx_loss     = int(total_yx_loss     / total_loss * 100)
        hw_loss     = int(total_hw_loss     / total_loss * 100)
        obj_loss    = int(total_obj_loss    / total_loss * 100)
        no_obj_loss = int(total_no_obj_loss / total_loss * 100)
        cat_loss    = int(total_cat_loss    / total_loss * 100)
        avg_loss = total_loss / total
        avg_rate = (total * args.batch_size) / (time.time() - start)
        write(name + '.results', 'total: %d, rate: %f, loss %f (%d %d %d %d %d)' % (total * args.batch_size, avg_rate, avg_loss, yx_loss, hw_loss, obj_loss, no_obj_loss, cat_loss))

        trained_weights = model.get_weights()
        np.save(name + '_weights', trained_weights)

####################################

if args.train and (args.epochs == 0):
    total = 0
    start = time.time()
    for (id, x, y) in dataset:
        out = collect(model, x)
        total += 1
        if total % 1000 == 0:
            avg_rate = (total * args.batch_size) / (time.time() - start)
            print('total: %d, rate: %f' % (total * args.batch_size, avg_rate))

    trained_weights = model.get_weights()
    np.save(name + '_weights', trained_weights)

####################################

if not args.train:

    total = 0
    start = time.time()
    ids = []
    ys = []
    preds = []

    for (id, x, y) in dataset:

        out = predict(model, x)

        total += 1
        if total % 1000 == 0:
            avg_rate = (total * args.batch_size) / (time.time() - start)
            print('total: %d, rate: %f' % (total * args.batch_size, avg_rate))

        true, pred = y.numpy(), out.numpy()
        ids.append(np.array(id))
        ys.append(np.copy(true))
        preds.append(np.copy(pred))
        # draw_box('./results/%d.jpg' % (total), np.sum(x.numpy()[0, :, :, :], axis=2), true[0], pred[0])

    print (total * args.batch_size)
    ids = np.concatenate(ids, axis=0).astype(np.float32)
    ys = np.concatenate(ys, axis=0).astype(np.float32)
    preds = np.concatenate(preds, axis=0).astype(np.float32)

    # results = {}
    # results['id'] = ids
    # results['true'] = ys
    # results['pred'] = preds
    # np.save('results', results)

    calc_map(ids, ys, preds)

####################################



