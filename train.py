
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--local', type=int, default=0)
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
x abscissa of the top left corner in pixels
y ordinate of the top left corner in pixels
w width of the boxes in pixel
h height of the boxes in pixel
ts timestamp of the box in the sequence in microseconds
class_id 0 for cars and 1 for pedestrians
'''

# (49499999, 218., 84., 96., 91., 0, 1., 3758)
# (49499999, 93., 84., 12., 43., 1, 1., 3759)

'''
ts timestamp of the box in the sequence in microseconds
x abscissa of the top left corner in pixels
y ordinate of the top left corner in pixels
w width of the boxes in pixel
h height of the boxes in pixel
class_id 0 for cars and 1 for pedestrians
obj (0,1)
[3758, 3759] ???
'''

def create_labels(dets):
    max_nd = 0
    for b in range(len(dets)):
        nd = len(dets[b])
        max_nd = max(max_nd, nd)

    coords = []; objs = []; no_objs = []; cats = []; vlds = []
    for b in range(len(dets)):
        coord, obj, no_obj, cat, vld = det_tensor(dets[b], max_nd)
        coords.append(coord); objs.append(obj); no_objs.append(no_obj); cats.append(cat); vlds.append(vld)
    
    coords  = np.stack(coords, axis=0).astype(np.float32)
    objs    = np.stack(objs, axis=0).astype(np.float32)
    no_objs = np.stack(no_objs, axis=0).astype(np.float32)
    cats    = np.stack(cats, axis=0).astype(np.float32)
    vlds    = np.stack(vlds, axis=0).astype(np.float32)

    return coords, objs, no_objs, cats, vlds

def det_tensor(dets, max_nd):

    coord   = np.zeros(shape=[max_nd, 5, 6, 5])
    obj     = np.zeros(shape=[max_nd, 5, 6])
    no_obj  = np.ones(shape=[max_nd, 5, 6])
    cat     = np.zeros(shape=[max_nd, 5, 6])
    vld     = np.zeros(shape=[max_nd, 5, 6])
    
    for idx in range(len(dets)):

        _, x, y, w, h, c, _, _ = dets[idx]
        x = np.clip(x + 0.5 * w, 0, 288)
        y = np.clip(y + 0.5 * h, 0, 240)

        xc = int(np.clip(x // 48, 0, 5))
        yc = int(np.clip(y // 48, 0, 4))
        
        x = (x - xc * 48.) / 48. # might want to clip this to zero
        y = (y - yc * 48.) / 48. # might want to clip this to zero
        w = np.sqrt(w / 288.)
        h = np.sqrt(h / 240.)

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        w = np.clip(w, 0, 1)
        h = np.clip(h, 0, 1)
        
        coord [idx, yc, xc, :] = np.array([y, x, h, w, 1.])
        obj   [idx, yc, xc] = 1.
        no_obj[idx, yc, xc] = 0.
        cat   [idx, yc, xc] = c
        vld   [idx, :, :] = 1.
        
    return coord, obj, no_obj, cat, vld

####################################

if args.train:
    weights = None # weights = np.load('resnet18.npy', allow_pickle=True).item()
else:
    weights = np.load('models/small_resnet_yolo.npy', allow_pickle=True).item()
    
####################################

# 240, 288
model = model(layers=[
conv_block((7,7,12,64), 1, weights=weights), # 240, 288

max_pool(s=3, p=3),

res_block1(64,   64, 1, weights=weights), # 80, 96
# res_block1(64,   64, 1, weights=weights), # 80, 96

max_pool(s=2, p=2),

res_block2(64,   128, 1, weights=weights), # 40, 48
# res_block1(128,  128, 1, weights=weights), # 40, 48

max_pool(s=2, p=2),

res_block2(128,  256, 1, weights=weights), # 20, 24
# res_block1(256,  256, 1, weights=weights), # 20, 24

max_pool(s=2, p=2),

res_block2(256,  512, 1, weights=weights), # 10, 12
# res_block1(512,  512, 1, weights=weights), # 10, 12

max_pool(s=2, p=2),

# res_block2(512,  512, 1, weights=weights), # 5, 6
# res_block1(512,  512, 1, weights=weights), # 5, 6

# dense_block(5*6*512, 1024, weights=weights),
# dense_block(1024, 5*6*12, weights=weights, relu=False),

dense_block(5*6*512, 5*6*12, weights=weights, relu=False),
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1.)

def gradients(model, x, coord, obj, no_obj, cat, vld):
    with tf.GradientTape() as tape:
        out = model.train(x)
        out = tf.reshape(out, (args.batch_size, 5, 6, 12))
        '''
        out = tf.concat((out[:, :, :, 0:4], tf.nn.sigmoid(out[:, :, :, 4:5]),
                         out[:, :, :, 5:9], tf.nn.sigmoid(out[:, :, :, 9:10]),
                         out[:, :, :, 10:12]), axis=3)
        '''
        loss, losses = yolo_loss(args.batch_size, out, coord, obj, no_obj, cat, vld)
    
    grad = tape.gradient(loss, params)
    return out, loss, losses, grad

####################################

def write(filename, text):
    print (text)
    f = open(filename, "a")
    f.write(text + "\n")
    f.close()

####################################

N = 500
def run_train():

    for epoch in range(args.epochs):
        total_yx_loss = 0
        total_hw_loss = 0
        total_obj_loss = 0
        total_no_obj_loss = 0
        total_cat_loss = 0

        total_loss = 0        
        total = 0
        start = time.time()

        for n in range(N):
            if args.local:
                filename = './dataset/data/%d.npy' % (n)
            else:
                filename = '/home/bcrafton3/Data_SSD/6254/dataset/%d.npy' % (n)
            
            load = np.load(filename, allow_pickle=True).item()
            xs, ys = load['x'], load['y']

            for batch in range(0, len(xs), args.batch_size):
                s = batch
                e = batch + args.batch_size
                if e > len(xs): continue
                
                x = xs[s:e].astype(np.float32)
                coord, obj, no_obj, cat, vld = create_labels(ys[s:e])
                
                out, loss, losses, grad = gradients(model, x, coord, obj, no_obj, cat, vld)
                if args.train:
                    optimizer.apply_gradients(zip(grad, params))
                
                if not args.train:
                    try:
                        calc_map(ys[s:e], out.numpy())
                    except:
                        pass
                
                (yx_loss, hw_loss, obj_loss, no_obj_loss, cat_loss) = losses
                total_yx_loss     += yx_loss.numpy()
                total_hw_loss     += hw_loss.numpy()
                total_obj_loss    += obj_loss.numpy()
                total_no_obj_loss += no_obj_loss.numpy()
                total_cat_loss    += cat_loss.numpy()

                total_loss += loss.numpy()
                total += args.batch_size
                
                if (epoch % 5) == 0:
                    nd = np.count_nonzero(obj[0])
                    draw_box('./results/%d_%d.jpg' % (n, batch), np.sum(x[0, :, :, :], axis=2), coord[0], out.numpy()[0], nd)


        yx_loss     = int(total_yx_loss     / total_loss * 100)
        hw_loss     = int(total_hw_loss     / total_loss * 100)
        obj_loss    = int(total_obj_loss    / total_loss * 100)
        no_obj_loss = int(total_no_obj_loss / total_loss * 100)
        cat_loss    = int(total_cat_loss    / total_loss * 100)

        # avg_loss = total_loss / total
        avg_loss = total_loss / (total / args.batch_size) # we reduce_mean over (batch,detection) (0,1)
        avg_rate = total / (time.time() - start)
        # print (avg_rate, avg_loss)
        write(name + '.results', 'total: %d, rate: %f, loss %f (%d %d %d %d %d)' % (total, avg_rate, avg_loss, yx_loss, hw_loss, obj_loss, no_obj_loss, cat_loss))

        trained_weights = model.get_weights()
        np.save(name + '_weights', trained_weights)

####################################

run_train()

####################################














