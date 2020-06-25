
import numpy as np
import tensorflow as tf
from layers import *
from load import Loader
import time

####################################

'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[0]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

####################################

# load weights, hope the weight id matches up.
weights = np.load('resnet18.npy', allow_pickle=True).item()

# 240, 288
model = model(layers=[
conv_block((5,5,12,64), 3), # 80, 96

res_block1(64,   64, 1, weights=weights), # 80, 96
res_block1(64,   64, 1, weights=weights), # 80, 96

res_block2(64,   128, 2, weights=weights), # 40, 48
res_block1(128,  128, 1, weights=weights), # 40, 48

res_block2(128,  256, 2, weights=weights), # 20, 24
res_block1(256,  256, 1, weights=weights), # 20, 24

res_block2(256,  512, 2, weights=weights), # 10, 12
res_block1(512,  512, 1, weights=weights), # 10, 12

res_block2(512,  512, 2, weights=None), # 5, 6
res_block1(512,  512, 1, weights=None), # 5, 6

dense_block(5*6*512, 1024, weights=None),
dense_block(1024, 5*6*12, weights=None),
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1.)

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    
    grad = tape.gradient(loss, params)
    return loss, correct, grad

####################################

def predict(model, x, y):
    pred_logits = model.train(x)
    # pred_label = tf.argmax(pred_logits, axis=1)
    # correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return pred_logits

####################################

def collect(model, x, y):
    pred_logits, stats = model.collect(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct, stats

####################################

def run_train():

    total = 100
    total_correct = 0
    total_loss = 0
    batch_size = 1

    # load = Loader('', total // batch_size, batch_size, 8)
    load = np.load('dataset.npy', allow_pickle=True).item()
    xs, ys = load['x'], load['y']
    
    start = time.time()

    for batch in range(0, total, batch_size):
        # while load.empty(): pass # print ('empty')
        
        # x, y = load.pop()
        s = batch; e = s + batch_size
        x = xs[s:e].astype(np.float32); y = ys[s:e]
        
        pred = predict(model, x, y)
        
        '''
        loss, correct, grad = gradients(model, x, y)
        optimizer.apply_gradients(zip(grad, params))
        total_loss += loss.numpy()
        
        total_correct += correct.numpy()
        
        acc = round(total_correct / (batch + batch_size), 3)
        avg_loss = total_loss / (batch + batch_size)
        
        if (batch + batch_size) % (batch_size * 100) == 0:
            img_per_sec = (batch + batch_size) / (time.time() - start)
            print (batch + batch_size, img_per_sec, acc, avg_loss)
        '''

    trained_weights = model.get_weights()
    np.save('trained_weights', trained_weights)

####################################

run_train()

####################################














