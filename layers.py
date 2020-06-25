
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

# https://stackoverflow.com/questions/59656219/override-tf-floor-gradient

# this would also work:
# https://www.tensorflow.org/api_docs/python/tf/grad_pass_through

@tf.custom_gradient
def floor_no_grad(x):

    def grad(dy):
        return dy
    
    return tf.floor(x), grad
    
#############
    
@tf.custom_gradient
def round_no_grad(x):

    def grad(dy):
        return dy
    
    return tf.round(x), grad
    
#############

def quantize_and_dequantize(x, low, high):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = round_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    x = x * scale
    return x

def quantize(x, low, high):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = round_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    return x, scale
    
def quantize_predict(x, scale, low, high):
    x = x / scale
    x = tf.floor(x)
    x = tf.clip_by_value(x, low, high)
    return x

#############

def quantize_and_dequantize_np(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    x = x * scale
    return x, scale

def quantize_np(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    return x, scale
    
#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x, training=False):
        y = x
        for layer in self.layers:
            y = layer.train(y, training)
        return y
    
    def collect(self, x):
        y = x
        stats = {}
        for layer in self.layers:
            y, stat = layer.collect(y)
            stats.update(stat)
        return y, stats
    
    def get_weights(self):
        weights_dict = {}
        for layer in self.layers:
            weights_dict.update(layer.get_weights())
        return weights_dict
        
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
        
#############

class layer:
    weight_id = 0
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)
    
    def collect(self, qx, x):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, weights=None, relu=True):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        self.relu = relu

        if 'g' in weights[self.weight_id].keys():
            f, b, g, mean, var = weights[self.weight_id]['f'], weights[self.weight_id]['b'], weights[self.weight_id]['g'], weights[self.weight_id]['mean'], weights[self.weight_id]['var']

            # need to actually do mean/var in forward if using new weights.
            # f = np.random.normal(loc=0., scale=np.std(f), size=np.shape(f))
            # b = np.zeros_like(b)
            # g = np.ones_like(g)

            #############################

            '''
            var = np.sqrt(var + 1e-5)
            f = f * (g / var)
            b = b - (g / var) * mean
            '''
            
            #############################
            
            '''
            if self.weight_id == 0:
                std = np.array([0.229, 0.224, 0.225]) * 255. / 2.
                f = f / np.reshape(std, (3,1))

                mean = np.array([0.485, 0.456, 0.406]) * 255. / 2.
                expand_mean = np.ones(shape=(7,7,3)) * mean
                expand_mean = expand_mean.flatten()

                b = b - (expand_mean @ np.reshape(f, (7*7*3, 64)))
            '''
            
            #############################

            self.f = tf.Variable(f, dtype=tf.float32)
            self.g = tf.Variable(g, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
            
            self.mean = tf.constant(mean, dtype=tf.float32)
            self.std = tf.constant(np.sqrt(var + 1e-5), dtype=tf.float32)
            
        else:
            assert (False)

    def train(self, x, training=False):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        
        if training:
            conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
            # mean = tf.reduce_mean(conv, axis=[0,1,2])
            # _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
            # shouldnt this be the same thing ? 
            mean, var = tf.nn.moments(conv, axes=[0,1,2])
            std = tf.sqrt(var + 1e-5)
        else:
            mean = self.mean
            std = self.std
        
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        qf = quantize_and_dequantize(fold_f, -128, 127)
        qb = fold_b
        
        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + qb
        
        if self.relu: out = tf.nn.relu(conv)
        else:         out = conv

        out = quantize_and_dequantize(out, -128, 127)
        return out
        
    def collect(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
        mean, var = tf.nn.moments(conv, axes=[0,1,2])
        std = tf.sqrt(var + 1e-5)
        
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        # qf = quantize_and_dequantize(fold_f, -128, 127)
        # qb = fold_b
        qf, sf = quantize(fold_f, -128, 127)
        qb = quantize_predict(fold_b, sf, -2**24, 2**24-1)

        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + qb
        
        if self.relu: out = tf.nn.relu(conv)
        else:         out = conv

        # out = quantize_and_dequantize(out, -128, 127)
        qout, sout = quantize(out, -128, 127)
        return qout, {self.layer_id: {'std': std, 'mean': mean, 'scale': sf}}
    
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict

    def get_params(self):
        return [self.f, self.b, self.g]

#############

class res_block1(layer):
    def __init__(self, f1, f2, p, weights=None):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False)

        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
    def train(self, x, training=False):
        y1 = self.conv1.train(x, training)
        y2 = self.conv2.train(y1, training)
        y3 = tf.nn.relu(x + y2)
        return y3

    def collect(self, x):
        stats = {}
        y1, stat1 = self.conv1.collect(x)
        y2, stat2 = self.conv2.collect(y1)
        y3 = tf.nn.relu(x + y2)
        out, scale = quantize(y3, -128, 127)

        stats.update(stat1)
        stats.update(stat2)
        stats[self.layer_id] = {'scale': scale}
        return out, stats

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        return weights_dict
        
    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        return params

#############

class res_block2(layer):
    def __init__(self, f1, f2, p, weights=None):

        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, weights=weights, relu=False)
        
        self.layer_id = layer.layer_id
        layer.layer_id += 1

    def train(self, x, training=False):
        y1 = self.conv1.train(x, training)
        y2 = self.conv2.train(y1, training)
        y3 = self.conv3.train(x, training)
        y4 = tf.nn.relu(y2 + y3)
        return y4

    def collect(self, x):
        stats = {}
        y1, stat1 = self.conv1.collect(x)
        y2, stat2 = self.conv2.collect(y1)
        y3, stat3 = self.conv3.collect(x)

        y4 = tf.nn.relu(y2 + y3)
        out, scale = quantize(y4, -128, 127)

        stats.update(stat1)
        stats.update(stat2)
        stats.update(stat3)
        stats[self.layer_id] = {'scale': scale}
        return out, stats

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        weights3 = self.conv3.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)
        return weights_dict

    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        params.extend(self.conv3.get_params())
        return params

#############

class dense_block(layer):
    def __init__(self, isize, osize, weights=None):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.isize = isize
        self.osize = osize
        
        w, b = weights[self.weight_id]['w'], weights[self.weight_id]['b']
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def train(self, x, training=False):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc

    def collect(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc, {}

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'w': self.w, 'b': self.b}
        return weights_dict
        
    def get_params(self):
        return [self.w, self.b]

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x, training=False):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool, {}
    
    def get_weights(self):    
        weights_dict = {}
        return weights_dict
        
    def get_params(self):
        return []

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x, training=False):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool, {}
    
    def get_weights(self):    
        weights_dict = {}
        return weights_dict

    def get_params(self):
        return []





        
        
        
