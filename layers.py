
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x):
        y = x
        for layer in self.layers:
            y = layer.train(y)
        return y
    
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

        if weights:
            f, b, g = weights[self.weight_id]['f'], weights[self.weight_id]['b'], weights[self.weight_id]['g']
            self.f = tf.Variable(f, dtype=tf.float32)
            self.g = tf.Variable(g, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
        else:
            f_np = init_filters(size=[self.k, self.k, self.f1, self.f2], init='glorot_uniform')
            self.f = tf.Variable(f_np, dtype=tf.float32)
            self.g = tf.Variable(np.ones(shape=self.f2), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=self.f2), dtype=tf.float32)

    def train(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
        bn = tf.nn.batch_normalization(conv, mean, var, self.b, self.g, 1e-5)        
        if self.relu: out = tf.nn.relu(conv)
        else:         out = conv
        return out 
    
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
        
    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(x + y2)
        return y3

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

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = self.conv3.train(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4

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
        
        if weights:
            w, b = weights[self.weight_id]['w'], weights[self.weight_id]['b']
            self.w = tf.Variable(w, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
        else:
            w_np = init_matrix(size=[isize, osize], init='glorot_uniform')
            self.w = tf.Variable(w_np, dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=self.osize), dtype=tf.float32)

    def train(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc

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
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
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
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def get_weights(self):    
        weights_dict = {}
        return weights_dict

    def get_params(self):
        return []





        
        
        
