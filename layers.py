
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
    def __init__(self, shape, p, weights=None, train=True, relu=True):
        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        self.relu = tf.constant(relu)
        self.train_flag = tf.constant(train)

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
        mean, var = tf.nn.moments(conv, axes=[0,1,2])
        bn = tf.nn.batch_normalization(conv, mean, var, self.b, self.g, 1e-5)        
        if self.relu: out = tf.nn.relu(bn)
        else:         out = bn
        return out 
    
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict

    def get_params(self):
        if self.train_flag:
            return [self.f, self.b, self.g]
        else:
            return []

#############

class res_block1(layer):
    def __init__(self, f1, f2, p, weights=None, train=True):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights, train=train)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, train=train, relu=False)

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
    def __init__(self, f1, f2, p, weights=None, train=True):

        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights, train=train)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, train=train, relu=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, weights=weights, train=train, relu=False)
        
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
    def __init__(self, isize, osize, weights=None, train=True, relu=True, dropout=False):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.isize = isize
        self.osize = osize
        self.relu = tf.constant(relu)
        self.dropout = tf.constant(dropout)
        self.train_flag = tf.constant(train)

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

        if self.relu: out = tf.nn.relu(fc)
        else:         out = fc

        if self.dropout: out = tf.nn.dropout(out, 0.5)

        return out

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'w': self.w, 'b': self.b}
        return weights_dict
        
    def get_params(self):
        if self.train_flag:
            return [self.w, self.b]
        else:
            return []

#############

class avg_pool(layer):
    def __init__(self, s, p):
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
    def __init__(self, s, p):
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

#############

class conv_lstm(layer):
    def __init__(self, shape, weights=None, train=True, relu=True):

        self.k, _, self.f1, self.f2 = shape
        self.pad = self.k // 2
        self.train_flag = tf.constant(train)

        f_np = init_filters(size=[self.k, self.k, self.f1, self.f2], init='glorot_uniform')
        self.f = tf.Variable(f_np, dtype=tf.float32)

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'VALID')
        return conv
    
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'f': self.f}
        return weights_dict

    def get_params(self):
        if self.train_flag:
            return [self.f]
        else:
            return []

#############

class lstm_block(layer):
    def __init__(self, shape, weights=None, train=True):
        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.t, self.k, _, self.f1, self.f2 = shape
        self.pad = self.k // 2
        self.train_flag = tf.constant(train)

        self.a_x = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.i_x = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.f_x = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.o_x = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)

        self.a_h = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.i_h = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.f_h = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)
        self.o_h = conv_lstm(shape=(self.k, self.k, 1, self.f2), relu=False)


    def train(self, x):

        h_l = []
        s_l = []

        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0], [0, 0]])

        for t in range(self.t):
            x_t = x_pad[..., t]

            if t == 0:
                a = tf.nn.tanh(    self.a_x.train(x_t) )
                i = tf.nn.sigmoid( self.i_x.train(x_t) )
                f = tf.nn.sigmoid( self.f_x.train(x_t) )
                o = tf.nn.sigmoid( self.o_x.train(x_t) )
                s = a * i
            else:
                h_t = tf.pad(h_l[t-1], [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
                a = tf.nn.tanh(    self.a_x.train(x_t) + self.a_h.train(h_t) ) 
                i = tf.nn.sigmoid( self.i_x.train(x_t) + self.i_h.train(h_t) )
                f = tf.nn.sigmoid( self.f_x.train(x_t) + self.f_h.train(h_t) )
                o = tf.nn.sigmoid( self.o_x.train(x_t) + self.o_h.train(h_t) )
                s = a * i + s_l[t-1] * f

            h = tf.nn.tanh(s) * o
            h_l.append(h)
            s_l.append(s)

        # out = tf.stack(h_l, axis=1)
        out = h_l[self.t-1]
        return out 

    '''
    def train(self, x):

        h_l = []
        s_l = []

        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0], [0, 0]])
    
        x_t = x_pad[..., 0]

        a = tf.nn.tanh(    self.a_x.train(x_t) )
        i = tf.nn.sigmoid( self.i_x.train(x_t) )
        f = tf.nn.sigmoid( self.f_x.train(x_t) )
        o = tf.nn.sigmoid( self.o_x.train(x_t) )
        s = a * i

        h = tf.nn.tanh(s) * o

        # out = tf.stack(h_l, axis=1)
        out = h
        return out 
    '''

    def get_weights(self):
        assert (False)

    def get_params(self):
        params = []
        params.extend(self.a_x.get_params())
        params.extend(self.i_x.get_params())
        params.extend(self.f_x.get_params())
        params.extend(self.o_x.get_params())
        params.extend(self.a_h.get_params())
        params.extend(self.i_h.get_params())
        params.extend(self.f_h.get_params())
        params.extend(self.o_h.get_params())
        return params

#############
        
        
