
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix
from bc_utils.conv_utils import conv_output_length

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        self.macs = 0
        
    def forward(self, x):
        y = x
        for layer in self.layers:
            macs,y = layer.forward(y)
            # print("Macs : {}".format(macs))
            self.macs += macs
        return self.macs,y
    
#############

class layer:
    weight_id = 0
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def forward(self, x):        
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, relu=True):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        self.relu = tf.constant(relu)

    def forward(self, x):
        h,w,c = x
        h = conv_output_length(input_length=h+2*self.pad,filter_size=self.k, padding='valid', stride=self.p)
        w = conv_output_length(input_length=w+2*self.pad,filter_size=self.k, padding='valid', stride=self.p)
        c = self.f2
        out = (h,w,c)
        macs = self.k * self.k * h * w * self.f1 * self.f2
        print("Layer ID : {} size : {} number of macs : {}".format(self.layer_id,out,macs))
        return macs,out 

#############

class res_block1(layer):
    def __init__(self, f1, f2, p):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p)
        self.conv2 = conv_block((3, 3, f2, f2), 1, relu=False)

        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
    def forward(self, x):
        mac1,y1 = self.conv1.forward(x)
        mac2,y2 = self.conv2.forward(y1)
        macs= mac1 + mac2
        return macs,y2
#############

class res_block2(layer):
    def __init__(self, f1, f2, p):
        self.total_macs = 0
        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        self.conv1 = conv_block((3, 3, f1, f2), p)
        self.conv2 = conv_block((3, 3, f2, f2), 1, relu=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, relu=False)
        
        self.layer_id = layer.layer_id
        layer.layer_id += 1

    def forward(self, x):
        mac1,y1 = self.conv1.forward(x)
        mac2,y2 = self.conv2.forward(y1)
        mac3, y3 = self.conv3.forward(x)
        macs = mac1+ mac2 + mac3
        print("Layer ID : {} size : {}".format(self.layer_id,y3))
        return macs,y3

#############

class dense_block(layer):
    def __init__(self, isize, osize, relu=True, dropout=False):
        self.total_macs = 0
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.isize = isize
        self.osize = osize
        self.relu = tf.constant(relu)
        self.dropout = tf.constant(dropout)

    def forward(self, x):
        assert(np.prod(x)== self.isize)
        out = self.osize
        macs = out * self.isize
        print("Layer ID : {} size : {} number of macs : {}".format(self.layer_id,out,macs))
        return macs,out

#############

class avg_pool(layer):
    def __init__(self, s, p):
        self.total_macs = 0
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def forward(self, x):
        h,w,c = x
        h = conv_output_length(input_length=h,filter_size=self.p, padding='same', stride=self.s)
        w = conv_output_length(input_length=w,filter_size=self.p, padding='same', stride=self.s)
        c = c
        out = (h,w,c)
        print("Layer ID : {} size : {}".format(self.layer_id,out))        
        return 0,out

#############

class max_pool(layer):
    def __init__(self, s, p):
        self.total_macs = 0
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def forward(self, x):
        h,w,c = x
        h = conv_output_length(input_length=h,filter_size=self.p, padding='same', stride=self.s)
        w = conv_output_length(input_length=w,filter_size=self.p, padding='same', stride=self.s)
        c = c
        out = (h,w,c)
        print("Layer ID : {} size : {}".format(self.layer_id,out))        
        return 0,out
    






        
        
        
