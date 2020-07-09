
import os
import numpy as np
import cv2
from PIL import Image
import random
from multiprocessing import Process, Queue
import tensorflow as tf

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
    '''
    max_nd = 0
    for b in range(len(dets)):
        nd = len(dets[b])
        max_nd = max(max_nd, nd)
    '''
    max_nd = 8

    coords = []; objs = []; no_objs = []; cats = []; vlds = []
    for b in range(len(dets)):
        coord, obj, no_obj, cat, vld = det_tensor(dets[b], max_nd)
        coords.append(coord); objs.append(obj); no_objs.append(no_obj); cats.append(cat); vlds.append(vld)
    
    coords  = np.stack(coords, axis=0).astype(np.float32)
    objs    = np.stack(objs, axis=0).astype(np.float32)
    no_objs = np.stack(no_objs, axis=0).astype(np.float32)
    cats    = np.stack(cats, axis=0).astype(np.int32)
    vlds    = np.stack(vlds, axis=0).astype(np.float32)

    coords  = tf.convert_to_tensor(coords)
    objs    = tf.convert_to_tensor(objs)
    no_objs = tf.convert_to_tensor(no_objs)
    cats    = tf.convert_to_tensor(cats)
    vlds    = tf.convert_to_tensor(vlds)

    return coords, objs, no_objs, cats, vlds

def det_tensor(dets, max_nd):

    coord   = np.zeros(shape=[max_nd, 5, 6, 5])
    obj     = np.zeros(shape=[max_nd, 5, 6])
    no_obj  = np.ones(shape=[max_nd, 5, 6])
    cat     = np.zeros(shape=[max_nd, 5, 6])
    vld     = np.zeros(shape=[max_nd, 5, 6])
    
    nd = min(max_nd, len(dets))
    for idx in range(nd):

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

def fill_queue(tid, nbatch, batch_size, nthread, samples, q):
    # for batch in range(tid, len(samples), nthread):
    for batch in range(tid, nbatch, nthread):
        while q.full(): pass
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            example = batch * batch_size + i
            # print (example, len(samples), nbatch * batch_size)
            assert (example < len(samples))
            sample = np.load(samples[example], allow_pickle=True).item()
            x, y = sample['x'], sample['y']
            batch_x.append(x)
            batch_y.append(y)

        img = np.stack(batch_x, axis=0).astype(np.float32)
        coord, obj, no_obj, cat, vld = create_labels(batch_y)
        q.put((img, coord, obj, no_obj, cat, vld))

#########################################

class Loader:

    def __init__(self, path, nbatch, batch_size, nthread):
      
        ##############################
    
        self.path = path
        self.q = Queue(maxsize=32)
        self.nbatch = nbatch
        self.batch_size = batch_size
        self.nthread = nthread

        ##############################
        
        self.samples = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file == 'placeholder':
                    continue
                self.samples.append(os.path.join(subdir, file))
        
        ##############################

        random.shuffle(self.samples)
        remainder = len(self.samples) % self.batch_size
        if remainder: 
            self.samples = self.samples[:(-remainder)]

        ##############################
        
        self.threads = []
        for tid in range(self.nthread):
            thread = Process(target=fill_queue, args=(tid, self.nbatch, self.batch_size, self.nthread, self.samples, self.q))
            thread.start()
            self.threads.append(thread)

        ##############################

    def pop(self):
        assert(not self.empty())
        return self.q.get()

    def empty(self):
        # return self.q.qsize() == 0
        return self.q.empty()

    def full(self):
        return self.q.full()

    def join(self):
        assert (self.q.empty())
        for tid in range(self.nthread):
            self.threads[tid].join()

###################################################################
'''
def preprocess(filename):
    data = np.load(filename, allow_pickle=True).item()
    x, y = data['x'], data['y']
    coord, obj, no_obj, cat, vld = det_tensor(y)
    return x, coord, obj, no_obj, cat, vld

def collect_filenames(path):
    samples = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == 'placeholder':
                continue
            samples.append(os.path.join(subdir, file))

    return samples

def create_dataset(path):
    filenames = collect_filenames(path)
    dataset = tf.data.Dataset.list_files(filenames)
    labeled_dataset = dataset.map(preprocess)
    return labeled_dataset
'''
###################################################################









