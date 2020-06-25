
import os
import numpy as np
import cv2
from PIL import Image
import random
from multiprocessing import Process, Queue

#########################################
'''
import torch
import torchvision
from torchvision import transforms

preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(filename):
    image = Image.open(filename).convert('RGB')
    image = preprocess_transform(image)
    print (filename)
    return image.numpy()
'''
#########################################

# if there is a mismatch between this and pytorch
# would first guess the np.floor we are using here.
# but the best thing we found to do
# was diff this and 'pytorch_resnet.py' for np.std(image) at various places.

def preprocess(filename):
    image = Image.open(filename).convert('RGB')
    
    H, W = image.height, image.width
    new_H = max(int(np.floor(H / W * 256)), 256)
    new_W = max(int(np.floor(W / H * 256)), 256)
    image = image.resize((new_W, new_H), Image.BILINEAR)
    image = np.array(image)
    assert (np.shape(image) == (new_H, new_W, 3))
    
    h1 = (new_H - 224) // 2
    h2 = h1 + 224
    w1 = (new_W - 224) // 2
    w2 = w1 + 224
    
    assert(h2 <= new_H)
    assert(w2 <= new_W)
    image = image[h1:h2, w1:w2, :]
    
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = image / 255
    # image = (image - mean) / std
    
    image = image // 2
    
    return image

#########################################

def fill_queue(tid, nbatch, batch_size, nthread, images, labels, q):
    for batch in range(tid, nbatch, nthread):
        while q.full(): pass
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            example = batch*batch_size + i
            assert (example < nbatch * batch_size)

            image = preprocess(images[example])
            batch_x.append(image)
            batch_y.append(labels[example])

        batch_x = np.stack(batch_x, axis=0).astype(np.float32)
        batch_y = np.array(batch_y)
        q.put((batch_x, batch_y))

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
        
        self.images = []
        self.labels = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file in ['keras_imagenet_val.py', 'keras_imagenet_train.py']:
                    continue
                
                self.images.append(os.path.join(subdir, file))
                label = int(subdir.split('/')[-1])
                self.labels.append(label)
        
        ##############################

        merge = list(zip(self.images, self.labels))
        random.shuffle(merge)
        self.images, self.labels = zip(*merge)

        remainder = len(self.images) % self.batch_size
        if remainder: 
            self.images = self.images[:(-remainder)]
            self.labels = self.labels[:(-remainder)]

        # print (len(self.images))
        # print (len(self.labels))

        ##############################
        
        self.threads = []
        for tid in range(self.nthread):
            thread = Process(target=fill_queue, args=(tid, self.nbatch, self.batch_size, self.nthread, self.images, self.labels, self.q))
            thread.start()
            self.threads.append(thread)

        ##############################

    def pop(self):
        assert(not self.empty())
        return self.q.get()

    def empty(self):
        return self.q.qsize() == 0

    def full(self):
        return self.q.full()

    def join(self):
        assert (self.q.empty())
        for tid in range(self.nthread):
            self.threads[tid].join()

###################################################################


















