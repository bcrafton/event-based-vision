
import os

import csv
import numpy as np

import cv2

import queue
import threading

import json

#########################################

'''
exxact = 0
icsrl2 = 1

if exxact:
    path = '/home/bcrafton3/Data_HDD/mscoco/'
elif icsrl2:
    path = '/usr/scratch/bcrafton/mscoco/'
else:
    assert(False)
'''

#########################################

def get_images(path):

    images = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if 'jpg' in file:
                full_path = os.path.join(subdir, file)
                if (full_path not in images):
                    images.append(full_path)

    return images

#########################################

# we need a class remap table because COCO has 80 classes ... but the ids are not ordered.
# max(class_id) = 90 ... even though there are only 80 classes.
def get_cat_table(path, json_filename):
    table = {}

    json_file = open(json_filename)
    data = json.load(json_file)
    annotations = list(data['annotations'])
    json_file.close()

    cats = {}
    for annotation in annotations:
        id = annotation['category_id']
        if id not in cats.keys():
            cats[id] = id

    sorted_cats = sorted(cats.keys())
    for ii in range(len(sorted_cats)):
        table[sorted_cats[ii]] = ii

    return table

#########################################

def get_det_table(path, json_filename):
    table = {}

    json_file = open(json_filename)
    data = json.load(json_file)
    annotations = list(data['annotations'])
    json_file.close()

    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        cat_id = annotation['category_id']
        image_filename = path + 'train_images/COCO_train2014_%012d.jpg' % (int(image_id))

        new_det = (bbox, cat_id)

        if image_filename in table.keys():
            table[image_filename].append(new_det)
        else:
            table[image_filename] = [new_det]

    return table

#########################################

def preprocess(filename, det_table, cat_table):
    # image
    image = cv2.imread(filename)
    shape = np.shape(image)
    (h, w, _) = shape
    image = cv2.resize(image, (448, 448))
    image = np.reshape(image, [1, 448, 448, 3])
    assert(not np.any(np.isnan(image)))
    scale_w = 448 / w
    scale_h = 448 / h

    # dets
    dets = det_table[filename]
    ndets = len(dets)
    coords  = np.zeros(shape=[ndets, 7, 7, 5])
    obj     = np.zeros(shape=[ndets, 7, 7])
    no_obj  = np.ones(shape=[ndets, 7, 7])
    cats    = np.zeros(shape=[ndets, 7, 7])

    for ii in range(ndets):
        det = dets[ii]
        # are we sure this is bounding box encoding ? 
        [x, y, w, h], cat_id = det

        cat = cat_table[cat_id]

        # these are bbox coords, not image shape.
        # all should be less than 448, but no lower bound.
        x = x * scale_w
        y = y * scale_h
        w = w * scale_w
        h = h * scale_h

        # try centering everything.
        x = x + 0.5 * w
        y = y + 0.5 * h

        if not (x <= 448.1 and y <= 448.1 and w <= 448.1 and h <= 448.1):
            print (x, y, w, h, shape, scale_w, scale_h)
            assert(x <= 448.1 and y <= 448.1 and w <= 448.1 and h <= 448.1)

        xc = int(x) // 64
        yc = int(y) // 64

        # should put more asserts in here...
        # make sure we are between 0 and 64 before / 64.

        x = (x - xc * 64.) / 64. # might want to clip this to zero
        y = (y - yc * 64.) / 64. # might want to clip this to zero
        w = w / 448.
        h = h / 448.

        coords[ii, xc, yc, :] = np.array([x, y, w, h, 1.])
        obj[ii, xc, yc] = 1.
        no_obj[ii, xc, yc] = 0.
        cats[ii, xc, yc] = cat

    return image, (coords, obj, no_obj, cats)

#########################################

def fill_queue(images, det_table, cat_table, q):
    ii = 0
    epoch = 0
    last = len(images) - 1

    while(True):
        if not q.full():
            filename = images[ii]

            if ii < last:
                ii = ii + 1
            else:
                ii = 0
                epoch = epoch + 1
                print (epoch) 

            if filename in det_table.keys():
                image, det = preprocess(filename, det_table, cat_table)
            else:
                continue

            q.put((image, det))

#########################################

class LoadCOCO:

    def __init__(self, batch_size, path):
        self.batch_size = batch_size
        self.path = path

        self.cat_table = get_cat_table(self.path, self.path + 'train_labels/instances_train2014.json')

        self.train_images = sorted(get_images(self.path + 'train_images'))
        self.train_det_table = get_det_table(self.path, self.path + 'train_labels/instances_train2014.json')

        self.q = queue.Queue(maxsize=128)
        thread = threading.Thread(target=fill_queue, args=(self.train_images, self.train_det_table, self.cat_table, self.q))
        thread.start()

    def pop(self):
        # pred   = [4, -1, 7, 7, 2 * 5 + 80]
        # coord  = [4, -1, 4, 7, 7, 5]
        # obj    = [4, -1, 7, 7]
        # no_obj = [4, -1, 7, 7]
        # cat    = [4, -1, 7, 7]

        assert(not self.empty())

        ################################

        batch = []; max_ndet = 0
        for b in range(self.batch_size):
            image, (coord, obj, no_obj, cat) = self.q.get()
            ndet = len(coord)
            max_ndet = max(max_ndet, ndet)
            batch.append((image, (coord, obj, no_obj, cat)))

        ################################

        images = []; coords = []; objs = []; no_objs = []; cats = []; vlds = []
        for b in range(self.batch_size):
            image, (coord, obj, no_obj, cat) = batch[b]
            vld = np.ones_like(obj)
            ndet = len(coord)
            pad = max_ndet - ndet
            if pad > 0:
                coord_pad  = np.zeros(shape=(pad, 7, 7, 5)); coord  = np.concatenate((coord, coord_pad), axis=0)
                obj_pad    = np.zeros(shape=(pad, 7, 7));    obj    = np.concatenate((obj, obj_pad), axis=0)
                no_obj_pad = np.zeros(shape=(pad, 7, 7));    no_obj = np.concatenate((no_obj, no_obj_pad), axis=0)
                cat_pad    = np.zeros(shape=(pad, 7, 7));    cat    = np.concatenate((cat, cat_pad), axis=0)
                vld_pad    = np.zeros(shape=(pad, 7, 7));    vld    = np.concatenate((vld, vld_pad), axis=0)

            images.append(image); coords.append(coord); objs.append(obj); no_objs.append(no_obj); cats.append(cat); vlds.append(vld)

        ################################

        images = np.concatenate(images, axis=0)

        coords  = np.stack(coords, axis=0)
        objs    = np.stack(objs, axis=0)
        no_objs = np.stack(no_objs, axis=0)
        cats    = np.stack(cats, axis=0)

        vlds    = np.stack(vlds, axis=0)

        ################################

        return images, (coords, objs, no_objs, cats, vlds)

    def empty(self):
        return self.q.qsize() < self.batch_size

    def full(self):
        return self.q.full()

###################################################################

















