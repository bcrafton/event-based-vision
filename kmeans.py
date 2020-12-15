
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt



#####################################

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#####################################

def collect_filenames(path):
    samples = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == 'placeholder':
                continue
            samples.append(os.path.join(subdir, file))
    return samples

#####################################

def extract(record):
    _feature={
        'id_raw':    tf.io.FixedLenFeature([], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(record, _feature)

    id = sample['id_raw']

    label = tf.io.decode_raw(sample['label_raw'], tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (8, 5, 6, 8))

    image = tf.io.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.cast(image, dtype=tf.float32) # this was tricky ... stored as uint8, not float32.
    image = tf.reshape(image, (240, 288, 12))

    return [id, image, label]

#####################################

offset_np = np.array([
[  [0, 0],   [0, 48],   [0, 96],   [0, 144],   [0, 192],   [0, 240]],
[ [48, 0],  [48, 48],  [48, 96],  [48, 144],  [48, 192],  [48, 240]],
[ [96, 0],  [96, 48],  [96, 96],  [96, 144],  [96, 192],  [96, 240]],
[[144, 0], [144, 48], [144, 96], [144, 144], [144, 192], [144, 240]],
[[192, 0], [192, 48], [192, 96], [192, 144], [192, 192], [192, 240]]
])

def grid_to_pix(box):
    box[..., 2] = np.square(box[..., 2]) * 240.
    box[..., 3] = np.square(box[..., 3]) * 288.
    box[..., 0] = 48. * box[..., 0] + offset_np[..., 0] - 0.5 * box[..., 2]
    box[..., 1] = 48. * box[..., 1] + offset_np[..., 1] - 0.5 * box[..., 3]

    tmp = np.around(box[..., 0])
    box[..., 0] = np.around(box[..., 1])
    box[..., 1] = tmp

    tmp = np.around(box[..., 2])
    box[..., 2] = np.around(box[..., 3])
    box[..., 3] = tmp
    return box

#####################################

files = collect_filenames('./dataset/train_v1')
tfrecord = tf.data.TFRecordDataset(files)
dataset = tfrecord.map(extract)

count = 0
boxes = []
for (id, image, label) in dataset:
    count += 1
    if count % 100 == 0:
        print (count)
    label = label.numpy()
    # print (np.shape(label))
    # (8, 5, 6, 8)
    obj = np.where(label[:, :, :, 4] == 1)
    box = grid_to_pix(label[:, :, :, 0:4])[obj]
    boxes.append(box)

#####################################

boxes = np.concatenate(boxes, axis=0)
boxes = boxes[:, 2:4]

#####################################

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=9).fit(boxes)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=8).fit(boxes)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=7).fit(boxes)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=6).fit(boxes)
print (kmeans.cluster_centers_)

kmeans = KMeans(n_clusters=5).fit(boxes)
print (kmeans.cluster_centers_)





