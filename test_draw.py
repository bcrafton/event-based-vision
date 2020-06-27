
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

x = np.load('/home/brian/Desktop/event-based-vision/dataset/data/0.npy', allow_pickle=True).item()

x, y = x['x'], x['y']

idx = 5
image, dets  = x[idx], y[idx]
image = image[:, :, -1]
assert (np.shape(image) == (240, 288))
image = image.astype(float)

for det in dets:
    _, x, y, w, h, _, _, _ = det
    # pt1 = (int(x-0.5*w), int(y-0.5*h))
    # pt2 = (int(x+0.5*w), int(y+0.5*h))
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    cv2.rectangle(image, pt1, pt2, 0, 1) # grayscale image so color (arg 4) = integer.

plt.imshow(image)
plt.show()

