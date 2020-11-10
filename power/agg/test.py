
import numpy as np
import matplotlib.pyplot as plt

events = np.load('./data/555.npy', allow_pickle=True).item()
x = np.concatenate(events['x']).tolist()
y = np.concatenate(events['y']).tolist()
frame = np.zeros((240, 304), dtype=np.uint8)
frame[y, x] = 1.

plt.imshow(frame)
plt.show()


