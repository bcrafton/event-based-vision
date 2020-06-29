
import numpy as np
from src.metrics.coco_eval import evaluate_detection

RESULT_FILE_PATHS = ["./src_data/17-10-19_10-21-22_1037500000_1097500000_bbox.npy"]
GT_FILE_PATHS = ["./src_data/17-10-19_10-21-22_1037500000_1097500000_bbox.npy"]

result_boxes_list = [np.load(p, allow_pickle=True) for p in RESULT_FILE_PATHS]
gt_boxes_list = [np.load(p, allow_pickle=True) for p in GT_FILE_PATHS]

# evaluate_detection(gt_boxes_list, result_boxes_list)
'''
print (gt_boxes_list[0]['ts'])
print (type(gt_boxes_list[0]))
print (gt_boxes_list[0])
print (dir(gt_boxes_list[0]))
'''
print (gt_boxes_list[0].dtype.names)

#################################

# print (len(gt_boxes_list[0]))
for i in range(len(gt_boxes_list[0])):
    (a,b,c,d,e,f,g,h) = gt_boxes_list[0][i]
    gt_boxes_list[0][i] = (a,b,c,d,e,f,g,0)
    
# for box in gt_boxes_list[0]:
#     print (box)
    
# evaluate_detection(gt_boxes_list, result_boxes_list)

# (59749999, 118.,  96.,  43.,  39., 0, 1., 2753)
# dtype=[('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])]

# so track_id is just nonsense.

# we just have to fill in the middle 6 boxes.

#################################

'''

why is data not in array of array anymore ?
OHH, before the visualization tool handled that for us
but our preprocessed data should be fine
where we index by the [x, y].

so we will need to keep the time stamp and id.

really hoping the id dosnt matter at all...

'''
