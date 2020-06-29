
import numpy as np
from src.metrics.coco_eval import evaluate_detection

RESULT_FILE_PATHS = ["./src_data/17-12-07_10-27-28_549500000_609500000_bbox.npy"]
GT_FILE_PATHS = ["./src_data/17-12-07_10-27-28_549500000_609500000_bbox.npy"]

result_boxes_list = [np.load(p, allow_pickle=True) for p in RESULT_FILE_PATHS]
gt_boxes_list = [np.load(p, allow_pickle=True) for p in GT_FILE_PATHS]

evaluate_detection(gt_boxes_list, result_boxes_list)
