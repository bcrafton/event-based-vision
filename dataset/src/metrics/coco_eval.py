"""
Compute the COCO metric on bounding box files by matching timestamps

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    # print (gts)
    # print (len(gts)) # 16
    # print (len(gts[0])) # 1, 1, 3, 2, ...
    # print (gts[0][0]) # (33099999, 0., 150., 34., 21., 0, 0.6735892, 0)
    
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox[1], bbox[2]
            w, h = bbox[3], bbox[4]
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox[5]) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox[5]) + 1,
                'score': float(bbox[6]),
                'bbox': [bbox[1], bbox[2], bbox[3], bbox[4]],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
               
    return dataset, results
    
    
    
    
