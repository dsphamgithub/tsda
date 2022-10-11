"""Third party module that implements the VOC 2012 AP measurement to evaluate object detection predictions."""

import sys
from collections import Counter, defaultdict
from typing import List, Dict

import numpy as np

'''
Code from https://github.com/yfpeng/object_detection_metrics/blob/master/podm/podm.py

Copyright (c) 2020, Yifan Peng
All rights reserved.
Modified by Allen Antony

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

class Box:
    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
        """
                    0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)
        Args:
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
        """
        assert xtl <= xbr, f'xtl < xbr: xtl:{xtl}, xbr:{xbr}'
        assert ytl <= ybr, f'ytl < ybr: ytl:{ytl}, xbr:{ybr}'

        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def __str__(self):
        return 'Box[xtl={},ytl={},xbr={},ybr={}]'.format(self.xtl, self.ytl, self.xbr, self.ybr)

    @classmethod
    def intersection_over_union(cls, box1: 'Box', box2: 'Box') -> float:
        """
        Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
        two bounding boxes.
        """
        # If boxes dont intersect
        if not Box.is_intersecting(box1, box2):
            return 0
        intersection = Box.intersection_area(box1, box2)
        union = Box.union_areas(box1, box2, intersection_area=intersection)
        # Intersection over union
        iou = intersection / union
        assert iou >= 0, '{} = {} / {}, box1={}, box2={}'.format(iou, intersection, union, box1, box2)
        return iou

    @classmethod
    def is_intersecting(cls, box1: 'Box', box2: 'Box') -> bool:
        if box1.xtl > box2.xbr:
            return False  # boxA is right of boxB
        if box2.xtl > box1.xbr:
            return False  # boxA is left of boxB
        if box1.ybr < box2.ytl:
            return False  # boxA is above boxB
        if box1.ytl > box2.ybr:
            return False  # boxA is below boxB
        return True

    @classmethod
    def intersection_area(cls, box1: 'Box', box2: 'Box') -> float:
        xtl = max(box1.xtl, box2.xtl)
        ytl = max(box1.ytl, box2.ytl)
        xbr = min(box1.xbr, box2.xbr)
        ybr = min(box1.ybr, box2.ybr)
        # Intersection area
        return (xbr - xtl) * (ybr - ytl)

    @staticmethod
    def union_areas(box1: 'Box', box2: 'Box', intersection_area: float = None) -> float:
        if intersection_area is None:
            intersection_area = Box.intersection_area(box1, box2)
        return box1.area + box2.area - intersection_area


class BoundingBox(Box):
    def __init__(self, image_id: float, class_id: float, xtl: float, ytl: float, xbr: float, ybr: float,
                 score=None, damages: List[float] = None):
        """Constructor.
        Args:
            image_id: the image name.
            class_id: class id.
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
            score: (optional) the confidence of the detected class.
        """
        super().__init__(xtl, ytl, xbr, ybr)
        self.image_id = image_id
        self.score = score
        self.class_id = class_id
        self.damages = damages
        
        
def calculate_all_points_average_precision(recall, precision):
    """
    All-point interpolated average precision
    Returns:
        average precision
        interpolated precision
        interpolated recall
        interpolated points
    """
    mrec = [0] + [e for e in recall] + [1]
    mpre = [0] + [e for e in precision] + [0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii


class MetricPerClass:
    def __init__(self):
        self.class_id = None
        self.precision = None
        self.recall = None
        self.ap = None
        self.interpolated_precision = None
        self.interpolated_recall = None
        self.num_groundtruth = None
        self.num_detection = None
        self.tp = None
        self.fp = None
        self.tp_IOUs = None
        self.tp_scores = None

    @staticmethod
    def mAP(results: Dict[str, 'MetricPerClass']):
        return np.average([m.ap for m in results.values() if m.num_groundtruth > 0])
    

def get_voc_metrics(gold_standard: List[BoundingBox],
                    predictions: List[BoundingBox],
                    iou_threshold: float = 0.5) -> Dict[str, MetricPerClass]:
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Args:
        gold_standard: ground truth bounding boxes;
        predictions: detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        method: Pascal VOC metrics or damage metrics
    Returns:
        A dictionary containing metrics of each class.
    """
    ret = {}  # List containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.class_id for b in gold_standard))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.class_id == c]  # type: List[BoundingBox]
        golds = [b for b in gold_standard if b.class_id == c]  # type: List[BoundingBox]
        npos = len(golds)

        # Sort detections by decreasing confidence
        preds = sorted(preds, key=lambda b: b.score, reverse=True)
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))

        # Create dictionary with amount of gts for each image
        counter = Counter([cc.image_id for cc in golds])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Pre-processing groundtruths of the some image
        image_id2gt = defaultdict(list)
        for b in golds:
            image_id2gt[b.image_id].append(b)

        tp_IOUs = []
        tp_scores = []
        # Loop through detections to map them with respective ground truths
        for i in range(len(preds)):
            # Find ground truth image
            gt = image_id2gt[preds[i].image_id]
            max_iou = sys.float_info.min
            mas_idx = -1
            # Check IOU with ground truths from image
            for j in range(len(gt)):
                iou = Box.intersection_over_union(preds[i], gt[j])
                if iou > max_iou:
                    max_iou = iou
                    mas_idx = j
            
            ## DEBUG
            # print(f'preds[i].image_id: {preds[i].image_id}')
            # print(f'counter[preds[i].image_id]: {counter[preds[i].image_id]}')
            # print("<-------------------------------------------------------- ID NOT IN LIST ->") if not preds[i].image_id in stuff else None
            # print()
            ##

            # Metrics that are invariant with iou_threshold
            try:
                if counter[preds[i].image_id][mas_idx] == 0:
                    # Add IOU of best detection for this ground truth
                    tp_IOUs.append(max_iou)
                    # Add score of best detection for this ground truth
                    tp_scores.append(preds[i].score)
            except(TypeError):
                None  # Prevent crashing on image ids which weren't in counter
                # Crash:
                # "py', line 258, in get_voc_metrics
                #      if counter[preds[i].image_id][mas_idx] == 0:
                #  TypeError: 'int' object is not subscriptable"
            
            # Assign detection as true positive/don't care/false positive
            if max_iou >= iou_threshold:
                if counter[preds[i].image_id][mas_idx] == 0:
                    tps[i] = 1  # count as true positive
                    counter[preds[i].image_id][mas_idx] = 1  # Flag as already 'seen'
                else:
                    # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                    fps[i] = 1  # count as false positive
            else:
                fps[i] = 1  # count as false positive
                
        # Compute precision, recall and average precision
        cumulative_fps = np.cumsum(fps)
        cumulative_tps = np.cumsum(tps)
        recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
        precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
        # Depending on the method, call the right implementation
        ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
        # Add class result in the dictionary to be returned
        r = MetricPerClass()
        r.class_id = c
        r.precision = precisions
        r.recall = recalls
        r.ap = ap
        r.interpolated_recall = np.array(mrec)
        r.interpolated_precision = np.array(mpre)
        r.tp = np.sum(tps)
        r.fp = np.sum(fps)
        r.num_groundtruth = len(golds)
        r.num_detection = len(preds)
        r.tp_IOUs = tp_IOUs
        r.tp_scores = tp_scores
        ret[c] = r
        
    if len(ret.keys()) == 1:
        ret = list(ret.values())[0]
    return ret