"""Module that contains a set of functions to evaluate predicted damages against ground truth damages to return a
metrics object. Metrics include average precision (AP), receiver operating curve (roc), and basic error
metrics (MAE, RMSE, MBE).
"""

import sys
import sklearn.metrics as metrics
from collections import Counter, defaultdict
from typing import List, Dict
from tqdm import tqdm

import numpy as np

from detection_eval import Box, BoundingBox, calculate_all_points_average_precision


# Class to store metrics
class MetricPerClass:
    def __init__(self, cls, num_gts, num_dets, gt_pred_map):
        self.class_id = cls
        self.num_groundtruth = num_gts
        self.num_detection = num_dets
        self.gt_pred_map = gt_pred_map
        
    def update_ap_metrics(self, pre, rec, ap, mpre, mrec):
        self.precision = pre
        self.recall = rec
        self.ap = ap
        self.interpolated_precision = mpre
        self.interpolated_recall = mrec
        
    def update_roc_metrics(self, tp_rates, fp_rates, auc):
        self.tp_rates = tp_rates
        self.fp_rates = fp_rates
        self.roc_auc = auc
        
    def update_basic_metrics(self, mae, rmse, mbe):
        self.mae = mae
        self.rmse = rmse
        self.mbe = mbe

    @staticmethod
    def mAP(results: Dict[str, 'MetricPerClass']):
        return np.average([m.ap for m in results.values() if m.num_groundtruth > 0])


# Get average precision
def get_ap_metrics(gt_pred_map, num_sectors, dmg_threshold):
    num_preds = len(gt_pred_map)
    tps = np.zeros((num_preds, num_sectors))
    fps = np.zeros((num_preds, num_sectors))
    npos = 0
    
    for i in range(num_preds):
        gt, pred = gt_pred_map[i]
        pred_damages = np.array([d > dmg_threshold for d in pred])
        gt_damages = np.array([d > dmg_threshold for d in gt])
        npos += np.sum(gt_damages)
        # tp if sector is damaged above threshold in both gt and pred
        tps[i, :] = np.bitwise_and(pred_damages, gt_damages)
        # fp if sector is damaged above threshold in pred but not in gt
        fps[i, :] = np.bitwise_and(pred_damages, np.invert(gt_damages))
    
    # Compute precision, recall and average precision
    tps = np.reshape(tps, -1)
    fps = np.reshape(fps, -1)
    cumulative_fps = np.cumsum(fps)
    cumulative_tps = np.cumsum(tps)
    recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
    precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
    ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
    return ap, mpre, mrec, precisions, recalls


# Get receiver operating characteristic (ROC) curve
def get_roc_metrics(gt_pred_map, num_thres=50):
    num_preds = len(gt_pred_map)
    tp_rates = []
    fp_rates = []
    
    dmg_thresholds = np.linspace(0, 1, num_thres)
    for thres in tqdm(dmg_thresholds):
        tps = 0; fns = 0; fps = 0; tns = 0
        
        for i in range(num_preds):
            gt, pred = gt_pred_map[i]
            pred_damages = np.array([d > thres for d in pred])
            gt_damages = np.array([d > thres for d in gt])
            tps += np.sum(np.bitwise_and(pred_damages, gt_damages))  # true positive
            fns += np.sum(np.bitwise_and(np.invert(pred_damages), gt_damages))  # false negative
            fps += np.sum(np.bitwise_and(pred_damages, np.invert(gt_damages)))  # false positive
            tns += np.sum(np.bitwise_and(np.invert(pred_damages), np.invert(gt_damages)))  # true negative
        tp_rate = tps / (tps + fns) if (tps + fns) > 0 else 0
        fp_rate = tps / (tps + fps) if (tps + fps) > 0 else 0
        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)
    
    coords_dict = dict(zip(fp_rates, tp_rates))  # Remove identical instances of x coords  
    coords = sorted(coords_dict.items(), key=lambda x: x[0])  # Ensure x values are monotonic
    fp_rates, tp_rates = zip(*coords)
    auc = metrics.auc(np.array(fp_rates), np.array(tp_rates))
    return fp_rates, tp_rates, auc


# Get basic regression style error metrics
def get_basic_metrics(gt_pred_map):
    gt_damages, pred_damages = zip(*gt_pred_map)
    gt_damages = np.array(gt_damages)
    pred_damages = np.array(pred_damages)
    mae = np.mean(np.abs(gt_damages - pred_damages))
    rmse = np.sqrt(np.mean((gt_damages - pred_damages) ** 2))
    mbe = np.mean(gt_damages - pred_damages)

    ## DEBUG
    # print("\ngt_damages | pred_damages")
    # for gt_damages, pred_damages in zip(gt_damages, pred_damages):
    #     print(gt_damages, '|', pred_damages)
    # print()
    ##

    return mae, rmse, mbe


# TODO: iou_threshold in docstring but not used... perhaps it should be used?
def get_all_metrics(gold_standard: List[BoundingBox],
                    predictions: List[BoundingBox],
                    dmg_threshold: float = 0.2, 
                    num_sectors: int = 4, 
                    metrics: List[str] = ['ap', 'roc', 'basic']) -> Dict[str, MetricPerClass]:
    """
    Args:
        gold_standard: ground truth bounding boxes;
        predictions: detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        dmg_threshold: required damage level for sector to be marked as 'damaged';
        num_sectors: number of sectors in the damage assessment;
    Returns:
        A dictionary containing metrics of each class.
    """
    ret = {}  # list containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.class_id for b in gold_standard))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.class_id == c]  # type: List[BoundingBox]
        golds = [b for b in gold_standard if b.class_id == c]  # type: List[BoundingBox]
        gt_pred_map = []

        # Sort detections by decreasing confidence
        preds = sorted(preds, key=lambda b: b.score, reverse=True)

        # Create dictionary with amount of gts for each image
        counter = Counter([cc.image_id for cc in golds])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Pre-processing groundtruths of the some image
        image_id2gt = defaultdict(list)
        for b in golds:
            image_id2gt[b.image_id].append(b)
            
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
                  
            if counter[preds[i].image_id][mas_idx] == 0:
                # Append to output list
                gt_pred_map.append((gt[mas_idx].damages, preds[i].damages))
                counter[preds[i].image_id][mas_idx] = 1  # Flag as already 'seen'
        
        # Add class result in the dictionary to be returned
        r = MetricPerClass(c, len(golds), len(preds), gt_pred_map)
        ret[c] = r
        
        # Calculate metrics
        if 'ap' in metrics:
            print('Calculating AP metrics')
            ap, mpre, mrec, precisions, recalls = get_ap_metrics(gt_pred_map, num_sectors, dmg_threshold)
            r.update_ap_metrics(precisions, recalls, ap, mpre, mrec)
        if 'roc' in metrics:
            print('Generating ROC values')
            fp_rates, tp_rates, auc = get_roc_metrics(gt_pred_map)
            r.update_roc_metrics(tp_rates, fp_rates, auc)
        if 'basic' in metrics:
            print('Computing basic metrics')
            mae, rmse, mbe = get_basic_metrics(gt_pred_map)
            r.update_basic_metrics(mae, rmse, mbe)
    
    if len(ret.keys()) == 1:
        ret = list(ret.values())[0]
    return ret
