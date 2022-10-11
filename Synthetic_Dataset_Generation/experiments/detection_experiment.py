"""Functions to define three experiments to apply on prediction data against ground truth data (note: evaluation dataset
must have only 1 ground truth annotation per image):

- sequence experiment: evaluates metrics of different sequences separately; rounds average sequence damage to nearest
  0.1; averages metrics for sequences with the same rounded damage level; plots metric vs damage curve. Only applicable
  if the evaluation dataset is a sequences dataset.
- distance experiment: calculates metrics for detections split by the ground truth bounding box area; plots metrics vs.
  area curve.
- damage experiment: calculate metrics for detections split by ground truth damage (rounded to nearest 0.1); plots
  metrics vs damage curve.
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from detection_eval import BoundingBox, get_voc_metrics, Box


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences_8/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences_8/1.0_augmented_efficientdet-d2.npy', 
                    help='File containing evaluated detections as a numpy file')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence in dataset')
parser.add_argument('--experiment', default='distance', choices=['damage', 'distance', 'sequence'] , help='Type of experiment to evaluate')


class SequenceEvaluation:
    def __init__(self, gt_boxes, pred_boxes):
        self.pred_boxes = pred_boxes
        self.gt_boxes = sorted(gt_boxes, key=lambda x: x.image_id)

    # TODO: These herustics currently aren't used anywhere, add them to new versions of downstream functions
    #       See here, search for 'heuristic': https://github.com/ai-research-students-at-curtin/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/commit/ee2afa2687d3f35230539826bde784d101efb22f
    
    # Proposed heuristic 1: use the bounding box with the maximum score for each image sequence.    
    def score_heuristic(self):
        """
        Choose the detection among the ones generated for this sequence, which has the highest score.
        """
        det = self.pred_boxes[np.argmax([bbox.score for bbox in self.pred_boxes])]
        [ann] = list(filter(lambda x: x.image_id == det.image_id, self.gt_boxes))
        return Box.intersection_over_union(det, ann)
    
    # Proposed heuristic 2: use the bounding box with area closest to optimal area
    def area_heuristic(self, opt_area):
        """
        Choose the detection among the ones generated for this sequence which has size closest to the optimal width.
        """
        area_diffs = []  # Diffs between this sequence's detection areas and optimal area
        for det in self.pred_boxes:
            det_area = (det.xbr - det.xtl) * (det.ybr - det.ytl)
            area_diffs.append(abs(det_area - opt_area))
        optimal_det = self.pred_boxes[np.argmin(area_diffs)]
        [ann] = list(filter(lambda x: x.image_id == optimal_det.image_id, self.gt_boxes))
        return Box.intersection_over_union(optimal_det, ann)
        
                
def get_metrics(gt, pred):
    """
    Calculates the metrics for a given set of ground truth and predicted detections.
    Metrics in the format [AP50, AP75, AP95, max precision, max recall, min precision, min recall]
    """
    # TODO: Match format listed in metrics_by_param(), or change format listed there:
    #       [AP50, mAP, Maximum IOU, Minimum IOU, Mean IOU, Maximum Score, Minimum Score, Mean Score] 
    columns = ['AP50', 'mAP', 'Mean IOU', 'Mean Score']
    metrics = np.zeros(len(columns))
    AP40_metrics = get_voc_metrics(gt, pred, iou_threshold=0.4)
    tp_IOUs = AP40_metrics.tp_IOUs
    tp_scores = AP40_metrics.tp_scores
    APs = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        APs.append(get_voc_metrics(gt, pred, iou_threshold=threshold).ap)
    metrics[0] = APs[0]
    metrics[1] = np.mean(APs)
    metrics[2] = np.mean(tp_IOUs)
    metrics[3] = np.mean(tp_scores)
    return metrics, columns


def get_bounding_boxes(gt_detections, pred_detections):
    # gt detections array in format [image_id, xtl, ytl, width, height, damage, class_id]
    gt_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4]) 
                        for det in gt_detections]
    # pred detections array in format [image_id, xtl, ytl, width, height, score, class_id]
    pred_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], score=det[5])
                    for det in pred_detections]
    return gt_boxes, pred_boxes


def prune_detections(detections_array, max_detections=50):
    sorted_indices = np.argsort(detections_array[:, 5], axis=0)
    pruned_arr = detections_array[sorted_indices][::-1]
    return pruned_arr[:max_detections]


def split_by_area(gt_arr, split_arr, round_areas=True):
    out_dict = defaultdict(list)
    gt_dict = defaultdict(set)
    
    for i in range(len(split_arr)):
        id = int(split_arr[i, 0])
        area = gt_arr[id, 3] * gt_arr[id, 4]
        if round_areas:
            area = round(area / 2500, 1) * 2500
        out_dict[area].append(split_arr[i])
        gt_dict[area].add(id)
        
    areas = list(gt_dict.keys())
    for area in areas:
        if len(gt_dict[area]) < 5:
            out_dict.pop(area)
        
    areas, dist_arrs = zip(*out_dict.items())
    dist_arrs = np.array([np.array(dist_arrs[i]) for i in range(len(dist_arrs))], dtype=object)
    dist_arrs = dist_arrs[np.argsort(areas)]
    return dist_arrs, sorted(areas)


def split_by_damage(gt_arr, split_arr):
    out_dict = defaultdict(list)
    
    for i in range(len(split_arr)):
        id = int(split_arr[i, 0])
        dmg = round(gt_arr[id, -2], 1)
        out_dict[dmg].append(split_arr[i])
        
    damages, dmg_arrs = zip(*out_dict.items())
    dmg_arrs = np.array([np.array(dmg_arrs[i]) for i in range(len(dmg_arrs))], dtype=object)
    dmg_arrs = dmg_arrs[np.argsort(damages)]
    return dmg_arrs, sorted(damages)
 

def split_by_sequence(split_arr, num_frames):
    split_indices = []
    # Sort array by image_id
    split_arr = split_arr[np.argsort(split_arr[:, 0])]
    image_ids = split_arr[:, 0]
    for i in range(len(image_ids)):
        id = int(image_ids[i])
        is_boundary_index = image_ids[i - 1] != image_ids[i]  # Check if image_id has changed
        if id % num_frames == 0 and id != 0 and is_boundary_index:
            split_indices.append(i)
    out_arr = np.split(split_arr, split_indices)
    damages = np.around([np.mean(seq[:, -2]) for seq in out_arr], 1)
    return out_arr, damages  


def metrics_by_param(gt_arr, pred_arr, num_frames=8, param='sequence'):
    gt_arr = gt_arr[np.argsort(gt_arr[:, 0])]
    pred_arr = pred_arr[np.argsort(pred_arr[:, 0])]
    
    if param == 'sequence':
        param_gts, vars = split_by_sequence(gt_arr, num_frames)
        param_preds, _ = split_by_sequence(pred_arr, num_frames)
    else:
        ## DEBUG
        # print(len(gt_arr))
        # print(len(pred_arr))
        ##
        param_gts, vars = globals()["split_by_" + param](gt_arr, gt_arr)
        param_preds, _ = globals()["split_by_" + param](gt_arr, pred_arr)
    
    max_detections = int(np.mean([len(arr) for arr in param_gts]) * 5)
    param_preds = [prune_detections(arr, max_detections) for arr in param_preds]
    
    # Format [param, AP50, mAP, Maximum IOU, Minimum IOU, Mean IOU, Maximum Score, Minimum Score, Mean Score] 
    # TODO: ^ This is out of date
    metrics_array = None
    
    # Iterate over each image sequence
    for i in range(min(len(param_preds), len(param_gts))):
        ## DEBUG
        # print()
        # print("i:", i)
        # print("len(param_gts):", len(param_gts))
        # print("len(param_preds):", len(param_preds))
        ##
        gt_boxes, pred_boxes = get_bounding_boxes(param_gts[i], param_preds[i])
        metrics, columns = get_metrics(gt_boxes, pred_boxes)
        row = np.zeros((1, len(metrics) + 1))
        row[0, 0] = vars[i]
        row[0, 1:] = metrics
        if metrics_array is None:
            metrics_array = row
        else:
            metrics_array = np.append(metrics_array, row, axis=0)
    if param == 'sequence':
        columns.insert(0, 'Damage')
    else:
        columns.insert(0, param.capitalize())
    return metrics_array, columns


def sequence_experiment(gt_arr, pred_arr, num_frames):
    metrics_array, columns = metrics_by_param(gt_arr, pred_arr, num_frames, param='sequence')
    
    damages = np.unique(metrics_array[:, 0])
    dmg_metrics = np.zeros((len(damages), metrics_array.shape[1]))
    for i, dmg in enumerate(damages):
        dmg_arr = metrics_array[np.nonzero(metrics_array[:, 0] == dmg)]
        dmg_metrics[i] = np.mean(dmg_arr, axis=0)
    return pd.DataFrame(dmg_metrics, columns=columns)


def distance_experiment(gt_arr, pred_arr):
    metrics_array, columns = metrics_by_param(gt_arr, pred_arr, param='area')
    return pd.DataFrame(data=metrics_array, columns=columns)


def damage_experiment(gt_arr, pred_arr):
    metrics_array, columns = metrics_by_param(gt_arr, pred_arr, param='damage')
    return pd.DataFrame(data=metrics_array, columns=columns)


if __name__ == '__main__':
    args = parser.parse_args()
    gt_arr = np.array(np.load(args.gt_file), dtype=np.float32)
    pred_arr = np.array(np.load(args.eval_file), dtype=np.float32)
    
    # A plot of a metric (e.g., mAP) against various damage average levels (e.g., 10%, 20%, etc.), where AP is evaluated
    # across a sequence, and metrics are averaged across sequences with the same average damage level.
    if args.experiment == 'sequence':
        df = sequence_experiment(gt_arr, pred_arr, args.num_frames)
        print(df)
        # See get_metrics() columns for possible value_vars values
        df_long = pd.melt(df, id_vars=['Damage'], value_vars=['Mean IOU'])
        fig = px.line(df_long, x='Damage', y='value', title='IOU vs. Damage Level', color='variable')
        
    # A plot of a metric (e.g., mAP) against pixel area of the sign in the image. Closer signs have higher pixel area.
    elif args.experiment == 'distance':
        df = distance_experiment(gt_arr, pred_arr)
        print(df)
        fig = px.line(df, x='Area', y='mAP', title='Average Precision (AP) vs. width of sign in pixels')
        
    # A plot of a metric (e.g., mAP) against damage level (10%, 20%, etc.), where AP is evaluated against annotations
    # with the same (mapped) damage level.
    elif args.experiment == 'damage':
        df = damage_experiment(gt_arr, pred_arr)
        print(df)
        fig = px.line(df, x='Damage', y='mAP', title='Average Precision (AP) vs. Damage Level')
    
    cwd = os.getcwd()
    name = args.eval_file.split('.')[0].split(os.sep)[-1]
    fig.write_html(f"{cwd}/{name}.html")
    with open(f"{cwd}/{name}.txt", 'w') as f:
        f.write(str(df))
