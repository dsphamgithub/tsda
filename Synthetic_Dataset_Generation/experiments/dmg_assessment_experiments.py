"""Script that uses damage_assessment_eval.py to generate a set of metrics by evaluating prediction data against ground
truth data. Creates matplotlib plot of precision vs. recall (with AP displayed), plot of the ROC curve with (AUC
displayed), and misc. metrics such as MAE, RMSE, MBE."""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from damage_assessment_eval import BoundingBox, get_all_metrics, get_roc_metrics, get_ap_metrics

current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/SGTS_Dataset/_single_annotations_array.npy')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/SGTS_Dataset_dmg_assess/dmg_net_d2.npy')
parser.add_argument('--out_dir', default='.')


def get_bounding_boxes(gt_detections, pred_detections):
    # gt detections array in format [image_id, xtl, ytl, width, height, damage_1, damage_2, ..., damage_n, class_id]
    gt_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], 
                        damages=det[5:-1]) for det in gt_detections]
    # pred detections array in format [image_id, xtl, ytl, width, height, score, damage_1, damage_2, ..., damage_n, class_id]
    pred_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], 
                              score=det[5], damages=det[6:-1]) for det in pred_detections]
    return gt_boxes, pred_boxes


if __name__ == '__main__':
    # TODO: command line arg for number of damage sectors

    args = parser.parse_args()
    gt_arr = np.load(args.gt_file)
    pred_arr = np.load(args.eval_file)
    gt_boxes, pred_boxes = get_bounding_boxes(gt_arr, pred_arr)
    metrics = get_all_metrics(gt_boxes, pred_boxes, dmg_threshold=0.2, num_sectors=4)
    
    real_roc_x = metrics.fp_rates
    real_roc_y = metrics.tp_rates
    
    real_pre = metrics.precision
    real_rec = metrics.recall
    
    # Random ROC curve
    gt_damages, pred_damages = zip(*metrics.gt_pred_map)
    random_damages =  np.random.rand(len(gt_damages), 4)
    random_map = list(zip(gt_damages, random_damages))
    
    fake_roc_x, fake_roc_y, rand_roc_auc = get_roc_metrics(random_map)
    fake_ap, _, _, fake_pre, fake_rec = get_ap_metrics(random_map, num_sectors=4, dmg_threshold=0.2)
    
    plt.plot(real_roc_x, real_roc_y, label='EfficientDet (AUC: {:.2f})'.format(metrics.roc_auc))
    plt.plot(fake_roc_x, fake_roc_y, label='Random (AUC: {:.2f})'.format(rand_roc_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig('ROC Curve', dpi=300)
    
    plt.clf()
    plt.plot(real_rec, real_pre, label='EfficientDet (AP: {:.2f})'.format(metrics.ap))
    plt.plot(fake_rec, fake_pre, label='Random (AP: {:.2f})'.format(fake_ap))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.savefig('Precision-Recall Curve', dpi=300)
    
    mean_pred_damage = np.mean(pred_damages)
    mean_gt_damage = np.mean(gt_damages)
    
    print(f'MAE: {metrics.mae}\nRMSE: {metrics.rmse}\nMBE: {metrics.mbe}\nMean Predicted Damage: {mean_pred_damage}\nMean Ground Truth Damage: {mean_gt_damage}')
