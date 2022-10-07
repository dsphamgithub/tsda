import os
import argparse
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datetime import datetime
from glob import glob

current_dir = os.path.dirname(os.path.realpath(__file__))


# singularity run --nv -B $MYGROUP:/temp ${MYGROUP}/singularity_images/tf_container_2203_efficientdet.sif python eval_dataset.py

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/group/director2191/krados/datasets/gtsdb/test', 
                    help='Path to dataset')
parser.add_argument('--tfrecord_dir', default='/group/director2191/krados/datasets/tfrecords/gtsdb', 
                    help='Path to tfrecord dataset to use')
parser.add_argument('--model_name', default='efficientdet-d0', 
                    help='Model name')
parser.add_argument('--models_dir', default='/group/director2191/krados/model_dir/gtsdb', 
                    help='Path to directory containing the models')
parser.add_argument('--eval_dir', default='/group/director2191/krados/eval_dir/gtsdb', 
                    help='Path to output directory for evaluation json file')
parser.add_argument('--eval_type', choices=['valid', 'test', 'none'], default='valid', 
                    help='The type of dataset being used for evaluation')
parser.add_argument('--annotation_type', choices=['single', 'original'], default='single', 
                    help='Whether we are using single class labels or original labels')
parser.add_argument('--batch_size', default='4')

args = parser.parse_args()


def evaluate_dataset(model_dir, batch_size):
    hparams = "num_classes=1,moving_average_decay=0,mixed_precision=True,optimizer=adam,anchor_scale=1.5,max_level=7"
    if args.eval_type == 'none':
        val_file_pattern = os.path.join(args.tfrecord_dir, f'*.tfrecord')
    else:
        val_file_pattern = os.path.join(args.tfrecord_dir, f'{args.eval_type}-*.tfrecord')
    eval_samples = len(glob(args.dataset_dir + '/**/*.jpg', recursive=True))
    save_name = f'{Path(args.models_dir).parts[-1]}_{args.model_name}'
    save_path = os.path.join(args.eval_dir, f'{save_name}.npy')
    print(f'Saving results to {save_path}')

    cmd = f'python3 {current_dir}/efficientdet/tf2_eval.py \
        --eval_samples={eval_samples} \
        --val_file_pattern={val_file_pattern} \
        --model_name={args.model_name} \
        --model_dir={model_dir} \
        --batch_size={batch_size} \
        --hparams={hparams} \
        --eval_dir={args.eval_dir}'
    print(cmd)
    os.system(cmd)
    
    output_file = os.path.join(args.eval_dir, 'detections_testdev_results.npy')
    os.rename(output_file, save_path)
    return save_path


def calc_metrics(annotation_path, evaluation_path):
    with open(annotation_path, 'r') as f:
        gt_annotations = json.load(f)
    
    coco_gt = COCO(annotation_path)
    coco_dt = coco_gt.loadRes(np.load(evaluation_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    img_ids = list(range(gt_annotations['images'][-1]['id'] + 1))
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                    'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']  # From coco_metric.py
    coco_metrics = coco_eval.stats
    metric_dict = {}
    for i, name in enumerate(metric_names):
      metric_dict[name] = coco_metrics[i]
    print(args.model_name, metric_dict)


if __name__ == '__main__':
    # Evaluate the dataset
    if not os.path.exists(args.eval_dir):
        os.mkdir(args.eval_dir)
    
    models = glob(args.models_dir + f'/*{args.model_name}*')
    model_dir = sorted(models, key=os.path.getmtime)[-1]
    print(model_dir)
    eval_file = evaluate_dataset(model_dir, batch_size=args.batch_size)

    ann_name = "_single_annotations" if args.annotation_type == 'single' else "_annotations"
    ann_file = os.path.join(args.dataset_dir, f"{ann_name}.coco.json")
    calc_metrics(ann_file, eval_file)
