import argparse
import os
import json
import shutil
import argparse
import numpy as np
from glob import glob

from coco_utils import convert_to_single_label

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='/home/allenator/Pawsey-Internship/datasets/augmented/0.1_augmented',
                    help='Path to dataset')
parser.add_argument('--datasets_dir', default='/home/allenator/Pawsey-Internship/datasets', 
                    help='Path to directory containing datasets')
parser.add_argument('--single_dir', action='store_true',
                    help='Use this flag if the dataset only contains images and annotations with no subdirectories')
parser.add_argument('--num_damage_sectors', type=int, default=0,
                    help='Damage annotations will not be included in the tfrecord if this value is 0')


def pull_gtsdb(dataset_path):
    if os.path.basename(dataset_path) == 'gtsdb' and not os.path.exists(dataset_path):
        os.system(f'mkdir {args.dataset_path}')
        os.system('wget "https://app.roboflow.com/ds/UQRkeMI1UW?key=TuRmW7Gi5I" -O gtsdb.zip')
        os.system(f'unzip -q gtsdb.zip -d {args.dataset_path}')
        os.system(f'rm gtsdb.zip')


def convert_dataset_to_tfrecord(dataset_path, tfrecords_dir, annotations_file, num_shards, num_damage_sectors):
    dirname = os.path.basename(dataset_path)
    module_path = 'efficientdet/dataset/create_coco_tfrecord.py'
    os.system(f'python3 {module_path} \
    --image_dir={dataset_path} \
    --object_annotations_file={dataset_path}/{annotations_file} \
    --output_file_prefix={tfrecords_dir}/{dirname} \
    --num_shards={num_shards} \
    --num_damage_sectors={num_damage_sectors}')
    

def get_num_shards(dataset_path):
    # 800 images per shard
    image_paths = glob(dataset_path + '/**/*.jpg', recursive=True)
    return max(len(image_paths) // 800 + 1, 8)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    dataset_path = args.dataset.rstrip('/')
    dataset_name = os.path.basename(dataset_path)
    tfrecords_dir = os.path.join(args.datasets_dir, 'tfrecords')
    
    if not os.path.exists(tfrecords_dir):
        os.mkdir(tfrecords_dir)
    tfrecords_dir = os.path.join(tfrecords_dir, f'{dataset_name}')
    if os.path.exists(tfrecords_dir):
        shutil.rmtree(tfrecords_dir)
    
    pull_gtsdb(dataset_path)
    
    use_damages = (args.num_damage_sectors > 0 and args.num_damage_sectors is not None)
    sectors = args.num_damage_sectors
    num_shards = get_num_shards(dataset_path)
    
    if args.single_dir is not True:
        for subdir in glob(dataset_path + '/*'):
            if os.path.split(subdir)[-1] in ['train', 'valid', 'test']:
                num_shards = 8 if dataset_name == 'gtsdb' else num_shards  # 1 shard for each worker (when using 4 compute nodes)
                if not os.path.isdir(subdir):
                    continue
                convert_to_single_label(subdir, '_annotations.coco.json', '_single_annotations.coco.json')   
                convert_dataset_to_tfrecord(subdir, tfrecords_dir, '_single_annotations.coco.json', num_shards, sectors)
    else:
        if not os.path.exists(os.path.join(dataset_path, '_single_annotations.coco.json')):
            convert_to_single_label(dataset_path, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=use_damages)
        convert_dataset_to_tfrecord(dataset_path, tfrecords_dir, '_single_annotations.coco.json', num_shards, sectors)
