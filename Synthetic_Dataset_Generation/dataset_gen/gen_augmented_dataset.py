"""Module to generate a dataset consisting of the original dataset augmented with images from the augment dataset.
Used to augment a real dataset with synthetic images.
"""

import argparse
import os
import json
import shutil
import argparse
import random
from datetime import datetime
from glob import glob
import ntpath

from utils import initialise_coco_anns, convert_to_single_label

parser = argparse.ArgumentParser()

parser.add_argument('--orig_dataset', default=None, type=str, required=True, help='Path to dataset to be augmented.')
parser.add_argument('--orig_annotations', default=None, type=str, required=True,
                    help='COCO annotations for original dataset.')
parser.add_argument('--augment_dataset', default=None, type=str,
                    help='Path to dataset that will be used to augment the original dataset.', required=True)
parser.add_argument('--augment_annotations', default=None, type=str, required=True,
                    help='Annotations for augment dataset')
parser.add_argument('--datasets_dir', default='./SGTS_Augmented', type=str,
                    help='Path to directory storing the dataset')
parser.add_argument('--augmentation', default=0.25, type=float,
                    help='Proportion of the original dataset to be augmented.')
parser.add_argument('--extend', default=False, help='Whether or not to keep the entire original dataset, '
                    'extending it so that it constitues the augmentation ratio in the final dataset.')
parser.add_argument('--check_dir', default=None, type=str, help='Path to a directory which we want to check for '
                    'cross-dataset duplicates. If one of our synthetic images is there already, it will not be added to '
                    'this new dataset. This is helpful for ensuring a test set contains no training set images if '
                    'pulling from the same pool of synthetic images.')
parser.add_argument('--check_dir_2', default=None, type=str)
parser.add_argument('--seed', default=0, type=int, help='Seed for random.sample function.')
parser.add_argument('--no_damage', action='store_true', help='Specify when we do not want to include (and enforce) the '
                    'presence of damage in annotations')

NUM_DMG_SECTORS = 4  # TODO: Make an argument?


def extend_annotations(final_annotations, image_paths, annotations, no_damage):
    image_paths = set([os.path.basename(p) for p in image_paths])
    image_id = len(final_annotations['images'])
    annotation_id = len(final_annotations['annotations'])
    
    for img_json in annotations['images']:
        path = os.path.basename(img_json['file_name'])
        
        if path not in image_paths:
            continue
        
        image_anns = list(filter(lambda ann: ann['image_id'] == img_json['id'], annotations['annotations']))
        img_json['id'] = image_id
        img_json['file_name'] = path
        final_annotations['images'].append(img_json)

        for ann in image_anns:
            ann['id'] = annotation_id
            ann['image_id'] = image_id
            if not no_damage and 'sector_damage' not in ann:
                # Assume no damage for original (presumably real) images
                ann['damage'] = 0.0
                ann['damage_type'] = "no_damage"
                ann['sector_damage'] = [0.0 for i in range(NUM_DMG_SECTORS)]
            final_annotations['annotations'].append(ann)
            annotation_id += 1
        image_id += 1


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    aug_factor = args.augmentation
    orig_dataset = args.orig_dataset.rstrip('/')
    augment_dataset = args.augment_dataset.rstrip('/')
    args.datasets_dir = os.path.abspath(args.datasets_dir)

    if args.no_damage is not True:
        # Make sure that the value isn't None
        args.no_damage = False
    
    if not os.path.exists(args.datasets_dir):
        os.mkdir(args.datasets_dir)

    identifier = 'extended' if args.extend else 'augmented'
    outdir = os.path.join(args.datasets_dir, f'{aug_factor}-synth_{identifier}')
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    with open(args.orig_annotations, 'r') as f:
        orig_anns = json.load(f)
    with open(args.augment_annotations, 'r') as f:
        augment_anns = json.load(f)
    classes = [cat['name'] for cat in orig_anns['categories'][1:]]
    final_anns = initialise_coco_anns(classes)

    # Retrieve the full sets of data
    orig_paths = glob(orig_dataset + '/**/*.jpg', recursive=True)
    num_train = len(orig_paths)
    augment_paths = glob(augment_dataset + '/**/*.jpg', recursive=True)
    def check_duplicates(check_dir_path, augment_paths):
        if check_dir_path is not None:
            if os.path.exists (check_dir_path):
                print('Checking for cross-dataset duplicates...')
                def get_fn(path):  # Can't compare full paths for cross-dataset files
                    return ntpath.basename(path)
                fn_pairs = {ntpath.basename(p): p for p in augment_paths}
                check_paths = glob(check_dir_path + '/**/*.jpg', recursive=True)
                augment_paths = list(
                    # Map returns iterable and duplicate subtraction requires sets
                    set(list(map(get_fn, augment_paths))) - set(list(map(get_fn, check_paths)))
                )
                augment_paths = [fn_pairs[p] for p in augment_paths]  # Retrieve full non-duplicate paths
            else:
                print(f'Check directory {check_dir_path} does not exist. Continuing without checking.')
        return augment_paths
    augment_paths = check_duplicates(args.check_dir, augment_paths)
    augment_paths = check_duplicates(args.check_dir_2, augment_paths)  # TODO: The check_dir argument should be a list that is iterated through

    # Randomly sample the full datasets to create a new dataset
    print('Sampling datasets...')
    if args.extend:
        orig_paths = random.sample(orig_paths, num_train)
        augment_paths = random.sample(augment_paths, int(num_train / (1 - aug_factor) - num_train))
    else:
        orig_paths = random.sample(orig_paths, int((1 - aug_factor) * num_train))
        augment_paths = random.sample(augment_paths, int(num_train * aug_factor))
    final_paths = orig_paths + augment_paths

    print('Generating annotations...')
    extend_annotations(final_anns, orig_paths, orig_anns, args.no_damage)
    extend_annotations(final_anns, augment_paths, augment_anns, args.no_damage)

    outpath = os.path.join(outdir, '_annotations.coco.json')
    with open(outpath, 'w') as f:
        json.dump(final_anns, f, indent=4)

    for i, p in enumerate(final_paths):
        print(f"Copying files: {float(i) / float(len(final_paths)):06.2%}", end='\r')
        shutil.copyfile(p, os.path.join(outdir, os.path.basename(p)))
    print(f"Copying files: 100.0%\r\n")
    
    convert_to_single_label(outdir, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=(not args.no_damage))
