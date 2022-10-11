import argparse
import json
from glob import glob
import os
import random
import shutil
import ntpath

from utils import initialise_coco_anns, convert_to_single_label

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str, required=True)
parser.add_argument('--orig_annotations', default=None, type=str, required=True)
parser.add_argument('--num_images', default=None, type=int, required=True)
parser.add_argument('--check_dir', default=None, type=str, help='Path to a directory which we want to check for '
                    'cross-dataset duplicates. If one of our synthetic images is there already, it will not be added to '
                    'this new dataset. This is helpful for ensuring a test set contains no training set images if '
                    'pulling from the same pool of synthetic images.')
parser.add_argument('--check_dir_2', default=None, type=str)

def extend_annotations(final_annotations, image_paths, annotations):
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
            # if not no_damage and 'sector_damage' not in ann:
            #     # Assume no damage for original (presumably real) images
            #     ann['damage'] = 0.0
            #     ann['damage_type'] = "no_damage"
            #     ann['sector_damage'] = [0.0 for i in range(NUM_DMG_SECTORS)]
            final_annotations['annotations'].append(ann)
            annotation_id += 1
        image_id += 1

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset.rstrip('/')

    with open(args.orig_annotations, 'r') as f:
        orig_anns = json.load(f)
    classes = [cat['name'] for cat in orig_anns['categories'][1:]]
    final_anns = initialise_coco_anns(classes)

    outdir = os.path.join(os.getcwd(), "shortened")
    if os.path.exists(outdir):
        raise ValueError('Output directory already exists')
    else:
        os.mkdir(outdir)

    def subtract_paths(paths_a, paths_b):
        """Subtract the set of paths_b from paths_a."""
        def get_fn(path):  # Can't compare full paths for cross-dataset files
            return ntpath.basename(path)
        fn_pairs = {ntpath.basename(p): p for p in paths_a}
        paths_a = list(
            # Map returns iterable and duplicate subtraction requires sets
            set(list(map(get_fn, paths_a))) - set(list(map(get_fn, paths_b)))
        )
        paths_a = [fn_pairs[p] for p in paths_a]  # Retrieve full non-duplicate paths
        return paths_a

    def check_duplicates(check_dir_path, data_paths):
        if check_dir_path is not None:
            if os.path.exists (check_dir_path):
                # print('Checking for cross-dataset duplicates...')
                check_paths = glob(check_dir_path + '/**/*.jpg', recursive=True)
                data_paths = subtract_paths(data_paths, check_paths)
            else:
                print(f'Check directory {check_dir_path} does not exist. Continuing without checking.')
        return data_paths
    
    print('Sampling datasets...')
    dataset_paths_draw_pool = glob(dataset + '/**/*.jpg', recursive=True)
    dataset_paths = []
    while len(dataset_paths) < args.num_images:
        # Iterate to reach precise number of images
        new_paths = random.sample(dataset_paths_draw_pool, args.num_images - len(dataset_paths))
        new_paths = check_duplicates(args.check_dir, new_paths)
        new_paths = check_duplicates(args.check_dir_2, new_paths)  # TODO: The check_dir argument should be a list that is iterated through
        dataset_paths_draw_pool = subtract_paths(dataset_paths_draw_pool, new_paths)  # Remove drawn images from draw pool
        dataset_paths.extend(new_paths)
        print(f'Duplicate-free paths found: {len(dataset_paths)}')
    assert len(dataset_paths) == args.num_images, "Number of images sampled does not match number of images requested"

    print('Generating annotations...')
    extend_annotations(final_anns, dataset_paths, orig_anns)

    outpath = os.path.join(outdir, '_annotations.coco.json')
    with open(outpath, 'w') as f:
        json.dump(final_anns, f, indent=4)

    for i, p in enumerate(dataset_paths):
        print(f"Copying files: {float(i) / float(len(dataset_paths)):06.2%}", end='\r')
        shutil.copyfile(p, os.path.join(outdir, os.path.basename(p)))
    print(f"Copying files: 100.0%\r\n")
    
    convert_to_single_label(outdir, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=True)
