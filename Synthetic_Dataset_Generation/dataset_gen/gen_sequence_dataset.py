"""Module to generate a dataset consisting of sequences. Each sequence is a set of images where a foreground (sign) is
interpolated across a background to mimic changing distance.
"""

import argparse
from ast import parse
import os
import sys
import glob
import json
import shutil
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from sequence_gen.create_sequences_auto import produce_anchors, get_world_coords
from utils import initialise_coco_anns, convert_to_single_label, write_label_coco

parser = argparse.ArgumentParser()

parser.add_argument("--min_dist", type=int, help="startpoint distance of sequence", default=4)
parser.add_argument("--max_dist", type=int, help="endpoint distance of sequence", default=20)
parser.add_argument("--max_fg_height", type=int, default=0.15, help="maximum sign height proportional to window height")
parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
parser.add_argument("-o", "--out_dir", type=str, help="path to output directory of sequences", default='./SGTS_Sequences')
parser.add_argument("--augment", type=str, help="augmentation type", choices=['manipulated', 'transformed', 'none'], default='manipulated')

args = parser.parse_args()
args.out_dir = os.path.abspath(args.out_dir)

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from signbreaker.utils import *

os.chdir(current_dir)
bg_dir = os.path.join('..', 'signbreaker', 'Backgrounds', 'GTSDB')
manipulated_sign_dir = os.path.join('..', 'signbreaker', 'Sign_Templates', '5_Manipulated', 'GTSDB')
transformed_sign_dir = os.path.join('..', 'signbreaker', 'Sign_Templates', '4_Transformed')
damaged_sign_dir     = os.path.join('..', 'signbreaker', 'Sign_Templates', '3_Damaged')
original_sign_dir    = os.path.join('..', 'signbreaker', 'Sign_Templates', '2_Processed')

# TODO: Having fixed coordinates is an issue, especially for light sourced bent signs which are assigned a specific
#       coordinate that is ignored here
COORDS = {'x':0.75, 'y':0.45}  # Default proportional (0-1) sign coordinates in 2D plane  # TODO: Where's the origin?


def create_sequence(bg_img, fg_img, bg_name, fg_name, sequence_id):
    # Randomly place initial sign in bottom third
    
    fg_img = remove_padding(fg_img)
    bg_height, bg_width, _ = bg_img.shape
    fg_height, fg_width, _ = fg_img.shape

    sign_aspect = fg_width / fg_height
    y_size = args.max_fg_height
    x_size = sign_aspect * args.max_fg_height
    
    x_world, y_world, x_wsize, y_wsize = get_world_coords(bg_width / bg_height, 
                    COORDS['x'], COORDS['y'], args.min_dist, (x_size, y_size))
    
    anchors = produce_anchors(bg_img.shape, x_world, y_world, (x_wsize, y_wsize),
                                args.min_dist, args.max_dist, args.num_frames)
    bounding_boxes = []
    image_paths = []
    
    for frame, anchor in enumerate(anchors):
        save_path = os.path.join(args.out_dir, f"{sequence_id + frame}-{bg_name}-{fg_name}-{frame}.jpg")
        scaled_fg_img = cv2.resize(fg_img, (anchor.width, anchor.height), interpolation=cv2.INTER_AREA)
        new_img = overlay(scaled_fg_img, bg_img, anchor.screen_x, anchor.screen_y)
        cv2.imwrite(save_path, new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        bounding_boxes.append([anchor.screen_x, anchor.screen_y, anchor.width, anchor.height])
        image_paths.append(save_path)
    return (image_paths, bounding_boxes)


def get_damage(path):
    data_file_path = os.path.join(damaged_sign_dir, "damaged_data.npy")
    if os.path.exists(data_file_path):
        damaged_data = np.load(data_file_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"Error: Damaged data file does not exist - cannot recover damage label data.\n")

    dmg_synth = next((x for x in damaged_data if x.fg_path == path), None)
    return dmg_synth


def generate_labels(image_id, bg_img, bg_name, bg_dims, annotations, augment='transformed'):
    if augment == 'manipulated':
        sign_dirs = glob.glob(os.path.join(manipulated_sign_dir, f'BG_{bg_name}') + '/*/*/')
    elif augment == 'transformed':
        sign_dirs = glob.glob(transformed_sign_dir + '/*/*/')
    elif augment == 'none':
        sign_dirs = glob.glob(damaged_sign_dir + '/*/')
    sign_dirs = sorted(sign_dirs, key=lambda p: int(Path(p).stem.split('_')[0]))

    if sign_dirs == []:
        raise FileNotFoundError("Error: could not find any data for the chosen augment method.\n")
    for sign_dir in tqdm(sign_dirs):
        if glob.glob(sign_dir + '/*.png') != []:
            sign_path = random.choice(glob.glob(sign_dir + '/*.png'))
        else:
            continue
        
        if augment == 'manipulated' or augment == 'transformed':
            sign_name = Path(sign_path).parts[-2]
        elif augment == 'none':
            sign_name = Path(sign_path).stem
        cls_name = sign_name.split('_')[0]
        
        damaged_sign_path = os.path.join(damaged_sign_dir, cls_name, sign_name + '.png')
        
        sign_img = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)
        image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id)
        
        for i in range(len(image_paths)):
            # Format [x, y, width, height, distance]
            path = os.path.join(*os.path.normpath(damaged_sign_path).split(os.path.sep)[-4:])
            dmg_synth = get_damage(path)
            file_path = os.path.basename(image_paths[i])
            write_label_coco(
                annotations, file_path, image_id, int(cls_name), bounding_boxes[i], bg_dims,
                dmg_synth.damage_ratio, dmg_synth.damage_type, dmg_synth.sector_damage
            )
            image_id += 1
    return image_id


if __name__ == '__main__':
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)
    
    classes = sorted([Path(p).stem for p in os.listdir(original_sign_dir)], key=lambda p: int(p))  
    annotations = initialise_coco_anns(classes)
    bg_paths = glob.glob(bg_dir + "/*.png")
    image_id = 0
    
    for bg_id, bg_path in enumerate(bg_paths):
        bg_name = Path(bg_path).stem
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        height, width, _ = bg_img.shape
        print(f"Generating sequences for background image {bg_name}")
        image_id = generate_labels(image_id, bg_img, bg_name, (height, width), annotations, args.augment)
                
    annotations_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    convert_to_single_label(args.out_dir, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=True)
