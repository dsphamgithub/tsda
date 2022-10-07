import argparse
import os
import json
import shutil
import argparse
import random
from datetime import datetime
from glob import glob

from coco_utils import convert_to_single_label

parser = argparse.ArgumentParser()
parser.add_argument('--split_path', default='/group/director2191/krados/datasets/mtsd/mtsd_v2_fully_annotated/splits/train.txt')
parser.add_argument('--annotations_dir', default='/group/director2191/krados/datasets/mtsd/mtsd_v2_fully_annotated/annotations')


def initialise_coco_anns(datetime):
    labels = {}
    labels['info'] = {
        'year': "2022",
        'version': "1",
        'description': "The Mapillary Traffic Sign Dataset converted to the COCO format",
        'contributor': "Mapillary and Curtin University",
        'date_created': datetime,
    }
    labels['licenses'] = [{
        'id': 1,
        'url': "https://opensource.org/licenses/MIT",
        'name': "MIT License"
    }]
    labels['categories'] = [
        {
            'id': 0,
            'name': "signs",
            'supercategory': "none"
        }
    ]
    labels['images'] = []
    labels['annotations'] = []
    return labels


if __name__ == '__main__':
    args = parser.parse_args()
    classes = {}

    with open(args.split_path, 'r') as f:
        image_ids = f.read().splitlines()

    datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    anns = initialise_coco_anns(datetime)

    # Build output directory string for json dump later
    mtsd_dir = os.path.split(os.path.split(args.annotations_dir)[0])[0]
    split = args.split_path.split('/')[-1].split('.')[0]
    split = "valid" if split == "val" else split
    split_img_dir = os.path.join(mtsd_dir, split)
    print(split_img_dir)
    
    count = 1
    img_id = 0
    det_id = 0
    cls_id = 1
    files = glob(args.annotations_dir + '/*.json')
    for img_ann_file in files:
        print(f"Checking annotated image {count}/{len(files)}", end='\r')
        img_filename = img_ann_file.split('/')[-1]
        if img_filename.split('.')[0] in image_ids:
            with open(img_ann_file, 'r') as f:
                img_anns = json.load(f)

                # Process image
                new_img = {
                    'id': img_id,
                    'license': 1,
                    'file_name': f"{img_filename.split('.')[0]}.jpg",
                    'height': img_anns["height"],
                    'width':  img_anns["width"],
                    'date_captured': datetime,
                    'ispano': img_anns["ispano"]
                }
                anns['images'].append(new_img)

                for det in img_anns['objects']:
                    # Process any new classes (assumption: each predefined datset split contains >0 samples for each class)
                    label = det['label']
                    if not label in classes:
                        classes[label] = cls_id  # Used to retrieve IDs from labels later for detections
                        new_cls = {
                            'id': cls_id,
                            'name': label,
                            'supercategory': "signs"
                        }
                        anns['categories'].append(new_cls)
                        cls_id += 1

                    # Process the image's detections/objects
                    width = det['bbox']['xmax'] - det['bbox']['xmin']
                    height = det['bbox']['ymax'] - det['bbox']['ymin']
                    new_det = {
                        'id': det_id,
                        'image_id': img_id,
                        'category_id': classes[det['label']],
                        'bbox': [  # COCO coordinate origin is in upper left corner
                            det['bbox']['xmin'],
                            det['bbox']['ymax'],
                            width,
                            height
                        ],
                        'area': width * height,
                        'segmentation': [],
                        'iscrowd': 0,
                        'mtsd_key': det['key'],
                        'properties': det['properties']
                    }
                    anns['annotations'].append(new_det)
                    det_id += 1
                img_id += 1
        count += 1

    print("\nSaving annotations...")
    with open(os.path.join(split_img_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(anns, f, indent=4)

    convert_to_single_label(split_img_dir, '_annotations.coco.json', '_single_annotations.coco.json')

    # TODO: Test display images from final format using modified script from MTSD GitHub