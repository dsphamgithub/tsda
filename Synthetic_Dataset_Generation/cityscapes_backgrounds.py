"""Script to create backgrounds for the synthetic dataset generator from the
Cityscapes dataset. It filters out and saves Cityscapes images that have an
instance of a front-facing traffic sign.
"""

def main():
    import os
    import argparse
    from glob import glob
    import json
    import shutil

    parser = argparse.ArgumentParser(description='Filter Cityscapes images to have no front-facing traffic signs.')
    parser.add_argument('--gt', type=str, help='Complete path to directory containing Cityscapes ground truth',
                        default='C:\\Users\\krist\\Downloads\\gtFine_trainvaltest')
    args = parser.parse_args()

    def save_images(mode='train'):
        gt_dir = os.path.join(args.gt, 'gtFine', mode)
        gt_jsons = glob(f"{gt_dir}{os.sep}**{os.sep}*.json", recursive=True)
        signs = 0
        saved = 0
        for gt_json_fn in gt_jsons:  # Iterate through each annotation file
            # Check image-wise json annotations for traffic signs
            with open(gt_json_fn, 'r') as f:
                gt_json = json.load(f)
            traffic_sign = False
            for o in gt_json['objects']:
                if o['label'] == 'traffic sign':
                    signs += 1
                    traffic_sign = True
                    break
            if not traffic_sign:
                img_fn = gt_json_fn.replace('gtFine_trainvaltest', 'leftImg8bit_trainvaltest').replace(
                    'gtFine', 'leftImg8bit').replace('_polygons', '').replace('json', 'png')
                if os.path.exists(img_fn):
                    # Save image to new directory
                    img_fn_new = img_fn.replace(f'{os.sep}{mode}', f'{os.sep}backgrounds')
                    os.makedirs(os.path.dirname(img_fn_new), exist_ok=True)
                    shutil.copyfile(img_fn, img_fn_new)
                    print(f"Saved {img_fn_new}")
                    saved += 1
                else:
                    print(f"{img_fn} does not exist")
        print(f"Skipped {signs} {mode} images with traffic signs")
        print(f"Saved {saved} {mode} images without traffic signs")

    save_images('train')
    save_images('val')

if __name__ == "__main__":
    main()
