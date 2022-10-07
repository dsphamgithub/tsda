import os
import argparse
from pathlib import Path
import shutil
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging


parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', required=True, help="Directory containing checkpoints")
parser.add_argument('--model', required=True, help="Architecture to use")
parser.add_argument('--experiment', required=True, help="Name of this model's checkpoints directory")
parser.add_argument('--dataset', required=True, help="Which dataset to use")
parser.add_argument('--export_dir', required=True, help="Directory where exported models will be saved")
parser.add_argument('--out_dir', required=True, help="Directory where predictions will be saved")
args = parser.parse_args()


def get_latest_pretrained_checkpoint(model_dir, coco_ckpt_dir):
    checkpoint_list = glob(f"{model_dir}*")
    checkpoint_list = list(filter(lambda d: "checkpoint" in os.listdir(d), checkpoint_list))
    if checkpoint_list:
        # Return the newest checkpoint filename
        return sorted(checkpoint_list, key=os.path.getmtime)[-1]
    else:
        return coco_ckpt_dir

def main():
    coco_ckpt_dir = os.path.join(args.models_dir, f"{args.model}_cocoCkpt")
    if not os.path.isdir(coco_ckpt_dir):
        raise ValueError(f"Could not find the COCO checkpoints directory {coco_ckpt_dir}")

    save_dir = args.export_dir
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    pred_dir = os.path.join(args.out_dir, args.experiment)
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    model_dir = f"{args.models_dir}/{args.dataset}/{args.experiment}"
    ckpt_dir = get_latest_pretrained_checkpoint(model_dir, coco_ckpt_dir)

    if ckpt_dir == None:
        print(""); return
    if "cocoCkpt" in Path(ckpt_dir).stem:
        print(coco_ckpt_dir)
    else:
        print(ckpt_dir)


if __name__ == "__main__":
    main()
