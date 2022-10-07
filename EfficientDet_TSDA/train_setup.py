import os
import argparse
from pathlib import Path
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging


parser = argparse.ArgumentParser()

"""Define the flags."""
parser.add_argument('--models_dir', default='/home/krados/model_dir', help='Directory containing checkpoints')
parser.add_argument('--model', default='efficientdet-d0', help='Architecture to use')
parser.add_argument('--experiment', default='0.25-gtsdb-1class-d0', help='Name of this model\'s checkpoints directory')
parser.add_argument('--dataset', default='0.25_augmented_gtsdb', help='Which dataset to use')
    
    
args = parser.parse_args()
save_dir = os.path.join(args.models_dir, args.dataset)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


def download_ckpt(save_dir, m):
    '''Downloads Google's checkpoint for the specified EfficientDet model.'''
    ckpt_path = save_dir + f'/{m}_cocoCkpt'
    if not os.path.isdir(ckpt_path):
        os.system(f'wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/{m}.tar.gz')
        os.system(f'tar zxf {m}.tar.gz --directory {save_dir}')
        os.system(f'mv {save_dir}/{m} {ckpt_path}')
        os.system(f'rm {m}.tar.gz')
    return ckpt_path


def get_num_trained_epochs(ckpt_dir):
    checkpoint_file = os.path.join(ckpt_dir, 'checkpoint')
    with open(checkpoint_file, 'r') as f:
        lines = f.readlines()
        last_checkpoint = lines[0].split(':')[1].strip().strip('"')
        trained_epochs = int(last_checkpoint.split('-')[-1])
        return trained_epochs
    

def get_latest_pretrained_checkpoint(model_dir, coco_ckpt_dir):
    checkpoint_list = glob(f"{model_dir}*")
    checkpoint_list = list(filter(lambda d: "checkpoint" in os.listdir(d), checkpoint_list))
    if checkpoint_list:
        # Return the newest checkpoint filename
        return sorted(checkpoint_list, key=os.path.getmtime)[-1]
    else:
        return coco_ckpt_dir


def main():
    # Download pretrained COCO model
    coco_ckpt_dir = download_ckpt(args.models_dir, args.model)

    total_epochs = 0
    model_dir = f'{save_dir}/{args.experiment}'
    ckpt_dir = get_latest_pretrained_checkpoint(model_dir, coco_ckpt_dir)

    if ckpt_dir == None:
        print(""); return
    
    if "cocoCkpt" in Path(ckpt_dir).stem:
        print(coco_ckpt_dir)
    
    # Continue training from previous checkpoint
    elif ckpt_dir == model_dir:
        # Extract number of epochs
        if glob(f'{ckpt_dir}-*'):
            prev_ckpt = sorted(glob(f'{ckpt_dir}-*'), key=os.path.getmtime)[-1]
            prev_ckpt_name = Path(prev_ckpt).stem
            total_epochs += int(prev_ckpt_name.split('-')[-1])
        
        # Load weights from latest checkpoint and include number of epochs in file name
        total_epochs += get_num_trained_epochs(ckpt_dir)
        new_path = ckpt_dir + '-' + str(total_epochs) 
        os.system(f'mv {ckpt_dir} {new_path}')
        print(new_path)
        
    else:
        print(ckpt_dir)


if __name__ == '__main__':
    main()
