# TSDA EfficientDet

## Training Configuration

These are the training script parameters taken from the `train_pawsey.sh` scripts. `HPARAMS` modifies the default model configuration stored in `efficientdet/hparams_config.py`. The `#SBATCH` parameters relate to training on the Pawsey Supercomputer. `gpu:2` means use 2 GPUs.

```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="gtsdb_0.25_extended_2"
SINGLE_DIR=false  # true or false; does the dataset have train/val/test subdirs, or just images and annotations
EVAL_TYPE="test"  # "test" or "valid"
USE_VALIDATION="false"  # Whether or not to use the EVAL_TYPE data to do validation while training
MODEL_NAME="efficientdet-d0"
DMG_SECTORS="4"
EXPERIMENT_NAME="tsda-m${DMG_SECTORS}-${MODEL_NAME}-lr-003"
NUM_EPOCHS="1000"
BATCH_SIZE="16"
HPARAMS="num_classes=1,learning_rate=0.003,lr_warmup_init=0.0003,mixed_precision=False,anchor_scale=1.5,num_damage_sectors=${DMG_SECTORS},max_level=7,damage_net=True"
```

## Evaluation Results
### Standard Sign Detection
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.863
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.976
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.813
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.919
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.939
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.884
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.908
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.908
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.869
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.954
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.976
efficientdet-d0 {'AP': 0.8628005634854281, 'AP50': 0.9899716266345292, 'AP75': 0.9757823442339725, 'APs': 0.8134489801235569, 'APm': 0.919168320017602, 'APl': 0.9391599953826407, 'ARmax1': 0.8843666666666665, 'ARmax10': 0.907875, 'ARmax100': 0.9079083333333333, 'ARs': 0.8685646146966761, 'ARm': 0.9537899543378995, 'ARl': 0.9761410788381744}
```
### Damage Assessment
```
MAE: 0.037952420780690284
RMSE: 0.08702530069949199
Mean Predicted Damage: 0.19105912744998932
Mean Ground Truth Damage: 0.20559607274862296
```