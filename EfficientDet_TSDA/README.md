# TSDA EfficientDet

## Damage Network

### Training Configuration

These are the training script parameters taken from the `train_pawsey.sh` scripts. `HPARAMS` modifies the default model configuration stored in `efficientdet/hparams_config.py`. The `#SBATCH` parameters relate to training on the Pawsey Supercomputer. `gpu:2` means use 2 GPUs.

```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="0.75-synth_extended_dmg"
SINGLE_DIR=false  # true or false; does the dataset have train/val/test subdirs, or just images and annotations
EVAL_TYPE="test"  # "test" or "valid"
USE_VALIDATION="false"  # Whether or not to use the EVAL_TYPE data to do validation while training
MODEL_NAME="efficientdet-d0"
DMG_SECTORS="4"
EXPERIMENT_NAME="tsda-m${DMG_SECTORS}-${MODEL_NAME}-lr-003"
NUM_EPOCHS="775"
BATCH_SIZE="16"
HPARAMS="num_classes=1,learning_rate=0.003,lr_warmup_init=0.0003,mixed_precision=False,damage_net=True,anchor_scale=1.5,num_damage_sectors=${DMG_SECTORS},max_level=7"
```

### Evaluation Results
#### Standard Sign Detection
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
#### Damage Assessment
```
MAE: 0.037952420780690284
RMSE: 0.08702530069949199
Mean Predicted Damage: 0.19105912744998932
Mean Ground Truth Damage: 0.20559607274862296
```

## Class Network Damage Layer

### Training Configuration
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="0.75-synth_extended_dmg"
SINGLE_DIR=false  # true or false; does the dataset have train/val/test subdirs, or just images and annotations
EVAL_TYPE="test"  # "test" or "valid"
USE_VALIDATION="false"  # Whether or not to use the EVAL_TYPE data to do validation while training
MODEL_NAME="efficientdet-d0"
DMG_SECTORS="4"
EXPERIMENT_NAME="tsda-m${DMG_SECTORS}-${MODEL_NAME}-dmg_block-lr-003"
NUM_EPOCHS="775"
BATCH_SIZE="16"
HPARAMS="num_classes=1,learning_rate=0.003,lr_warmup_init=0.0003,mixed_precision=False,damage_net=False,anchor_scale=1.5,num_damage_sectors=${DMG_SECTORS},max_level=7"
```

### Evaluation Results

#### Standard Sign Detection
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.883
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.981
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.830
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.945
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.968
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.902
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.921
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.921
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.881
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.969
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.990
efficientdet-d0 {'AP': 0.8826563647911502, 'AP50': 0.9899810136155196, 'AP75': 0.9811980399467972, 'APs': 0.8295434897418371, 'APm': 0.9445596558743712, 'APl': 0.9680387022938511, 'ARmax1': 0.9018, 'ARmax10': 0.9213166666666666, 'ARmax100': 0.9213583333333333, 'ARs': 0.880876434640036, 'ARm': 0.9689041095890412, 'ARl': 0.9895228215767634}
```

#### Damage Assessment
```
MAE: 0.2060827232542021
RMSE: 0.36869899499771197
Mean Predicted Damage: 0.010000139474868774
Mean Ground Truth Damage: 0.205596072748623
```


#### Conclusion
Forcing damage to be predicted from shared class features seemed to result in *+1.99 mAP* compared to the Damage Network (i.e. a slight improvement).

However, the damage assessment performance is significantly worse with *+0.2817* RMSE compared to the Damage Network. 

Therefore using an integrated damage layer in the class network is likely not a viable approach.
