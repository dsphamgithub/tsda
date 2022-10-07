# Standard EfficientDet

The 1st box for each experiment are the training script parameters taken from the `train_pawsey.sh` scripts. `HPARAMS` modifies the default model configuration stored in `efficientdet/hparams_config.py`. The `#SBATCH` parameters relate to training on the Pawsey Supercomputer. `gpu:2` means use 2 GPUs.

The 2nd box for each experiment are AP results from standard sign detection evaluation using COCO metrics.

The experiments are divided by their augment level: the percentage of the dataset that is semi-synthetic images.

## 0% Synthetic (GTSDB) | 600 in Train
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="gtsdb"
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="efficientdet-d0-1class-NEW"
NUM_EPOCHS="1000"
BATCH_SIZE="8"
HPARAMS="num_classes=1,mixed_precision=True,anchor_scale=1.5,max_level=7"
```
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.729
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.948
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.544
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.815
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.852
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.804
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.816
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.695
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.877
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.933
efficientdet-d0 {'AP': 0.7291798540185976, 'AP50': 0.9479080820499678, 'AP75': 0.8713457624152117, 'APs': 0.5444199905089661, 'APm': 0.8151063711335592, 'APl': 0.8524763190604776, 'ARmax1': 0.5218836565096953, 'ARmax10': 0.8036011080332409, 'ARmax100': 0.8160664819944599, 'ARs': 0.6952, 'ARm': 0.8767932489451477, 'ARl': 0.9333333333333333}
```

## 10% Synthetic | 666 Images in Train
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="0.1-synth_extended"
USE_VALIDATION="true"  # Whether or not to use the EVAL_TYPE data to do validation while training
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="efficientdet-d0-1class-batch-32-lr-004"
NUM_EPOCHS="1000"
BATCH_SIZE="32"
HPARAMS="num_classes=1,learning_rate=0.004,lr_warmup_init=0.0004,mixed_precision=True,anchor_scale=1.5,max_level=7"
```
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.786
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.924
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.641
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.846
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.925
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.845
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.851
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.746
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.901
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.956
efficientdet-d0 {'AP': 0.7855112704530984, 'AP50': 0.9786675926725145, 'AP75': 0.9237304747051335, 'APs': 0.6405345949189749, 'APm': 0.8463955592701021, 'APl': 0.9253025302530253, 'ARmax1': 0.551246537396122, 'ARmax10': 0.8451523545706371, 'ARmax100': 0.8506925207756233, 'ARs': 0.7464, 'ARm': 0.9012658227848102, 'ARl': 0.9555555555555555}
```

## 25% Synthetic | 800 in Train
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="0.25-synth_extended"
USE_VALIDATION="true"  # Whether or not to use the EVAL_TYPE data to do validation while training
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="efficientdet-d0-1class-batch-32-lr-004"
NUM_EPOCHS="1000"
BATCH_SIZE="32"
HPARAMS="num_classes=1,learning_rate=0.004,lr_warmup_init=0.0004,mixed_precision=True,anchor_scale=1.5,max_level=7"
```
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.788
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.972
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.920
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.640
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.849
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.921
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.548
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.846
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.855
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.761
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.901
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.956
efficientdet-d0 {'AP': 0.7875313617125949, 'AP50': 0.9718000085989684, 'AP75': 0.9199575527566145, 'APs': 0.639963139950258, 'APm': 0.8488696582252689, 'APl': 0.9206365828890581, 'ARmax1': 0.5476454293628809, 'ARmax10': 0.8459833795013851, 'ARmax100': 0.8545706371191135, 'ARs': 0.7608, 'ARm': 0.9008438818565402, 'ARl': 0.9555555555555555}
```

## 50% Synthetic | 1,200 in Train
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="0.5-synth_extended"
USE_VALIDATION="true"  # Whether or not to use the EVAL_TYPE data to do validation while training
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="efficientdet-d0-1class-batch-32-lr-004"
NUM_EPOCHS="1000"
BATCH_SIZE="32"
HPARAMS="num_classes=1,learning_rate=0.004,lr_warmup_init=0.0004,mixed_precision=True,anchor_scale=1.5,max_level=7"
```
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.752
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.954
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.893
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.569
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.829
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.914
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.827
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.837
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.733
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.886
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.967
efficientdet-d0 {'AP': 0.7523633337951565, 'AP50': 0.9541078420443585, 'AP75': 0.8934964447098916, 'APs': 0.5693871430861711, 'APm': 0.8287332464110869, 'APl': 0.9141103584042615, 'ARmax1': 0.5265927977839335, 'ARmax10': 0.8274238227146815, 'ARmax100': 0.8368421052631578, 'ARs': 0.7328, 'ARm': 0.8864978902953586, 'ARl': 0.9666666666666666}
```

## 75% Synthetic | 2,400 in Train
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
DATASET="NO_DMG_gtsdb_0.25_extended_real_only_test"
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="efficientdet-d0-1class-NEW"
NUM_EPOCHS="1000"
BATCH_SIZE="8"
HPARAMS="num_classes=1,mixed_precision=True,anchor_scale=1.5,max_level=7"
```
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.742
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.955
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.881
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.568
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.818
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.903
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.520
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.807
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.818
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.702
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.875
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.944
efficientdet-d0 {'AP': 0.7421126071986901, 'AP50': 0.9551713602575981, 'AP75': 0.8805384286179007, 'APs': 0.5683731391399269, 'APm': 0.8178348779263643, 'APl': 0.9031628162816282, 'ARmax1': 0.5199445983379503, 'ARmax10': 0.8074792243767315, 'ARmax100': 0.8177285318559555, 'ARs': 0.7024, 'ARm': 0.8746835443037975, 'ARl': 0.9444444444444444}
```

## Damage Experiment

Not to be confused with the TSDA experiments, these are the raw results from the evaluation metrics divided into damage levels used to create Fig. 6 in the paper:
```
   Damage      AP50       mAP  Mean IOU  Mean Score
0     0.0  0.694586  0.505171  0.863640    0.885661
1     0.1  0.914721  0.609387  0.468661    0.409692
2     0.2  0.916810  0.582035  0.504416    0.426855
3     0.3  0.941599  0.637066  0.642997    0.557802
4     0.4  0.888229  0.599342  0.508856    0.416304
5     0.5  0.857335  0.561501  0.223621    0.169634
6     0.6  0.734590  0.469880  0.084211    0.057769
7     0.7  0.604504  0.356104  0.060172    0.042970
8     0.8  0.833333  0.585417  0.622631    0.706073
```
Summary of overall performance on the `12000_synth_test` dataset of each model (represented by its augment level) for reference:
```
0.1 : AP=0.6440, AP50=0.9350
0.25: AP=0.7465, AP50=0.9740
0.5 : AP=0.8152, AP50=0.9857
0.75: AP=0.8546, AP50=0.9898
```