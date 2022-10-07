#!/bin/bash

# TODO: File out of date, see train_pawsey.sh

DATASET="gtsdb"
MODEL_NAME="efficientdet-d0"
NUM_EPOCHS="2"
BATCH_SIZE="4"
HPARAMS="num_classes=1,moving_average_decay=0,mixed_precision=True,optimizer=adam,anchor_scale=1.5"

# 'test' or 'valid'
EVAL_TYPE="test"
USE_VALIDATION="false"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SAVE_DIR="$SCRIPT_DIR/../model_dir"

if [ $DATASET == "gtsdb" ]; then  
    TRAIN_DIR="$SCRIPT_DIR/../datasets/${DATASET}/train"
elif [ $DATASET == "SGTS_Dataset" ]; then
    TRAIN_DIR="$SCRIPT_DIR/../datasets/${DATASET}"
else
    TRAIN_DIR="$SCRIPT_DIR/../datasets/augmented/${DATASET}"
fi

VALID_DIR="$SCRIPT_DIR/../datasets/gtsdb/$EVAL_TYPE"

NUM_TRAIN=$( find $TRAIN_DIR -name "*.jpg" | wc -l )
NUM_VALID=$( find $VALID_DIR -name "*.jpg" | wc -l )

#------------------Building arguments for train_setup.py--------------------------

PREPROCESS_CMD="python3 train_setup.py \
                --models_dir=$SAVE_DIR --model=$MODEL_NAME \
                --dataset=$DATASET"

CHECKPOINT_PTH=$( cd $($PREPROCESS_CMD); pwd )
REPLACE_STRING="$( echo $(cd $SCRIPT_DIR/../; pwd ))"
CHECKPOINT_PTH=$( sed s+${REPLACE_STRING}+/temp+g <<< $CHECKPOINT_PTH )


#------------------Building arguments for tf2_train.py----------------------------

if [ $DATASET == "gtsdb" ]; then  
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/train*"
else
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/*"
fi

VALID_FILE_PATTERN="/temp/datasets/tfrecords/gtsdb/$EVAL_TYPE*"
MODEL_DIR="/temp/model_dir/$DATASET/$MODEL_NAME"

TFRECORD_CMD="python /temp/EfficientDet/generate_tfrecords.py \
    --dataset=/temp/datasets/$DATASET \
    --datasets_dir=/temp/datasets"

INSTALL_CMD="pip install -r /temp/EfficientDet/efficientdet/requirements.txt"

TRAIN_CMD="python /temp/EfficientDet/efficientdet/hvd_train.py \
    --train_file_pattern=${TRAIN_FILE_PATTERN} \
    --model_name=$MODEL_NAME \
    --model_dir=$MODEL_DIR \
    --num_examples_per_epoch=$NUM_TRAIN \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE \
    --hparams=$HPARAMS \
    --pretrained_ckpt=${CHECKPOINT_PTH} \
    --tf_random_seed=111111"

if [ $USE_VALIDATION == "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --val_file_pattern=${VALID_FILE_PATTERN}\
    --eval_samples=$NUM_VALID"
fi

echo $TRAIN_CMD
echo


#-----------------Running training command inside docker container-----------

START_TIME=`date +%s`

# Allen's original:
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#      --rm -v $SCRIPT_DIR/../:/temp -w /temp allenator/nvcr_tensorflow:latest $TRAIN_CMD

# Using nvidia's container:
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     --rm -v $SCRIPT_DIR/../:/temp -w /temp nvcr.io/nvidia/tensorflow:22.03-tf2-py3 \
     /bin/bash -c "$INSTALL_CMD; $TFRECORD_CMD; $TRAIN_CMD"

END_TIME=`date +%s`

echo "Time taken is equal to: $(( END_TIME - START_TIME )) seconds"
