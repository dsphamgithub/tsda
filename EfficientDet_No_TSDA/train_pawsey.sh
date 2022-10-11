#!/bin/bash

#SBATCH --job-name=edet-0.1
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq
#SBATCH --account=director2191
#SBATCH --time=3:00:00

module load python
module load singularity

# FIXME: PRE_DATASET not fully implemented
PRE_DATASET="none"   # If finetuning, this is the dataset the model was pretrained on; otherwise, use "none"
DATASET="0.1-synth_extended"
# FIXME: TRAIN_ONLY should be SINGLE_DIR
TRAIN_ONLY="false"  # "true" or "false"
USE_VALIDATION="true"  # Whether or not to use the EVAL_TYPE data to do validation while training
EVAL_TYPE="test"  # "test" or "valid"
MODEL_NAME="efficientdet-d0"
EXPERIMENT_NAME="${MODEL_NAME}-TF-FIX-TEST"  # Corresponds to model checkpoints directory name
NUM_EPOCHS="1000"  # If resuming, this will be on top of the original epochs
BATCH_SIZE="32"  # Should automatically scale with number of nodes
HPARAMS="num_classes=1,learning_rate=0.004,lr_warmup_init=0.0004,mixed_precision=True,anchor_scale=1.5,max_level=7"
# TODO: Automatic reduction of base learning rate based on BS from assumed baseline BS of 64

# # This is secret and shouldn't be checked into version control
# WANDB_API_KEY="a1fc71891427e8f94d60bad17f02c98f26b8ee67"
# # Name and notes optional
# WANDB_NAME="First test run"
# WANDB_NOTES="${DATASET}, ${MODEL_NAME}, ${NUM_EPOCHS} epochs, config.as_dict()."

SAVE_DIR="${MYGROUP}/model_dir"

if [ $TRAIN_ONLY == "false" ]; then
    TRAIN_DIR="${MYGROUP}/datasets/${DATASET}/train"
else
    TRAIN_DIR="${MYGROUP}/datasets/${DATASET}"
fi

VALID_DIR="${MYGROUP}/datasets/${DATASET}/${EVAL_TYPE}"

NUM_TRAIN=$( find ${TRAIN_DIR} -name "*.jpg" | wc -l )
NUM_VALID=$( find ${VALID_DIR} -name "*.jpg" | wc -l )


#-----------------Building arguments for train_setup.py-------------------------

if [ $PRE_DATASET == "none" ]; then
    PREPROCESS_CMD="python3 train_setup.py \
        --models_dir=${SAVE_DIR} \
        --model=${MODEL_NAME} \
        --experiment=${EXPERIMENT_NAME} \
        --dataset=${DATASET}"
else
    PREPROCESS_CMD="python3 train_setup.py \
        --models_dir=${SAVE_DIR} \
        --model=${MODEL_NAME} \
        --experiment=${EXPERIMENT_NAME} \
        --dataset=${PRE_DATASET}"
fi

# Get checkpoint path from setup script
CHECKPOINT_PTH=$( cd $(${PREPROCESS_CMD}); pwd )
CHECKPOINT_PTH=$( sed s+${MYGROUP}+/temp+g <<< ${CHECKPOINT_PTH} )


#-----------------Building arguments for hvd_train.py---------------------------

if [ $TRAIN_ONLY == "false" ]; then  
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/train-*"
else
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/*"
fi
# if [ $DATASET == "gtsdb" ] || [ $DATASET == "mtsd" ]; then  
#     TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/train-*"
# else
#     TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/*"
# fi

VALID_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/${EVAL_TYPE}-*"
if [ $PRE_DATASET == "none" ]; then
    MODEL_SUBDIR="model_dir/${DATASET}/${EXPERIMENT_NAME}"
else
    MODEL_SUBDIR="model_dir/${PRE_DATASET}/${EXPERIMENT_NAME}"
fi
MODEL_DIR="/temp/${MODEL_SUBDIR}"

# Any directory with this name should have been renamed by train_setup.py if we are resuming from it's checkpoints
if [ -d "${MYGROUP}/${MODEL_SUBDIR}" ]; then
    printf "\nA model directory with this name already exists but does not have valid checkpoints to resume from"
else
    TRAIN_CMD="python efficientdet/hvd_train.py \
        --train_file_pattern=${TRAIN_FILE_PATTERN} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --num_examples_per_epoch=${NUM_TRAIN} \
        --eval_samples=${NUM_VALID} \
        --num_epochs=${NUM_EPOCHS} \
        --batch_size=${BATCH_SIZE} \
        --hparams=${HPARAMS} \
        --pretrained_ckpt=${CHECKPOINT_PTH} \
        --tf_random_seed=111111"

    if [ $USE_VALIDATION == "true" ]; then
        TRAIN_CMD="$TRAIN_CMD --val_file_pattern=${VALID_FILE_PATTERN}"
    fi
fi

#--------------Running training command inside singularity container------------

START_TIME=$(date +%s)

IMAGE_PATH="${MYGROUP}/singularity_images/tf_container_2105_efficientdet_precise_tf.sif"
if [ -d "${MYGROUP}/datasets/tfrecords/${DATASET}" ]
then
    printf "\nFound pre-existing TFRECORD directory\n"
else
    export TF_DATASET=${DATASET}
    export TF_IMAGE_PATH=${IMAGE_PATH}
    export TF_TRAIN_ONLY=${TRAIN_ONLY}
    sh pawsey_tfrecords.sh
fi

echo
echo "Will run train command:"
echo $TRAIN_CMD

# Use semicolons to separate multi-commands in " " below

# Saving output to file (standard)
srun -o slurm_outputs/${EXPERIMENT_NAME}-%j_%t.out singularity run --nv \
     -B ${MYGROUP}:/temp ${IMAGE_PATH} \
     /bin/bash -c "${TRAIN_CMD}"

# Letting output go to terminal (for rapid testing)
# srun singularity run --nv \
#     -B ${MYGROUP}:/temp ${IMAGE_PATH} \
#     /bin/bash -c "${TRAIN_CMD}"

END_TIME=$(date +%s)

echo
echo "Time taken is equal to: $(( END_TIME - START_TIME )) seconds"
