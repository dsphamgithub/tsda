#!/bin/bash

#SBATCH --job-name=edet-tsda-0.5
#SBATCH --nodes=3
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=1:00:00

module load python
module load singularity

DATASET="0.5-synth_extended"
SINGLE_DIR=false  # true or false; does the dataset have train/val/test subdirs, or just images and annotations
EVAL_TYPE="test"  # "test" or "valid"
USE_VALIDATION="false"  # Whether or not to use the EVAL_TYPE data to do validation while training
MODEL_NAME="efficientdet-d0"
# FIXME/NOTE: pre-generated tfrecords for dataset MUST BE DELETED when changing DMG_SECTORS
# FIXME: Only "4" works, but should also work with "1"
DMG_SECTORS="4"  # Setting to 0 will mean no damage, which currently will break things
EXPERIMENT_NAME="tsda-m${DMG_SECTORS}-${MODEL_NAME}-lr-003"  # Corresponds to model checkpoints directory name
NUM_EPOCHS="1000"  # If resuming, this will be on top of the original epochs
BATCH_SIZE="24"  # Auto scales with no. nodes to some degree, but e.g. "24" with 1 node may OOM while it won't with 3
# Default learning_rate is 0.008
HPARAMS="num_classes=1,learning_rate=0.003,lr_warmup_init=0.0003,mixed_precision=False,anchor_scale=1.5,num_damage_sectors=${DMG_SECTORS},max_level=7,damage_net=True"

# FIXME: Model doesn't work with num_damage_sectors=1, only works with 4

# # This is secret and shouldn't be checked into version control
# WANDB_API_KEY="a1fc71891427e8f94d60bad17f02c98f26b8ee67"
# # Name and notes optional
# WANDB_NAME="First test run"
# WANDB_NOTES="${DATASET}, ${MODEL_NAME}, ${NUM_EPOCHS} epochs, config.as_dict()."

SAVE_DIR="${MYGROUP}/model_dir"

if [ ${SINGLE_DIR} = false ]; then
    TRAIN_DIR="${MYGROUP}/datasets/${DATASET}/train"
else
    TRAIN_DIR="${MYGROUP}/datasets/${DATASET}"
fi

VALID_DIR="${MYGROUP}/datasets/${DATASET}/${EVAL_TYPE}"

NUM_TRAIN=$( find ${TRAIN_DIR} -name "*.jpg" | wc -l )
NUM_VALID=$( find ${VALID_DIR} -name "*.jpg" | wc -l )


#-----------------Building arguments for train_setup.py-------------------------

PREPROCESS_CMD="python3 train_setup.py \
                --models_dir=${SAVE_DIR} \
                --model=${MODEL_NAME} \
                --experiment=${EXPERIMENT_NAME} \
                --dataset=${DATASET}"

# Get checkpoint path from setup script
CHECKPOINT_PTH=$( cd $(${PREPROCESS_CMD}); pwd )
CHECKPOINT_PTH=$( sed s+${MYGROUP}+/temp+g <<< ${CHECKPOINT_PTH} )


#-----------------Building arguments for hvd_train.py---------------------------

if [ $DATASET == "gtsdb" ] || [ $DATASET == "mtsd" ]; then  
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/train-*"
else
    TRAIN_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/*"
fi

VALID_FILE_PATTERN="/temp/datasets/tfrecords/${DATASET}/${EVAL_TYPE}-*"
MODEL_SUBDIR="model_dir/${DATASET}/${EXPERIMENT_NAME}"
MODEL_DIR="/temp/${MODEL_SUBDIR}"

# Any directory with this name should have been renamed by train_setup.py if we are resuming from it's checkpoints
if [ -d "${MYGROUP}/${MODEL_SUBDIR}" ]; then
    printf "\nA model directory with this name already exists but does not have valid checkpoints to resume from\n\n"
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

    echo
    echo "Train command:"
    echo $TRAIN_CMD
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
    export TF_SINGLE_DIR=${SINGLE_DIR}
    export TF_DMG_SECTORS=${DMG_SECTORS}
    sh pawsey_tfrecords.sh
fi

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
