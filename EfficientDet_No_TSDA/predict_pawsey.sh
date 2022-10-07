#!/bin/bash

#SBATCH --job-name=edet-predict
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=01:00:00

module load python
module load singularity

TRAIN_DATASET="0.5-synth_extended"
EVAL_DATASET="TSDA_Manual"
MODEL_NAME="efficientdet-d0"  # Evaluate whatever ckpt dir in model_dir/${DATASET} that has ${MODEL_NAME} in it's name
EXPERIMENT_NAME="${MODEL_NAME}-1class-batch-32-lr-004"  # Corresponds to model checkpoints directory name
EVAL_TYPE="none"  # "test", "valid", or "none"
HPARAMS="num_classes=1,learning_rate=0.004,lr_warmup_init=0.0004,mixed_precision=True,anchor_scale=1.5,max_level=7"
MIN_SCORE="0.0"  # Threshold for minimum confidence score of prediction
MAX_BOXES="200"  # The max number of boxes to draw in output image

# NOTE: Remember to change EXPERIMENT_NAME and HPARAMS when switching models


#-----------------Building arguments for model saving---------------------------

MODELS_DIR="${MYGROUP}/model_dir"
EXPORT_DIR="saved_model_dir/${TRAIN_DATASET}"
OUT_DIR="${MYGROUP}/model_predictions/${TRAIN_DATASET}~${EVAL_DATASET}"

PREPROCESS_CMD="python predict_setup.py \
    --models_dir=${MODELS_DIR} \
    --model=${MODEL_NAME} \
    --experiment=${EXPERIMENT_NAME} \
    --dataset=${TRAIN_DATASET} \
    --export_dir=${MYGROUP}/${EXPORT_DIR} \
    --out_dir=${OUT_DIR}"

# Get checkpoint path from setup script
CHECKPOINT_PTH=$( cd $(${PREPROCESS_CMD}); pwd )
CHECKPOINT_PTH=$( sed s+${MYGROUP}+/temp+g <<< ${CHECKPOINT_PTH} )

EXPORT_CMD="python efficientdet/model_inspect.py \
    --runmode=saved_model \
    --model_name=${MODEL_NAME} \
    --ckpt_path=${CHECKPOINT_PTH} \
    --saved_model_dir=/temp/${EXPORT_DIR} \
    --hparams=${HPARAMS}"

echo
echo ${EXPORT_CMD}
echo
echo ${CHECKPOINT_PTH}


#-----------------Building arguments for model inference------------------------

if [ $EVAL_TYPE == "none" ]; then
    DATASET_DIR="/temp/datasets/${EVAL_DATASET}"
else
    DATASET_DIR="/temp/datasets/${EVAL_DATASET}/${EVAL_TYPE}"
fi

OUT_DIR_FULL="/temp/model_predictions/${TRAIN_DATASET}~${EVAL_DATASET}/${EXPERIMENT_NAME}"

INFER_CMD="python efficientdet/model_inspect.py \
    --runmode=saved_model_infer \
    --saved_model_dir=/temp/${EXPORT_DIR} \
    --model_name=${MODEL_NAME} \
    --input_image=${DATASET_DIR}/*.jpg \
    --output_image_dir=${OUT_DIR_FULL} \
    --min_score_thresh=${MIN_SCORE} \
    --max_boxes_to_draw=${MAX_BOXES}"

echo
echo ${INFER_CMD}


#-----------Running prediction command inside singularity container-------------

START_TIME=$(date +%s)

IMAGE_PATH="${MYGROUP}/singularity_images/tf_container_2105_efficientdet_precise_tf.sif"
srun singularity run --nv \
    -B ${MYGROUP}:/temp ${IMAGE_PATH} \
    /bin/bash -c "${EXPORT_CMD}"
srun singularity run --nv \
    -B ${MYGROUP}:/temp ${IMAGE_PATH} \
    /bin/bash -c "${INFER_CMD}"

END_TIME=$(date +%s)

echo "Time taken is equal to: $(( END_TIME - START_TIME )) seconds"
