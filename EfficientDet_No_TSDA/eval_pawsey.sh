#!/bin/bash

#SBATCH --job-name=edet-eval
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=00:40:00

module load python
module load singularity

TRAIN_DATASET="0.25-synth_extended"
EVAL_DATASET="12000_synth_test"
MODEL_NAME="efficientdet-d0"  # Evaluate whatever ckpt dir in model_dir/${DATASET} that has ${MODEL_NAME} in it's name
EXPERIMENT="none"  # "damage", "distance", "sequence", or "none"
# TODO: ^ Clarify damage experiment isn't TSDA, and add "none" option
BATCH_SIZE="32"  # Each gpuq-dev GPU can handle at least 8 MTSD images for eval
EVAL_TYPE="none"  # "test", "valid", or "none"
ANNOTATION_TYPE="single"  # "single" or "original"

if [ $EVAL_TYPE == "none" ]; then
    DATASET_DIR="/temp/datasets/${EVAL_DATASET}"
else
    DATASET_DIR="/temp/datasets/${EVAL_DATASET}/${EVAL_TYPE}"
fi
EVAL_DIR="/temp/eval_dir/${EVAL_DATASET}"
EVAL_PATH="${EVAL_DIR}/${EVAL_DATASET}_${MODEL_NAME}.npy"

EVAL_CMD="python eval_dataset.py \
    --dataset_dir=${DATASET_DIR} \
    --tfrecord_dir=/temp/datasets/tfrecords/${EVAL_DATASET} \
    --model_name=${MODEL_NAME} \
    --models_dir=/temp/model_dir/${TRAIN_DATASET} \
    --eval_dir=${EVAL_DIR} \
    --eval_type=${EVAL_TYPE} \
    --annotation_type=${ANNOTATION_TYPE} \
    --batch_size=${BATCH_SIZE}"

if [ $ANNOTATION_TYPE == "single" ]; then
    ANN_NAME="_single_annotations_array.npy"
elif [ $ANNOTATION_TYPE == "original" ]; then
    ANN_NAME="_annotations_array.npy"
fi

if [ $EXPERIMENT != "none" ]; then
    EXPERIMENT_CMD="python ../../Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/experiments/detection_experiment.py \
        --experiment=${EXPERIMENT} \
        --eval_file=${EVAL_PATH} \
        --gt_file=${DATASET_DIR}/${ANN_NAME}"
else
    EXPERIMENT_CMD="echo Skippping experiment step"
fi

# ---------

START_TIME=$(date +%s)

IMAGE_PATH="${MYGROUP}/singularity_images/tf_container_2203_efficientdet.sif"
srun singularity run --nv \
    -B ${MYGROUP}:/temp ${IMAGE_PATH} \
    /bin/bash -c "${EVAL_CMD}; ${EXPERIMENT_CMD}"

END_TIME=$(date +%s)

echo "Time taken is equal to: $(( END_TIME - START_TIME )) seconds"
