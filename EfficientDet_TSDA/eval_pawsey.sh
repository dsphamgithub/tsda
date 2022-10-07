#!/bin/bash

#SBATCH --job-name=edet-eval
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=00:20:00

module load python
module load singularity

# NOTE: Retrieved model checkpoint must be from damage assessment model, can't be from standard EfficientDet
TRAIN_DATASET="gtsdb_0.25_extended_2"
EVAL_DATASET="12000_synth_test"
MODEL_NAME="efficientdet-d0"  # Eval most recent ckpt in model_dir/${TRAIN_DATASET} with ${MODEL_NAME} in name
EXPERIMENT="none"  # "damage", "distance", "sequence", "none", or "damage_assessment"
EVAL_TYPE="none"  # "test", "valid", or "none"
ANNOTATION_TYPE="single"  # "single" or "original"
BATCH_SIZE="32"  # Each gpuq-dev GPU can handle at least 8 MTSD images for eval
# TODO: HPARAMS parameter for eval_dataset.py

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

# TODO: Add sklearn to container
if [ $EXPERIMENT == "damage_assessment" ]; then
    EXPERIMENT_CMD="python ../Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/experiments/dmg_assessment_experiments.py \
        --eval_file=${EVAL_PATH} \
        --gt_file=${DATASET_DIR}/${ANN_NAME}"
else
    EXPERIMENT_CMD="python ../Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/experiments/detection_experiment.py \
        --experiment=${EXPERIMENT} \
        --eval_file=${EVAL_PATH} \
        --gt_file=${DATASET_DIR}/${ANN_NAME}"
fi


#---------------Running eval command inside singularity container---------------

START_TIME=$(date +%s)

IMAGE_PATH="${MYGROUP}/singularity_images/tf_container_2105_efficientdet_precise_tf.sif"
if [ $EXPERIMENT == "none" ]; then
    CMD="${EVAL_CMD}"
else
    CMD="${EVAL_CMD}; ${EXPERIMENT_CMD}"
fi
echo 
echo "Eval commands:"
echo $CMD
srun singularity run --nv \
    -B ${MYGROUP}:/temp ${IMAGE_PATH} \
    /bin/bash -c "${CMD}"

END_TIME=$(date +%s)

echo "Time taken is equal to: $(( END_TIME - START_TIME )) seconds"
