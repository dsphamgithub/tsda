#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=00:20:00

module load python
module load singularity

if [ $TF_TRAIN_ONLY == "true" ]; then
    TFRECORD_CMD="python generate_tfrecords.py \
        --dataset ${MYGROUP}/datasets/${TF_DATASET} \
        --datasets_dir ${MYGROUP}/datasets \
        --train_only"
else
    TFRECORD_CMD="python generate_tfrecords.py \
        --dataset ${MYGROUP}/datasets/${TF_DATASET} \
        --datasets_dir ${MYGROUP}/datasets"
fi

echo
echo ${TFRECORD_CMD}
echo

singularity run --nv \
    -B ${MYGROUP}:/temp ${TF_IMAGE_PATH} \
    /bin/bash -c "${TFRECORD_CMD}"

printf "\nFinished generating TFRECORD directory\n"
