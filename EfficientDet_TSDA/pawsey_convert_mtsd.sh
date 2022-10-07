#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuq-dev
#SBATCH --account=director2191
#SBATCH --time=00:20:00

module load python
module load singularity

# 'train', 'val', or 'test'
SPLIT="train"

CONVERT_CMD="python convert_mtsd.py \
    --split_path=/group/director2191/krados/datasets/mtsd/mtsd_v2_fully_annotated/splits/${SPLIT}.txt \
    --annotations_dir=/group/director2191/krados/datasets/mtsd/mtsd_v2_fully_annotated/annotations"

IMAGE_PATH="${MYGROUP}/singularity_images/tf_container_2203_efficientdet.sif"
singularity run --nv \
    -B ${MYGROUP}:/temp ${IMAGE_PATH} \
    /bin/bash -c "${CONVERT_CMD}"

printf "\nFinished MTSD label converstion to GTSDB for /${SPLIT}\n"
