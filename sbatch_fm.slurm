#!/bin/bash

#SBATCH --job-name=FM_IAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=144:00:00         # max time for running this job
#SBATCH --gres=gpu:v100:1         # request for any gpu available

##SBATCH --output=log/slurm-%j.out # [optional] output log file, need to create a folder named log
#SBATCH--error=log/slurm-%j.err

# Singularity path
ext3_path=/scratch/qg2089/FM_demo/overlay-50G-10M.ext3
sif_path=/scratch/qg2089/FM_demo/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
# conda_env=

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate fm
    python Conditioned_FM_2ref.py
"