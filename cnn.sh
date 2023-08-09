#!/bin/bash -l

# Request GPU
#$ -l gpu=1

# request V100 or A100 node
#$ -ac allow=EFL

# Request 20hrs
#$ -l h_rt=20:00:00

# Request 40GB RAM
#$ -l mem=40G

# Request 64 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=16G

# Set name of job
#$ -N diss-cnn-20x

# Set working directory
#$ -wd /home/rmhisyl/Scratch/diss

# Send email when job begins, ends, or aborts
#$ -m bea

# Change into temporary directory to run work
cd $TMPDIR

# Load modules
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

# Install packages
pip install --upgrade pip
pip install -q -r /home/rmhisyl/Scratch/diss/cnn_requirements.txt

# Run the script
python /home/rmhisyl/Scratch/diss/patch_transform_and_classifier.py

# Copy files back to scratch
tar -zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR