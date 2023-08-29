#!/bin/bash -l

# Request GPU
#$ -l gpu=1

# request an A100 node only
#$ -ac allow=L

# Request 72hrs
#$ -l h_rt=72:00:00

# Request 32GB RAM
#$ -l mem=32G

# Request 16 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=16G

# Set name of job
#$ -N fasterrcnn-rgb

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
pip install -q -r /home/rmhisyl/Scratch/diss/cnn_requirements.txt

# Run the script
python /home/rmhisyl/Scratch/diss/src/train_obj_detector.py -p /home/rmhisyl/Scratch/diss -c RGB
