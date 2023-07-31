#!/bin/bash -l

# Request 3hrs
#$ -l h_rt=03:00:00

# Request 2GB RAM
#$ -l mem=2G

# Request 40 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=40G

# Set name of job
#$ -N diss-extract-patches

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
module load openjpeg/2.4.0/gnu-4.9.2
module load openslide/3.4.1/gnu-4.9.2
module load python3/3.9-gnu-10.2.0

# Install packages
pip install -q opencv-python pillow matplotlib numpy tqdm patchify geojson openslide-python

# Run the notebook
python /home/rmhisyl/Scratch/diss/extract_patches.py

# Copy files back to scratch
tar -zcvf $HOME/Scratch/diss/files_from_job_$JOB_ID.tar.gz $TMPDIR
