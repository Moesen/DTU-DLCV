#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J **NAME**
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:45
#BSUB -R "rusage[mem=32GB]"
#BSUB -u **EMAIL**
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o **PATHTOLOG**/logs/%J.out
#BSUB -e **PATHTOLOGFOLDE**/logs/%J.err
# -- end of LSF options --


# Load the cuda module
module load python3/3.8.0
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2
echo "Running script..."
cd /zhome/69/1/137385/Desktop/Bachelor/deepgen/VAE/
ls
python3 **FILENAME**