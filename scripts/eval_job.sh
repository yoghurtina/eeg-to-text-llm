#!/bin/sh
### General options
### ALL INFO AT https://www.hpc.dtu.dk/?page_id=1416#JobScript
### ------------------------------------------------------------- NAME
#BSUB -J eval_t5.gz
### 
### name to give to your job
###
### ------------------------------------------------------------- QUEUE
#BSUB -q gpuv100
### 
### specify GPU queue
### 
### ------------------------------------------------------------- WALLTIME
#BSUB -W 24:00
### 
### set wall time limit after which the job gets killed max is 24:00
###
### ------------------------------------------------------------- N. CORES
#BSUB -n 4
###
### request the number of CPU cores (at least 4x the number of GPUs)
###
### ------------------------------------------------------------- GPU REQUEST
#BSUB -gpu "num=1:mode=exclusive_process"
###
### request the number of GPUs
###
### ------------------------------------------------------------- N. HOSTS/NODES
#BSUB -R "span[hosts=1]"
###
### This means that all the cores and GPUs must be on one single host.
###
### ------------------------------------------------------------- MEMORY (RAM) per CORE
#BSUB -R "rusage[mem=4GB]"
###
### request the memory per CPU core
###
### ------------------------------------------------------------- MEMORY (RAM) per PROCESS
#BSUB -M 4GB
###
### specify the per-process memory limit
###
### ------------------------------------------------------------- E-MAIL
##BSUB -u a different e-mail if you prefer
###
### to get notifications
###
### ------------------------------------------------------------- NOTIFY when JOB STARTS
#BSUB -B
###
### write an e-mail when the job begins
###
### -------------------------------------------------------------  NOTIFY when JOB ENDS
#BSUB -N
###
### write an e-mail when the job ends
###
### ------------------------------------------------------------- OUTPUT and ERROR FOLDERS
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
### These contain the standard output (what would normally be printed in the terminal)
### and the standard error (what would be printed in the terminal in case of error)
#BSUB -o batch_output/eval_t5_%J.out
#BSUB -e batch_output/eval_t5_%J.err
###
###

# Create output directory if it doesn't exist
mkdir -p batch_output

# Initialize Conda environment
# Source the conda.sh script to enable conda commands
source /zhome/76/2/203191/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment named "EEGToText"
conda activate EEGToText

# Run the shell script using the activated conda environment
bash /work3/s233095/eeg-to-text-llm/scripts/eval_t5.sh