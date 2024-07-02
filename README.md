# EEG-to-Text-LLM

This repository contains the implementation for converting EEG signals to text using various pre-trained language models such as BART, T5, and Llama 2. 
It is based on the [EEG-to-Text](https://github.com/MikeWangWZHL/EEG-To-Text) codes & implementation. 

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setting Up the Environment](#setting-up-the-environment)
- [Handling Scripts for Easy Management](#handling-scripts-for-easy-management)
- [File Overview](#file-overview)
- [Running Jobs on a Cluster](#running-jobs-on-a-cluster)
  - [Requesting CPU Resources in a GPU Job](#requesting-cpu-resources-in-a-gpu-job)
  - [Monitoring GPU Jobs](#monitoring-gpu-jobs)
  - [Preparing and Submitting a Job Script](#preparing-and-submitting-a-job-script)
  - [Example Workflow](#example-workflow)
    - [Preparing the Dataset](#preparing-the-dataset)
    - [Training a Model](#training-a-model)
    - [Evaluating a Model](#evaluating-a-model)

## Repository Structure

The repository contains several directories and files crucial for running the EEG-to-text models. Below is a brief overview:

- `scripts/`: Contains shell scripts for training and evaluation.
- `util/`: Utility functions for data processing.
- `.gitignore`: Specifies files to be ignored by Git.
- `README.md`: Initial setup and information about the repository.
- `config.py`, `config_t5.py`: Configuration files for different models.
- `data.py`: Scripts for data handling and processing.
- `environment.yml`: Environment setup file.
- `eval_*.py`: Evaluation scripts for different models.
- `metrics.py`: Script for calculating evaluation metrics.
- `model_*.py`: Model definitions for different architectures.
- `train_*.py`: Training scripts for different models.

## Setting Up the Environment

To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yoghurtina/eeg-to-text-llm.git
    cd eeg-to-text-llm
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate EEGToText
    ```

## Handling Scripts for Easy Management

The `scripts/` directory contains several shell scripts designed for easy handling of various tasks. Below is a brief description of each script:

- `eval_bart_base.sh`, `eval_bart_large.sh`, `eval_t5.sh`: Scripts to evaluate the BART Base, BART Large, and T5 models, respectively.
- `eval_job.sh`: General evaluation script.
- `prepare_dataset.sh`: Script to preprocess and prepare the dataset.
- `train_bart_base.sh`, `train_bart_large.sh`, `train_t5.sh`: Scripts to train the BART Base, BART Large, and T5 models, respectively.
- `train_job.sh`: General training script.
- `transform.sh`: Script for data transformation.

## File Overview

Below are the key files and their purposes:

- `eval_bart_base.py`, `eval_bart_large.py`, `eval_t5.py`: Python scripts for evaluating different models.
- `model_bart_base.py`, `model_bart_large.py`, `model_llama.py`, `model_t5.py`: Python scripts defining the architecture of various models.
- `train_bart_base.py`, `train_bart_large.py`, `train_llama.py`, `train_t5.py`: Python scripts for training different models.

## Running Jobs on a Cluster

To run training and evaluation jobs on a cluster efficiently, follow these essential guidelines (need to be adjusted based on the cluster you use):

### Requesting CPU Resources in a GPU Job

Each GPU job should request a minimum of 4 CPU cores per GPU. For optimal performance, adhere to the hardware configuration of the servers. For example:
- **Single GPU on a 16-core CPU server:** Request up to 16 cores.
- **Multiple GPUs per CPU server:** Adjust the cores accordingly (e.g., 4 GPUs on a 32-core server, request up to 8 cores per GPU).

### Monitoring GPU Jobs

Use the `bnvtop` command to monitor GPU jobs in real-time:
```bash
bnvtop JOBID
