#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source env/bin/activate

python translate_training_data.py