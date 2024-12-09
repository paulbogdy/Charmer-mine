#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer-mine/baseline/roben/scRNN
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/mnlp/bin/activate

TASK_NAME=SST-2
GLUE_DIR=data/glue_data
TC_DIR=tc_data

cd ..
python3 preprocess_tc.py --glue_dir $GLUE_DIR --save_dir $TC_DIR/glue_tc_preprocessed

cd scRNN
python3 train.py --task-name $TASK_NAME --preprocessed_glue_dir ../$TC_DIR/glue_tc_preprocessed --tc_dir ../$TC_DIR
