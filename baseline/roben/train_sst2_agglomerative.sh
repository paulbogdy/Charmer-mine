#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer-mine/baseline/roben
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/mnlp/bin/activate

TASK_NAME=SST-2
GLUE_DIR=data/glue_data
CLUSTERER_PATH=clusterers/vocab100000_ed1_gamma0.3.pkl

python3 run_glue.py --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir model_output_agglomerative/$TASK_NAME --overwrite_output_dir --seed_output_dir --save_results --save_dir codalab --recoverer clust-rep --clusterer_path $CLUSTERER_PATH --augmentor identity --run_test --do_robust
