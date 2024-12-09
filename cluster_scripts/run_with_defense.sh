#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer-mine
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 1:45:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/mnlp/bin/activate

model_name=$1
if [ -d "./$model_name" ]; then
    echo "Model directory $model_name already exists in the current directory."
elif [ -d "../Robust-Training/$model_name" ]; then
    # Move the model directory if it exists in ../Robust-Training
    mv ../Robust-Training/$model_name .
    echo "Moved $model_name from ../Robust-Training to the current directory."
else
    # Stop the script if the model directory does not exist in either location
    echo "Model directory $model_name does not exist in ../Robust-Training. Exiting."
    exit 1
fi

model_path=./$model_name # Modify the epoch if needed
dataset=$2
result_path=results_attack/lm_classifier/basiclm/$dataset

defense=$3

python attack.py \
    --device cuda \
    --loss margin \
    --dataset $dataset \
    --model $model_path \
    --k 1 \
    --n_positions 20 \
    --select_pos_mode batch \
    --size 1000 \
    --pga 0 \
    --checker ScRNN