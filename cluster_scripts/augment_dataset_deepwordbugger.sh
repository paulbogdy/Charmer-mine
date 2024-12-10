#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer-mine
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 24:00:00
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

charmer_ending_path=_50iter_encoder_margin_pga0_batch20_1000.csv
final_results_path=augmented_dataset

# Create the result path if it does not exist
mkdir -p $final_results_path

if [ -f "$final_results_path/deepwordbug.csv" ]; then
    echo "deepwordbug already exists, skipping..."
else
    python attack.py \
        --device cuda \
        --loss margin \
        --dataset $dataset \
        --model $model_path \
        --attack_name deepwordbug \
        --pga 0 \
        --on_train

    mv "$result_path/${attack}_${dataset}_${model_name}.csv" "$final_results_path/deepwordbug.csv"
fi