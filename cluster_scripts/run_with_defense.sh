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
result_path=results_attack/lm_classifier/basiclm_attack_checker/$dataset

defense=$3

charmer_ending_path=_50iter_encoder_margin_pga0_batch20_1000.csv
final_results_path=$model_name/results_$defense

# Create the result path if it does not exist
mkdir -p $final_results_path

# Charmer attacks
charmer_ks=(1 2 10)

for k in ${charmer_ks[@]}; do
    if [ -f "$final_results_path/charmer_$k.csv" ]; then
        echo "Charmer $k already exists, skipping..."
    else
        python attack.py \
            --device cuda \
            --loss margin \
            --dataset $dataset \
            --model $model_path \
            --k $k \
            --n_positions 20 \
            --select_pos_mode batch \
            --size 1000 \
            --pga 0 \
            --checker $defense

        mv "$result_path/${model_name}_${k}${charmer_ending_path}" "$final_results_path/charmer_$k.csv"
    fi
done

# Aggregate the results

python cluster_scripts/evaluate.py \
    --folder_path $final_results_path > $final_results_path/eval.out