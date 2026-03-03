#!/bin/bash

# Listes de datasets et de modèles
datasets=("kinship" "nations")
models=("ConvE")

# Parcours de tous les couples (model, dataset)
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Lancement pour modèle=$model, dataset=$dataset"
        oarsub -l gpu=1,walltime=4:00:00 -n measures_${model}_${dataset} -t besteffort \
            "python3 main.py --data_dir data/${dataset} --model $model --stability_measures"
    done
done