#!/bin/bash

# Listes de datasets et de modèles
datasets=("WN18RR")
models=("ConvE")

# Parcours de tous les couples (model, dataset)
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python3 main.py --data_dir data/${dataset} --model $model --use_gpu --GPU_reproductibility --stability_training --oar --oar_besteffort
    done
done