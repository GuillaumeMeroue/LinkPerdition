# Knowledge Graph Embedding (KGE) Training Library

A library for training Knowledge Graph Embeddings with full control over randomness sources, largely inspired by [LibKGE](https://github.com/uma-pi1/kge).


## Supplementary figures

Some supplementary figures that doesn't fit in the paper are present in the `Results` folder.

## Key Features

- **Four Independent Randomness Sources**: Complete control and independence over initialization, triple ordering, negative sampling, and dropout
- **Multiple KGE Models**: Support for TransE, DistMult, ConvE, RGCN, and Transformer architectures
- **Stability Analysis**: Computation of metrics to measure model stability across different random seeds
- **Hyperparameter Optimization**: Integration with Weights & Biases for sweep experiments


## Code Organization

```
├── data/                     # Dataset directory
├── main.py                   # Main entry point for training and experiments
├── kge/                      # Core KGE library
│   ├── models.py             # KGE model implementations (TransE, DistMult, ConvE, etc.)
│   ├── train.py              # Training loop and optimization
│   ├── data.py               # Data loading and preprocessing utilities
│   ├── eval.py               # Evaluation metrics and procedures
│   └── utils.py              # Utility functions and seed management
├── stability.py              # Stability experiment orchestration
├── training_utils.py         # Model initialization and training utilities
├── sweep_utils.py            # Hyperparameter sweep utilities
├── stability_measures/       # Stability analysis scripts and results
│   ├── stability_measures.py # Stability analysis script
│   ├── stability_measures_predictions.py  # For predictions metrics
│   └── stability_measures_space.py  # For space metrics
└── tests/...                  # Test suite
```

## Testing

Run the comprehensive test suite with pytest:

```bash
pytest tests/
```

### Test Categories

- **`test_seeds_MODELS.py`**: Verify that all randomness sources are reproducible and distinct for each model (TransE, DistMult, ConvE, Transformer)
- **`test_checkpointing.py`**: Ensure training can be resumed while maintaining reproducibility of random states
- **`test_reprod_train.py`** & **`test_reprod_sampler.py`**: Validate reproducibility of training procedures and negative samplers
- **`test_stability_space_equivalence.py`**: Comfirm that the optimised (and ugly) space metrics are equivalent to the original space metrics
- **Additional tests**: Non-critical, but give a nice dopamine hit when green.


## Usage

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```


### Example of  training with custom seed configuration

```bash
python3 main.py \
    --data_dir data/nations \
    --model DistMult \
    --seed_init 42 \
    --seed_neg 123 \
    --seed_order 456 \
    --seed_forward 789 \
    --use_gpu \
    --no-log_to_wandb
```


### Protocole from the paper to have results on Nations dataset with DistMult

#### 1. Hyperparameter Tuning

Run hyperparameter optimization using Weights & Biases:

```bash
python3 main.py --sweep_id=$SWEEP_ID --data_dir data/nations --model DistMult --use_gpu --GPU_reproductibility
```

if you don't have a sweep_id, you can use the sweep_luncher.sh script, to create one:

```bash
./sweep_luncher.sh
```

#### 2. Stability Training

Run multiple training sessions with different seeds to assess model stability:

```bash
python3 main.py --data_dir data/nations --model DistMult --use_gpu --GPU_reproductibility --stability_training --oar
```

**Options:**
- `--oar`: Launch parallel runs on OAR cluster
- Without `--oar`: Run sequentially on local machine

#### 3. Calculate Stability Metrics

Compute stability measures from multiple training runs:

```bash
python3 main.py --data_dir data/nations --model DistMult --stability_measures
```


## Hyperparameters

We fixed the seed configuration to 𝔖 = { 𝔖ₙ = s₁, 𝔖ₒ = s₁, 𝔖ᵢ = s₁, 𝔖𝔻 = s₁ } and conducted a compact hyperparameter search over embedding dimensions {128, 256, 512} and learning rates {10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹}.  
All models are initialized using Xavier normal, and we employ the cross-entropy loss, as prior work shows that this choice consistently yields reliable results.

Dropout rates for entities and relations are fixed at 0.2, the batch size is set to 256, and the number of negative samples is dataset-dependent: 10 for Kinship and Nations, and 500 for all other datasets.  
Inverse relations are enabled by default: for each training triple (h, r, t), we additionally generate (t, r⁻¹, h), and during inference, queries of the form (?, r, t) are replaced by (t, r⁻¹, ?).

As in most link prediction settings, early stopping is performed using the MRR as validation criterion, with a patience of 50 epochs and an improvement threshold of 10⁻³.

For TransE, we adopt the ℓ₂ norm.  

For ConvE, we set the projection dropout to 0.3 and the feature map dropout to 0.2, following configurations from public implementations.

For Transformer, we use 8 attention heads, a feed-forward dimension of 1280, 3 encoder layers, ReLU activation, and an encoder dropout of 0.1.

For RGCN, we apply a hidden dropout of 0.2 and use two encoding layers, leveraging basis decomposition with two basis functions.

### Hyperparameter Configurations

| Model       | Dataset      | Best (LR,Dim)     | Median (LR,Dim)   | Worst (LR,Dim)    |
|-------------|-------------|------------------|------------------|------------------|
| **Transformer** | WN18RR     | (1e-02,512) | (1e-01,128) | (1e-02,128) |
|             | FB15k-237   | (1e-02,256) | (1e-01,128) | (1e-03,512) |
|             | codex-s     | (1e-01,256) | (1e-02,512) | (1e-04,128) |
|             | kinship     | (1e-01,128) | (1e-03,512) | (1e-01,512) |
|             | nations     | (1e-01,256) | (1e-04,128) | (1e-06,256) |
|-------------|-------------|------------------|------------------|------------------|
| **TransE**  | WN18RR      | (1e-02,512) | (1e-01,256) | (1e-03,128) |
|             | FB15k-237   | (1e-02,512) | (1e-04,512) | (1e-05,128) |
|             | codex-s     | (1e-02,512) | (1e-03,512) | (1e-05,512) |
|             | kinship     | (1e-01,128) | (1e-01,512) | (1e-03,128) |
|             | nations     | (1e-01,512) | (1e-04,512) | (1e-06,256) |
|-------------|-------------|------------------|------------------|------------------|
| **DistMult**| WN18RR      | (1e-02,512) | (1e-03,512) | (1e-03,128) |
|             | FB15k-237   | (1e-02,128) | (1e-03,256) | (1e-04,128) |
|             | codex-s     | (1e-01,128) | (1e-02,512) | (1e-03,128) |
|             | kinship     | (1e-02,128) | (1e-04,256) | (1e-06,512) |
|             | nations     | (1e-01,128) | (1e-04,512) | (1e-06,256) |
|-------------|-------------|------------------|------------------|------------------|
| **ConvE**   | WN18RR      | (1e-03,512) | (1e-03,256) | (1e-03,128) |
|             | FB15k-237   | (1e-03,256) | (1e-01,512) | (1e-06,256) |
|             | codex-s     | (1e-03,512) | (1e-03,128) | (1e-06,256) |
|             | kinship     | (1e-01,256) | (1e-04,512) | (1e-06,128) |
|             | nations     | (1e-02,512) | (1e-03,128) | (1e-06,128) |
|-------------|-------------|------------------|------------------|------------------|
| **RGCN**    | WN18RR      | (1e-01,256) | (1e-02,256) | (1e-03,128) |
|             | codex-s     | (1e-01,128) | (1e-03,128) | (1e-05,256) |
|             | kinship     | (1e-02,128) | (1e-02,512) | (1e-06,256) |
|             | nations     | (1e-01,128) | (1e-04,256) | (1e-06,128) |
|-------------|-------------|------------------|------------------|------------------|
| **RotatE**  | WN18RR      | (1e-01,128) | (1e-03,512) | (1e-03,128) |
|             | FB15k-237   | (1e-02,256) | (1e-01,256) | (1e-03,128) |
|             | codex-s     | (1e-02,512) | (1e-03,512) | (1e-04,256) |
|             | kinship     | (1e-01,256) | (1e-02,128) | (1e-04,128) |
|             | nations     | (1e-01,128) | (1e-04,128) | (1e-06,512) |
|-------------|-------------|------------------|------------------|------------------|
| **ComplEx** | WN18RR      | (1e-02,512) | (1e-01,256) | (1e-03,128) |
|             | FB15k-237   | (1e-02,128) | (1e-01,256) | (1e-04,256) |
|             | codex-s     | (1e-01,128) | (1e-01,256) | (1e-01,512) |
|             | kinship     | (1e-02,512) | (1e-04,512) | (1e-05,128) |
|             | nations     | (1e-02,512) | (1e-04,512) | (1e-05,128) |

### SOTA vs Ours 

#### MRR
| Dataset | TransE | ConvE | DistMult | Transformer | RGCN | ComplEx | RotatE |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **WN18RR (SOTA)** | 0.228 | 0.442 | 0.452 | 0.473 | — | 0.475 | 0.478 |
| **WN18RR (Ours)** | 0.194 ± 0.002 | 0.410 ± 0.004 | 0.422 ± 0.002 | 0.269 ± 0.011 | 0.389 ± 0.002 | 0.435 ± 0.001 | 0.326 ± 0.044 |
| **FB15k-237 (SOTA)** | 0.313 | 0.339 | 0.343 | — | 0.248 | 0.348 | 0.333 |
| **FB15k-237 (Ours)** | 0.315 ± 0.001 | 0.324 ± 0.002 | 0.312 ± 0.002 | 0.295 ± 0.001 | N/A | 0.315 ± 0.002 | 0.288 ± 0.001 |
| **CoDEx-S (SOTA)** | 0.354 | 0.444 | — | — | — | 0.465 | — |
| **CoDEx-S (Ours)** | 0.348 ± 0.002 | 0.434 ± 0.003 | 0.413 ± 0.003 | 0.360 ± 0.007 | 0.362 ± 0.011 | 0.396 ± 0.002 | 0.359 ± 0.003 |

#### Hits@1

| Dataset | TransE | ConvE | DistMult | Transformer | RGCN | ComplEx | RotatE |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **WN18RR (SOTA)** | 0.053 | 0.411 | 0.413 | 0.462 | — | 0.438 | 0.439 |
| **WN18RR (Ours)** | 0.020 ± 0.001 | 0.370 ± 0.005 | 0.395 ± 0.003 | 0.201 ± 0.012 | 0.363 ± 0.003 | 0.410 ± 0.002 | 0.252 ± 0.085 |
| **FB15k-237 (SOTA)** | 0.221 | 0.248 | 0.250 | 0.279 | 0.153 | 0.253 | 0.240 |
| **FB15k-237 (Ours)** | 0.224 ± 0.001 | 0.236 ± 0.001 | 0.226 ± 0.002 | 0.209 ± 0.001 | N/A | 0.226 ± 0.002 | 0.198 ± 0.001 |
| **CoDEx-S (SOTA)** | 0.219 | 0.343 | — | — | — | 0.372 | — |
| **CoDEx-S (Ours)** | 0.226 ± 0.002 | 0.329 ± 0.005 | 0.307 ± 0.006 | 0.255 ± 0.009 | 0.253 ± 0.011 | 0.289 ± 0.003 | 0.245 ± 0.006 |


#### Hits@10

| Dataset | TransE | ConvE | DistMult | Transformer | RGCN | ComplEx | RotatE |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **WN18RR (SOTA)** | 0.520 | 0.504 | 0.530 | 0.584 | — | 0.547 | 0.553 |
| **WN18RR (Ours)** | 0.484 ± 0.001 | 0.484 ± 0.004 | 0.474 ± 0.003 | 0.396 ± 0.013 | 0.431 ± 0.002 | 0.484 ± 0.003 | 0.423 ± 0.003 |
| **FB15k-237 (SOTA)** | 0.497 | 0.521 | 0.531 | 0.558 | 0.414 | 0.536 | 0.522 |
| **FB15k-237 (Ours)** | 0.497 ± 0.002 | 0.501 ± 0.002 | 0.483 ± 0.003 | 0.466 ± 0.002 | N/A | 0.491 ± 0.002 | 0.465 ± 0.002 |
| **CoDEx-S (SOTA)** | 0.634 | 0.635 | — | — | — | 0.646 | — |
| **CoDEx-S (Ours)** | 0.584 ± 0.004 | 0.638 ± 0.006 | 0.613 ± 0.006 | 0.564 ± 0.006 | 0.580 ± 0.014 | 0.607 ± 0.001 | 0.584 ± 0.002 |


Our goal is not to reach SOTA performance but to define three clearly distinct performance levels. The best identified configuration remains reasonably close to SOTA (source : https://github.com/uma-pi1/kge and https://arxiv.org/pdf/1703.06103 and https://arxiv.org/pdf/2008.12813).
