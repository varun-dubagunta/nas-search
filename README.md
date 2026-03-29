# Distributed Neural Architecture Search for Higgs Boson Classification

Automated hyperparameter and architecture search over a physics-aware transformer
for 4-class Higgs boson production mode classification, parallelized across 
multiple workers using Ray Tune and the ASHA scheduling algorithm.

## Background

As part of ongoing undergraduate research at Texas A&M University, I work on 
machine learning classification problems in high energy physics — specifically 
distinguishing Higgs boson production modes from proton-proton collision data 
collected by the CMS detector at CERN. The classification task involves 
separating ggH, VBFH, Drell-Yan, and electroweak processes from a set of 
physics-motivated kinematic features derived from reconstructed muon and jet 
objects.

One recurring challenge in that work is architectural decision-making. 
Transformer-based models introduce a large number of interacting hyperparameters 
— embedding dimensions, attention heads, dropout schedules, learning rates — 
and manual tuning at the scale required for research-quality results is both 
time-consuming and unreliable. This project is a direct response to that 
problem: rather than tuning by intuition, define a search space and let a 
distributed system find the optimal configuration automatically.

## Approach

Trials are parallelized across available hardware using Ray Tune. The ASHA 
scheduler eliminates underperforming configurations early, concentrating compute 
on promising architectures. Optuna provides Bayesian optimization over the 
search space rather than random sampling, improving efficiency over naive grid 
or random search.

The underlying model is a physics-aware transformer that incorporates domain 
knowledge through structured input tokenization — grouping features by physical 
object before the attention layers. This design choice reflects the structure 
of the underlying physics rather than treating all input features as 
interchangeable.

## Scaling to HPC

The search is designed to scale to multi-node clusters via SLURM. On TAMU HPRC 
or ACCESS/Expanse, Ray initializes across allocated nodes and distributes trials 
across all available GPUs simultaneously. A single flag change connects the 
driver to the cluster-wide Ray instance — no code changes required.

## Stack

Ray Tune · TensorFlow/Keras · Optuna · ASHA · Python

## Usage
```bash
# Run architecture search
python nas_search.py \
    --data_path ./data/dataset.pkl \
    --num_samples 50 \
    --max_epochs 30 \
    --num_cpus 4 \
    --num_gpus 1

# Train best found configuration to convergence
python train_best.py \
    --data_path ./data/dataset.pkl \
    --epochs 100
```
