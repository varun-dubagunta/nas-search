"""
Distributed Neural Architecture Search — Higgs Discovery Transformer
---------------------------------------------------------------------
Uses Ray Tune + ASHA scheduler to search over transformer hyperparameters
in parallel, finding the optimal architecture for 4-class Higgs classification.

Classes are read dynamically from the 'process' column — no hardcoding.

Usage:
    python nas_search.py --num_samples 50 --max_epochs 30 --data_path ./data/DNN_samples_v4.pkl
"""

import os
import argparse
import pickle
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import tensorflow as tf

from utils.data import load_data
from models.transformer import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_args():
    parser = argparse.ArgumentParser(description='NAS for Higgs Transformer')
    parser.add_argument('--data_path',    type=str,   default='./data/DNN_samples_v4.pkl')
    parser.add_argument('--num_samples',  type=int,   default=50)
    parser.add_argument('--max_epochs',   type=int,   default=30)
    parser.add_argument('--grace_period', type=int,   default=5)
    parser.add_argument('--num_cpus',     type=int,   default=4)
    parser.add_argument('--num_gpus',     type=float, default=1.0)
    parser.add_argument('--results_dir',  type=str,   default='./results')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load data in driver process ────────────────────────────────
    X_train, X_val, y_train, y_val, scaler, class_names = load_data(args.data_path)
    print(f"Classes: {class_names}")

    # ── Put data in Ray object store ──────────────────────────────
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus, ignore_reinit_error=True)

    X_train_ref = ray.put(X_train)
    X_val_ref   = ray.put(X_val)
    y_train_ref = ray.put(y_train)
    y_val_ref   = ray.put(y_val)

    print(f"\nRay initialized — data in object store")
    print(f"Running {args.num_samples} trials in parallel\n")

    # Store refs in a module-level dict so trainable can access them
    # without closing over the arrays directly (avoids 132MB limit)
    _DATA_REFS = {
        'X_train': X_train_ref,
        'X_val':   X_val_ref,
        'y_train': y_train_ref,
        'y_val':   y_val_ref,
    }

    def trainable(config):
        X_tr = ray.get(_DATA_REFS['X_train'])
        X_vl = ray.get(_DATA_REFS['X_val'])
        y_tr = ray.get(_DATA_REFS['y_train'])
        y_vl = ray.get(_DATA_REFS['y_val'])

        tf.keras.backend.clear_session()
        model = build_model(config)

        for epoch in range(args.max_epochs):
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_vl, y_vl),
                epochs=1,
                batch_size=config['batch_size'],
                verbose=0,
            )
            tune.report({
                'val_accuracy': history.history['val_accuracy'][0],
                'val_loss':     history.history['val_loss'][0],
                'epoch':        epoch
            })

    # ── Search space ───────────────────────────────────────────────
    search_space = {
        # num_classes is fixed — not searched, just passed through
        'num_classes': len(class_names),

        # Architecture
        'embedding_dim': tune.choice([64, 128, 256]),
        'num_heads':     tune.choice([4, 8, 16]),
        'num_blocks':    tune.choice([1, 2, 3]),
        'ff_multiplier': tune.choice([2, 4]),

        # Regularization
        'dropout_embed': tune.uniform(0.1, 0.3),
        'dropout_attn':  tune.uniform(0.05, 0.2),
        'dropout_head':  tune.uniform(0.1, 0.3),

        # Training
        'lr':           tune.loguniform(1e-4, 1e-2),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'batch_size': tune.choice([1024, 2048, 4096]),
    }

    # ── ASHA scheduler ─────────────────────────────────────────────
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=2,
        metric='val_accuracy',
        mode='max'
    )

    search_algo = OptunaSearch(metric='val_accuracy', mode='max')

    # ── Run search ─────────────────────────────────────────────────
    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={'cpu': 1, 'gpu': args.num_gpus / max(args.num_cpus, 1)}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
            search_alg=search_algo,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(args.results_dir),
            name='higgs_nas'
        )
    )

    results = tuner.fit()

    # ── Best results ───────────────────────────────────────────────
    best = results.get_best_result(metric='val_accuracy', mode='max')
    print("\n" + "="*60)
    print("BEST TRIAL RESULTS")
    print("="*60)
    print(f"Val Accuracy: {best.metrics['val_accuracy']:.4f}")
    print(f"Val Loss:     {best.metrics['val_loss']:.4f}")
    print("\nBest config:")

    save_config = best.config
    for k, v in save_config.items():
        print(f"  {k}: {v}")

    best_config_path = os.path.join(args.results_dir, 'best_config.pkl')
    with open(best_config_path, 'wb') as f:
        pickle.dump(save_config, f)
    print(f"\nBest config saved → {best_config_path}")
    print("Run train_best.py to train the final model with this config.")

    ray.shutdown()


if __name__ == '__main__':
    main()