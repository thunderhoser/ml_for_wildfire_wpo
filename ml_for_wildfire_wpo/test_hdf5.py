"""USE ONCE AND DESTROY."""

import h5py

MODEL_FILE_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/experiment17/training-strategy=20-30-daily_spectral-complexity=080_use-residual-blocks=0/model.weights.h5'

with h5py.File(MODEL_FILE_NAME, 'r') as f:
    print("Keys in the weights file:", list(f.keys()))
    # Optionally, inspect individual layers
    for layer in f.keys():
        print(f"Layer: {layer}, Weights: {list(f[layer].keys())}")
