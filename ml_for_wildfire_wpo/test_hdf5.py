"""USE ONCE AND DESTROY."""

import h5py

MODEL_FILE_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/experiment17/training-strategy=20-30-daily_spectral-complexity=080_use-residual-blocks=0/model.weights.h5'

# Open the file in read mode
with h5py.File(MODEL_FILE_NAME, 'r') as f:
    # List all groups and datasets in the file
    def print_structure(name, obj):
        print(f"{name}: {type(obj)}")
    f.visititems(print_structure)
