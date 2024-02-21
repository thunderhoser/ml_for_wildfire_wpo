#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

python3 -u "${CODE_DIR_NAME}/plot_gridded_evaluation.py" \
--input_evaluation_file_name="${model_dir_name}/validation/gridded_evaluation.nc" \
--min_colour_percentile=1 \
--max_colour_percentile=99 \
--output_dir_name="${model_dir_name}/validation/gridded_evaluation"
