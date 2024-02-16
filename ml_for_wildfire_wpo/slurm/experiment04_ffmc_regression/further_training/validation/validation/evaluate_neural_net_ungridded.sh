#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

python3 -u "${CODE_DIR_NAME}/evaluate_model.py" \
--input_prediction_dir_name="${model_dir_name}/validation" \
--init_date_limit_strings "20210101" "20211210" \
--num_bootstrap_reps=1 \
--target_field_names "fine_fuel_moisture_code" \
--num_relia_bins_by_target 102 \
--min_relia_bin_edge_by_target 0 \
--max_relia_bin_edge_by_target 102 \
--per_grid_cell=0 \
--output_file_name="${model_dir_name}/validation/ungridded_evaluation.nc"
