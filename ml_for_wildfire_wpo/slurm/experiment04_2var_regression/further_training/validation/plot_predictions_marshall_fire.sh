#!/bin/sh

model_dir_name=$1
init_date_string=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

python3 -u "${CODE_DIR_NAME}/plot_predictions.py" \
--input_prediction_dir_name="${model_dir_name}/validation" \
--field_names "fine_fuel_moisture_code" "buildup_index" \
--init_date_string="${init_date_string}" \
--output_dir_name="${model_dir_name}/validation/prediction_plots"
