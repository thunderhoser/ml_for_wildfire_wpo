#!/bin/sh

model_file_name=$1
output_dir_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

GFS_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed/quantile_normalized_params_from_2019-2020"
TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi"
GFS_FORECAST_TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed_fwi_forecasts"

python3 -u "${CODE_DIR_NAME}/make_shapley_maps.py" \
--input_model_file_name="${model_file_name}" \
--input_gfs_directory_name="${GFS_DIR_NAME}" \
--input_target_dir_name="${TARGET_DIR_NAME}" \
--input_gfs_fcst_target_dir_name="${GFS_FORECAST_TARGET_DIR_NAME}" \
--model_lead_time_days=3 \
--baseline_init_date_strings "20200801" "20200802" "20200803" "20200804" "20200805" "20200806" "20200807" "20200808" "20200809" "20200810" \
--new_init_date_strings "20210809" "20210810" \
--input_region_mask_file_name="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/smaller_colorado_mask.nc" \
--target_field_name="fine_fuel_moisture_code" \
--use_inout_tensors_only=0 \
--disable_tensorflow2=1 \
--disable_eager_execution=1 \
--output_dir_name="${output_dir_name}"
