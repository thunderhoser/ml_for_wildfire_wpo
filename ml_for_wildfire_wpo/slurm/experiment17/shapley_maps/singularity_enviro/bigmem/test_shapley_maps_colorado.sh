#!/bin/sh

model_file_name=$1
use_deep_explainer=$2
use_inout_tensors_only=$3
disable_tensorflow2=$4
disable_eager_execution=$5
output_dir_name=$6

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"
TEMPLATE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/experiment17/templates"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_models/experiment17"

GFS_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed/quantile_normalized_params_from_2019-2020"
TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi"
GFS_FORECAST_TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed_fwi_forecasts"
TARGET_NORM_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi/normalization_params_2019-2020.nc"
ERA5_CONSTANT_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/era5_constants.nc"
ERA5_CONSTANT_NORM_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/normalization_params_era5_constants.nc"

python3 -u "${CODE_DIR_NAME}/make_shapley_maps.py" \
--input_model_file_name="${model_file_name}" \
--input_gfs_directory_name="${GFS_DIR_NAME}" \
--input_target_dir_name="${TARGET_DIR_NAME}" \
--input_gfs_fcst_target_dir_name="${GFS_FORECAST_TARGET_DIR_NAME}" \
--model_lead_time_days=2 \
--baseline_init_date_strings "20200801" "20200802" "20200803" "20200804" "20200805" "20200806" "20200807" "20200808" "20200809" "20200810" \
--new_init_date_strings "20210809" "20210810" \
--input_region_mask_file_name="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/colorado_mask.nc" \
--target_field_name="fire_weather_index" \
--use_deep_explainer=${use_deep_explainer} \
--use_inout_tensors_only=${use_inout_tensors_only} \
--disable_tensorflow2=${disable_tensorflow2} \
--disable_eager_execution=${disable_eager_execution} \
--output_dir_name="${output_dir_name}"
