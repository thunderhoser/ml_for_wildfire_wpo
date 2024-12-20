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
--baseline_init_date_strings "20180730" "20180731" "20180801" "20180802" "20180803" "20180804" "20180805" "20180806" "20180807" "20180808" "20180809" "20180810" "20180811" "20180812" "20180813" "20180814" "20180815" "20180816" "20180817" "20180818" "20180819" "20190730" "20190731" "20190801" "20190802" "20190803" "20190804" "20190805" "20190806" "20190807" "20190808" "20190809" "20190810" "20190811" "20190812" "20190813" "20190814" "20190815" "20190816" "20190817" "20190818" "20190819" \
--new_init_date_strings "20210809" \
--input_region_mask_file_name="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/smaller_colorado_mask.nc" \
--target_field_name="fine_fuel_moisture_code" \
--use_inout_tensors_only=0 \
--disable_tensorflow2=1 \
--disable_eager_execution=1 \
--output_dir_name="${output_dir_name}"
