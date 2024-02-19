#!/bin/sh

template_file_name=$1
output_dir_name=$2
batch_size=$3

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

GFS_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed/normalized_params_from_2019-2020"
TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi"
GFS_FORECAST_TARGET_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed_fwi_forecasts"
TARGET_NORM_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi/z_score_params_2019-2020.nc"

# Number of 8-sample batches is 8, 16, 24, or 32 -- corresponding to {1, 2, 3, 4} gradient-accumulation steps, respectively, for optimizer.

python3 -u "${CODE_DIR_NAME}/train_neural_net.py" \
--input_template_file_name="${template_file_name}" \
--output_model_dir_name="${output_dir_name}" \
--inner_latitude_limits_deg_n 17 73 \
--inner_longitude_limits_deg_e 171 -65 \
--outer_latitude_buffer_deg=5 \
--outer_longitude_buffer_deg=5 \
--gfs_predictor_field_names "temperature_kelvins" "specific_humidity_kg_kg01" "geopotential_height_m_asl" "u_wind_m_s01" "v_wind_m_s01" "temperature_2m_agl_kelvins" "specific_humidity_2m_agl_kg_kg01" "u_wind_10m_agl_m_s01" "v_wind_10m_agl_m_s01" "accumulated_precip_metres" "volumetric_soil_moisture_fraction_0to10cm" "snow_depth_metres" \
--gfs_pressure_levels_mb 500 700 900 \
--gfs_predictor_lead_times_hours 0 12 24 36 48 60 72 \
--gfs_normalization_file_name="" \
--era5_constant_file_name="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/era5_constants.nc" \
--target_field_names "fine_fuel_moisture_code" "buildup_index" \
--target_lead_time_days=2 \
--target_lag_times_days 1 2 3 \
--gfs_forecast_target_lead_times_days 1 2 3 \
--target_normalization_file_name="${TARGET_NORM_FILE_NAME}" \
--num_examples_per_batch=16 \
--sentinel_value=-10 \
--gfs_dir_name_for_training="${GFS_DIR_NAME}" \
--target_dir_name_for_training="${TARGET_DIR_NAME}" \
--gfs_forecast_target_dir_name_for_training="${GFS_FORECAST_TARGET_DIR_NAME}" \
--gfs_init_date_limit_strings_for_training "20190101" "20201210" \
--gfs_dir_name_for_validation="${GFS_DIR_NAME}" \
--target_dir_name_for_validation="${TARGET_DIR_NAME}" \
--gfs_forecast_target_dir_name_for_validation="${GFS_FORECAST_TARGET_DIR_NAME}" \
--gfs_init_date_limit_strings_for_validation "20210101" "20211210" \
--num_epochs=1000 \
--num_training_batches_per_epoch=${batch_size} \
--num_validation_batches_per_epoch=8
