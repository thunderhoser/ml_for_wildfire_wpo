#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

set GFS_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/processed/merged/normalized_params_from_2019-2020"
set TARGET_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/canadian_fwi"
set GFS_FORECAST_TARGET_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/direct_fwi_calc/processed/daily/fwi"

python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
--input_model_file_name="${model_dir_name}/model.h5" \
--input_gfs_directory_name="${GFS_DIR_NAME}" \
--input_target_dir_name="${TARGET_DIR_NAME}" \
--input_gfs_fcst_target_dir_name="${GFS_FORECAST_TARGET_DIR_NAME}" \
--init_date_limit_strings "20210101" "20211210" \
--output_dir_name="${model_dir_name}/validation"
