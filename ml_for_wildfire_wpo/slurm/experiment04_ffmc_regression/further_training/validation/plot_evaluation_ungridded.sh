#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

python3 -u "${CODE_DIR_NAME}/plot_ungridded_evaluation.py" \
--input_eval_file_names "${model_dir_name}/validation/ungridded_evaluation.nc" \
--line_styles "solid" \
--line_colours "217_95_2" \
--set_descriptions "model" \
--plot_full_error_distributions=1 \
--confidence_level=0.95 \
--report_metrics_in_titles=1 \
--output_dir_name="${model_dir_name}/validation/ungridded_evaluation"
