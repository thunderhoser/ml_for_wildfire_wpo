# For each 0000 UTC model run in a consecutive period, converts GFS weather forecasts to explicit FWI forecasts.

library(cffdrs)
require(raster)

DAILY_GFS_DIR_NAME <- '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/direct_fwi_calc/processed/daily/tif'
TOP_OUTPUT_DIR_NAME <- '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_project/gfs_data/direct_fwi_calc/processed/daily/tif_with_fwi'
ALL_LEAD_TIMES_DAYS <- c('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14')

argument_list = commandArgs(trailingOnly=TRUE)
first_init_date_string <- argument_list[1]
last_init_date_string <- argument_list[2]

first_init_date_object <- as.Date(first_init_date_string)
last_init_date_object <- as.Date(last_init_date_string)
init_date_object <- first_init_date_object

while (init_date_object <= last_init_date_object) {
  init_date_string <- format(init_date_object, '%Y%m%d')
  input_file_name <- sprintf(
    '%s/init=%s/gfs_fwi_inputs_init=%s_lead=00days.tif',
    DAILY_GFS_DIR_NAME, init_date_string, init_date_string
  )
  
  if (!file.exists(input_file_name)) {
    init_date_object <- init_date_object + 1
    next
  }

  log_message <- sprintf('Reading data from: "%s"...', input_file_name)
  print(log_message)

  prev_fcst_day_fwi_raster_stack <- stack(input_file_name)
  names(prev_fcst_day_fwi_raster_stack) <- c('lat', 'temp', 'rh', 'ws', 'prec', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi', 'dsr')

  for (lead_time_days in ALL_LEAD_TIMES_DAYS) {
    input_file_name <- sprintf(
      '%s/init=%s/gfs_fwi_inputs_init=%s_lead=%s',
      DAILY_GFS_DIR_NAME, init_date_string, init_date_string, lead_time_days
    )
    input_file_name <- paste(input_file_name, 'days.tif', sep='')
    
    log_message <- sprintf('Reading data from: "%s"...', input_file_name)
    print(log_message)
    
    current_fcst_day_gfs_raster_stack <- stack(input_file_name)
    names(current_fcst_day_gfs_raster_stack) <- c('lat', 'temp', 'rh', 'ws', 'prec')

    valid_date_object <- init_date_object + strtoi(lead_time_days, base=10L)
    valid_date_string <- format(valid_date_object, '%Y%m%d')
    current_month_string <- substr(valid_date_string, 5, 6)
    current_month <- as.integer(current_month_string)
    
    log_message <- sprintf('Valid date = %s; month = %s', valid_date_string, current_month_string)
    print(log_message)

    current_fcst_day_fwi_raster_stack <- fwiRaster(current_fcst_day_gfs_raster_stack, init=prev_fcst_day_fwi_raster_stack, mon=current_month, lat.adjust=TRUE, uppercase=FALSE)

    output_dir_name <- sprintf('%s/init=%s', TOP_OUTPUT_DIR_NAME, init_date_string)
    if (!file.exists(output_dir_name)) {
      dir.create(output_dir_name, recursive=TRUE)
    }

    output_file_name <- sprintf(
      '%s/gfs_fwi_forecast_init=%s_lead=%s',
      output_dir_name, init_date_string, lead_time_days
    )
    output_file_name <- paste(output_file_name, 'days.tif', sep='')

    log_message <- sprintf('Writing data to: "%s"...', output_file_name)
    print(log_message)

    writeRaster(current_fcst_day_fwi_raster_stack, filename=output_file_name, format='GTiff', overwrite=TRUE)
    prev_fcst_day_fwi_raster_stack = current_fcst_day_fwi_raster_stack
  }

  init_date_object <- init_date_object + 1
}
