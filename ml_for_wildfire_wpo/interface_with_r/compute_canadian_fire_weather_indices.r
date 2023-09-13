# For each day in a consecutive time period, computes Canadian FWIs on a grid.

library(cffdrs)
require(raster)

INPUT_DIR_NAME <- '/home/ralager/ml_for_wildfire_wpo_stuff/processed_era5/tif_format'
OUTPUT_DIR_NAME <- '/home/ralager/ml_for_wildfire_wpo_stuff/processed_era5/tif_format/with_fire_weather_indices'

START_DATE_OBJECT <- as.Date('2018-01-02')
END_DATE_OBJECT <- as.Date('2022-12-30')

if (!file.exists(OUTPUT_DIR_NAME)) {
  dir.create(OUTPUT_DIR_NAME, recursive=TRUE)
}

current_date_object <- START_DATE_OBJECT
previous_date_fwi_raster_stack <- NULL

while (current_date_object <= END_DATE_OBJECT) {
  date_string <- format(current_date_object, '%Y%m%d')
  input_file_name <- sprintf('%s/era5_%s.tif', INPUT_DIR_NAME, date_string)
  log_message <- sprintf('Reading data from: "%s"...', input_file_name)
  print(log_message)

  current_date_era5_raster_stack <- stack(input_file_name)
  names(current_date_era5_raster_stack) <- c('lat', 'temp', 'rh', 'ws', 'prec')
  
  current_month_string <- substr(date_string, 5, 6)
  current_month <- as.integer(current_month_string)
  
  if (is.null(previous_date_fwi_raster_stack)) {
    current_date_fwi_raster_stack <- fwiRaster(current_date_era5_raster_stack, mon=current_month, lat.adjust=TRUE, uppercase=FALSE)
  } else {
    current_date_fwi_raster_stack <- fwiRaster(current_date_era5_raster_stack, init=previous_date_fwi_raster_stack, mon=current_month, lat.adjust=TRUE, uppercase=FALSE)
  }
  
  output_file_name <- sprintf('%s/era5_%s_fwi.tif', OUTPUT_DIR_NAME, date_string)
  log_message <- sprintf('Writing data to: "%s"...', output_file_name)
  print(log_message)
  
  writeRaster(current_date_fwi_raster_stack, filename=output_file_name, format='GTiff', overwrite=TRUE)
  
  current_date_object <- current_date_object + 1
  previous_date_fwi_raster_stack = current_date_fwi_raster_stack
  # previous_date_fwi_raster_stack = dget(dput(current_date_fwi_raster_stack))  # Deep copy.
}
