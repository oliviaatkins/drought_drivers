"""
Attempt to download SMAP data to locally-saved NumPy arrays using Google Earth Engine.
"""
import ee
import numpy as np
import time
from datetime import date, timedelta

# Record the start time (bonus step if interested in how long things take to run)
start_time = time.time()

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='atkins-droughts') # <-- Edit project name to the one which your account is linked

# Import grid to iterate over and any region shapefile for clipping
# A grid is super useful as the code will iterate over each cell in turn, which reduces the GEE memory usage (which is limited)
final_grid = ee.FeatureCollection('projects/atkins-droughts/assets/final_grid')
final_shp = ee.FeatureCollection('projects/atkins-droughts/assets/final_shp')
    
# Consistent Parameters
#years = range(2015, 2025) # 2015 to 2024
#months = range(1, 13) # 1 to 12
#days = range(1, 32) # 1 to 31

years = range(2016, 2017) # 2015 to 2024
months = range(1, 2) # 1 to 12
days = range(1, 4) # 1 to 31

# Select band names for processing, same as the names within the GEE ERA5-Land catalog
# Include parameters which will be needed to process other variables (eg for wind speed and relative humidity)
selected_bands = ['sm_surface','sm_rootzone']

# --- Function to generate an ee.Image from the ERA5-Land collection in GEE ---
def create_image(year, month, day, selected_bands):
    # Loop over each day using timedelta
    # Even though we only consider one day in turn, the end_date is the following day due to standard Python reasoning
    start = date(year, month, day)
    end = start + timedelta(days=1)
    start_date = start.isoformat()
    end_date = end.isoformat()

    # Load in each SMAP image for a day
    # This will initially be as an ImageCollection with 1 image - call using .first(). It's stupid logic but welcome to Earth Engine.
    # If interested in other available datasets (eg ERA5) change the ee.ImageCollection root as per the catalog and go from there
    smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(start_date, end_date)
    image = smap.first()

    # Reproject image
    image = image.reproject(crs='EPSG:4326', scale=4000) # <-- Edit CRS and projection scale (in metres) as necessary
    
    # Select specific bands
    image = image.select(selected_bands) # This means GEE will only consider the relevant bands rather than all 150

    # Fill missing data with -9999, clip and update the mask to another image 
    # Clipping and updating the mask is in effect a double-double check, 
    # not strictly necessary but we want all the arrays to be the same length otherwise we got big issues
    image = image.unmask(-9999, sameFootprint=True).clip(final_shp) #.updateMask(aspect)
    
    # Additional step for saving as GeoTIFF file only
    image = image.set({
        'year': year,
        'month': month,
        'day': day,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis()
    })
    return image

# Choosen final bands for processing to arrays - ignore additional bands
final_bands = ['sm_surface','sm_rootzone']

# --- Process each band in the image to an individual array ---
for year in years:
    for month in months:
        for day in days:
            try:
                image = create_image(year, month, day, selected_bands)
                extracted_data = final_grid.map(lambda cell: cell.set(image.reduceRegion( # This is where the grid is useful to iterate through the cells
                    reducer=ee.Reducer.toList(), # different Reducers are available for different tasks. Here is .toList()
                    geometry=cell.geometry(),
                    scale=4000, # <-- Edit as necessary
                    bestEffort=False, # bestEffort=False ensures that pixels at the edge of the cell aren't rounded
                    crs='EPSG:4326', # <-- Edit as necessary
                    maxPixels=1e9
                )))

                # 'features' is the GEE way of calling each band
                # .getInfo() is a common term for viewing/loading the separate parts of each GEE-formatted file
                features = extracted_data.getInfo()['features']
                concatenated_data = {band: [] for band in final_bands}
                
                # Process each band to array, converting the -9999 back to NaN (this is necessary for tree-based models)
                for feature in features:
                    for band in final_bands:
                        data = np.array(feature['properties'].get(band, []))
                        data = np.where(data == -9999, np.nan, data)
                        concatenated_data[band].extend(data)

                # Reshape the array into the correct shape and save for each individual band
                for band in final_bands:
                    final_band_array = np.array(concatenated_data[band]).reshape(-1, 1)
                    filename = f'/home/olivia/Flash_Droughts_Wildfires/SMAP_GEE/data/processed_{year}_{month:02d}_{day:02d}_{band}_array.npy' # <-- Edit as necessary
                    np.save(filename, final_band_array)
                    print(f"Saved: {filename} with shape {final_band_array.shape}")
                    
            except Exception as e:
                # Erroneous dates, such as "30th Feb" will appear as errors but the code will automatically continue. This is a good sanity check
                print(f"Error processing {year}-{month}-{day}: {e}") 

# Calculate the time difference (bonus step as with the start time)
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")
