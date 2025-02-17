#%% Example use case

from ONC_toolbox import onc_toolbox, write_shp
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import os

#### Define constants and search parameters
outPath = r'C:/Users/...' # choose output directory
token = '...' # insert your 36 character token
bounds = [-135, -123, 46, 56] # NE Pacific, west coast of Vancouver Island
# nodes to define a search area polygon - I've chosen here coordinates that capture the outflow of Juan de Fuca strait, and the northern coast near Haida Gwaii
polygon_coords = [
    (-127.5, 51),
    (-124.8, 49.35),
    (-123.6, 48.5),
    (-123.5, 47.97),
    (-127.5, 46.5),
    (-133.5, 51),
    (-133.5, 55.5),
    (-130.5, 55.5),
    (-127.5, 54),
]

# main search parameters for files
dateFrom = '2021-10-22T20:00:00.000Z'
dateTo = '2021-10-29T20:00:00.000Z'
extension = 'csv' # other common extensions are "nc", "mp4", "mat", "txt"

# optional parameters - un-comment these inputs in the relevant functions below
device_keyword = 'ADCP'
prop_keyword = 'pres' 

#%% Download data

# initialize server connection to Oceans 3.0 first
server = onc_toolbox(token=token, outPath=outPath)

# get list of property values to search by
props, prop_details = server.get_properties(
    # prop_keyword=prop_keyword, # optional - filter results by keyword
    )

# set properties to search and retrieve devices
properties = [
    'pressure',
    'seawatertemperature',
    'salinity',
    'currentdirection',
    'currentvelocity'
]

# ONC only allows one property search at a time - iteratively return devices by property in dict
devices = {}
print('\nretrieving devices...')
for prop_ in properties:
    params = {'propertyCode': f'{prop_}', 'token': token}
    devices[prop_] = server.get_devices(params=params)

# subset devices by chosen geo. bounds and polygon
selected_devices = {}
for prop_ in devices.keys():
    selected_devices[prop_] = server.select_devices(all_devices=devices[prop_],
                                                    bounds=bounds,
                                                    polygon_coords=polygon_coords,
                                                    )

# visualize device locations
for prop_ in devices.keys():
    fig, ax = server.map_selected_devices(all_devices=devices[prop_],
                                          selected_devices=selected_devices[prop_],
                                          bounds=bounds)
    ax.set_title(f'property: {prop_}')


# order data using selected devices
server.order_data(selected_devices=selected_devices,
                  dateFrom=dateFrom,
                  dateTo=dateTo,
                  token=token,
                  # device_keyword=device_keyword,
                  extension=extension,
                  use_subfolders=True,
                  )

#%% Create shapefiles from GeoDataFrames generated during data download

# compile GeoDataFrames from subfolders
folders = os.listdir(outPath)
metadata = {}
for folder in folders:
    path = outPath+os.sep+folder+os.sep+'metadata'
    files = os.listdir(path)
    for i, file in enumerate(files):
        metadf = gpd.read_file(path+os.sep+file,GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
        metadf = metadf.drop('field_1', axis=1)
        if i==0:
            metadfs = metadf
        else:
            metadfs = pd.concat([metadfs, metadf], axis=0)
    metadata[folder] = metadfs
                    
# write shapefiles from GeoDataFrames
for key in metadata.keys():
    write_shp(metadata[key],
              outPath+'/'+key+'/'+'shapefiles',
              key+'.shp')

