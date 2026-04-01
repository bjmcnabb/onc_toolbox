This repository provides Python scripts to:
1) Search for available instrument deployments on Ocean Networks Canada (ONC) Oceans 3.0 database by ocean properties (e.g 'salinity').
2) Order data products from the Oceans 3.0 server containing those properties. 

This toolbox allows the user to search for instrument deployments for a given time range and geographic extent, using property keywords (e.g. "salinity") or partial keywords (e.g. "sal"). Users can additionally define a polygon of geographic coordinates to further restrict the search area. Data orders will also automatically generate a metadata csv file, and the `write_shp` function can subsequently generate shapefiles for metadata input into GIS software. 

## Example use case (see "toolbox_example.py"):
### 1. Define the user ID and search parameters
```
outPath = r'C:/.../ONC_data/' # choose output directory
token = '...' # insert your 36 character token
bounds = [-135, -123, 46, 56] # [min longitude, max longitude, min latitude, max laitude] - e.g. constrain search to the NE Pacific (west coast of Vancouver Island)
dateFrom = '2021-01-01T00:00:00.000Z' # start date timestamp
dateTo = '2022-12-31T23:59:59.000Z' # end date timestamp
extension = 'csv' # data filetype to retrieve; other common extensions are "nc", "mp4", "mat", "txt"
```

### 2. Initialize server connection to Oceans 3.0 first
```
server = onc_toolbox(token=token, outPath=outPath)
```
### 3. Define a list of ocean properties to search by
This toolbox is designed to search by specific ocean properties, so define a list of properties to search by:
```
properties = [
    'seawatertemperature',
    'salinity',
    'currentdirection',
    'currentvelocity',
    'vectordirection',
]
```

If unsure what properties are available on Oceans 3.0, the `get_properties` funtion will retrieve a list of available properties. Users can narrow the retrieved list down by supplying a list of keywords (or partial matches) to filter by with the `prop_keyword` attribute (e.g. 'sal' for 'salinity'):
```
props, prop_details = server.get_properties(
    # prop_keyword=prop_keyword, # optional - filter results by keyword
    )
```
`prop` will return a Pandas Series containing the all property names on the Oceans 3.0 server, and `prop_details` will provide a Pandas DataFrame containing metadata for each property.

### 4. Find the available datasets by property
Since Oceans 3.0 only allows one property search at a time, iterate to return "devices" (instruments) by property inside a dict:
```
devices = {}
print('\nretrieving devices...')
for prop_ in properties:
    params = {'propertyCode': f'{prop_}', 'token': token}
    devices[prop_] = server.get_devices(params=params)
```

Users can optionally further narrow the list of devices down by supplying geographic bounds or even a custom bounding polygon:
```
selected_devices = {}
for prop_ in devices.keys():
    selected_devices[prop_] = server.select_devices(all_devices=devices[prop_],
                                                    bounds=bounds,
                                                    polygon_coords=polygon_coords,
                                                    )
```

Device locations can also be visualized using the `map_selected_devices` function, which produces a Cartopy map:
```
for prop_ in devices.keys():
    fig, ax = server.map_selected_devices(all_devices=devices[prop_],
                                          selected_devices=selected_devices[prop_],
                                          bounds=bounds)
    ax.set_title(f'property: {prop_}')
```

### 5. Finally, order a data product containing the selected devices
The following will download the datasets to the user-specified subfolder:
```
server.order_data(selected_devices=selected_devices, # or devices, to return all available devices found for a property
                  dateFrom=dateFrom,
                  dateTo=dateTo,
                  token=token,
                  # device_keyword=device_keyword,
                  extension=extension,
                  use_subfolders=True,
                  )
```
