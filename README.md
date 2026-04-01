This repository provides Python scripts built on Ocean Networks Canada API to search for available devices by property and order data products from the Oceans 3.0 server. This toolbox of functions allows the user to search the Oceans 3.0 database to find instrument deployments for a given time range and geographic extent using property keywords (e.g. "salinity") or partial keywords (e.g. "sal"). Users can additionally define a polygon of geogrpahic coordiantes to further restrict the search area. Data orders will also automatically generate a metadata csv file, and the "write_shp" function can be run to subsequently generate shapefiles for metadata input into GIS software. 
<div>
<p>&nbsp;  </p>
</div>

###Example use case (see "toolbox_example.py"):
#Define the user ID and search parameters
```
outPath = r'C:/.../ONC_data/' # choose output directory
token = '...' # insert your 36 character token
bounds = [-135, -123, 46, 56] # constrain search to the NE Pacific (west coast of Vancouver Island)
dateFrom = '2021-01-01T00:00:00.000Z' # start date timestamp
dateTo = '2022-12-31T23:59:59.000Z' # end date timestamp
```

initialize server connection to Oceans 3.0 first
```
server = onc_toolbox(token=token, outPath=outPath)
```

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

If unsure what properties are available on Oceans 3.0, the `get_properties` will retrieve a list of available properties. Users can narrow the retrieved list down by supplying a list of keywords or partial matches to filter by with the `prop_keyword` attribute (e.g. 'temp'):
```
props, prop_details = server.get_properties(
    # prop_keyword=prop_keyword, # optional - filter results by keyword
    )
```

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

Finally, device locations can be visualized using a Cartopy map:
```
for prop_ in devices.keys():
    fig, ax = server.map_selected_devices(all_devices=devices[prop_],
                                          selected_devices=selected_devices[prop_],
                                          bounds=bounds)
    ax.set_title(f'property: {prop_}')
```

