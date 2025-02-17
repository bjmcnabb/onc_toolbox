
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:52:37 2025

@author: bcamc
"""

from onc.onc import ONC
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from IPython import get_ipython
import os
import time

class onc_toolbox():
    def __init__(self, token, outPath, showInfo=False):
        # initialize a server connection with Oceans 3.0 and define download directory
        self.onc_ = ONC(token=token, outPath=outPath, showInfo=showInfo)
        self.rootPath = outPath
        super(onc_toolbox, self).__init__()

    def get_properties(self, prop_keyword=None):
        """
        Get all available property names from Oceans 3.0, and optionally filter these by a keyword.

        Parameters
        ----------
        prop_keyword : str, optional
            a keyword to find all relevant unique property names within available devices (e.g. "temp" for properties associated with temperature). The default is None.

        Returns
        -------
        props : Series
            Series of property names only.
        prop_details : Dataframe
            Dataframe containing property names and related metadata.

        """
        # pulls all available properties in the database
        properties = self.onc_.getProperties()
        prop_details = pd.DataFrame(properties)
        # extract out just the property names ('codes')
        props = prop_details.loc[:,'propertyCode'].squeeze()
        if prop_keyword is not None:
            props = props[props.str.contains(prop_keyword)==True] # filter property names that contain keyword
        return props, prop_details
    
    def get_devices(self, params):
        """
        Get all available devices from the Oceans 3.0 server.

        Parameters
        ----------
        params : dict
            dictionary containing "propertyCode" and download "token".

        Returns
        -------
        all_devices : Dataframe
            Dataframe containing all devices, with device codes and other metadata.

        """
        # get all all_devices names and attributes from database
        all_devices = self.onc_.getDeployments(params)
        all_devices = pd.DataFrame(all_devices)
        
        # apply longitude conversion degree minute -> decimal degree format for relevant coords 
        # at least one device is mislabelled in the directory
        for ind in all_devices[(all_devices.loc[:,'lon']<=-180) | (all_devices.loc[:,'lon']>=180)].index:
            lon = str(all_devices.loc[ind, 'lon'])
            if float(lon)<0:
                all_devices.loc[ind, 'lon'] = float(lon.split('.')[0][:-2]) - (float(lon.split('.')[0][-2:])+float('0.'+lon.split('.')[1]))/60
            if float(lon)>0:
                all_devices.loc[ind, 'lon'] = float(lon.split('.')[0][:-2]) + (float(lon.split('.')[0][-2:])+float('0.'+lon.split('.')[1]))/60
        
        # apply latitude conversion degree minute -> decimal degree format for coords
        # at least one device is mislabelled in the directory
        for ind in all_devices[(all_devices.loc[:,'lat']<=-90) | (all_devices.loc[:,'lat']>=90)].index:
            lat = str(all_devices.loc[ind, 'lat'])
            if float(lat)<0:
                all_devices.loc[ind, 'lat'] = float(lat.split('.')[0][:-2]) - (float(lat.split('.')[0][-2:])+float('0.'+lat.split('.')[1]))/60
            if float(lat)>0:
                all_devices.loc[ind, 'lat'] = float(lat.split('.')[0][:-2]) + (float(lat.split('.')[0][-2:])+float('0.'+lat.split('.')[1]))/60
        
        # convert str to timestamps
        all_devices['begin'] = [pd.Timestamp(i) for i in all_devices.loc[:,'begin']]
        all_devices['end'] = [pd.Timestamp(i) for i in all_devices.loc[:,'end']]
        # remove any entries with missing coords
        all_devices = all_devices.dropna(subset=['lon','lat'])
        # re-number index
        all_devices = all_devices.reset_index().drop('index',axis=1)
        return all_devices
    
    def select_devices(self, all_devices, bounds, polygon_coords=None):
        """
        Filter devices by regional bounds. Additional filtering by a user defined area (polygon) is permitted, and useful for restricting the search area to non-orthongal bounds.

        Parameters
        ----------
        all_devices : Dataframe
            Dataframe of all available devices and related metadata, returned by the get_devices() function.
        bounds : list
            Geographic bounds, in form of [min_lon, max_lon, min_lat, max_lat].
        polygon_coords : list, optional
            List of tuples, where each tuple is coordinate pair (longitude, latitude) used to construct the search area polygon. The default is None.

        Returns
        -------
        selected_devices : Dataframe
            Dataframe of selected devices and relevant metadata.

        """
        # intial filter - restrict data to regional bounds
        min_lon, max_lon, min_lat, max_lat = bounds
        inds = np.argwhere((all_devices.loc[:,'lon'].values>=min_lon)\
                          & (all_devices.loc[:,'lon'].values<=max_lon)\
                              & (all_devices.loc[:,'lat'].values>=min_lat)\
                                  & (all_devices.loc[:,'lat'].values<=max_lat)).flatten()
        selected_devices = all_devices.loc[inds,:]
        
        # second filter (optional) - create user-defined polygon to subset device locations further
        # (useful if only interested in a local, non-orthogonal area of the region)
        if polygon_coords is not None:
            self.polygon = Polygon(polygon_coords)
            locs_ = [self.polygon.contains(Point(lon,lat)) for lon,lat in all_devices.loc[:,['lon','lat']].values]
            selected_devices = all_devices[locs_].copy()
        
        # reset index numbering
        selected_devices = selected_devices.reset_index().drop('index', axis=1)
        
        return selected_devices
    
    def map_selected_devices(self, all_devices, selected_devices, bounds, backend='inline'):
        """
        Function to visually map and check selected device coordinates are correct using matplotlib and cartopy.

        Parameters
        ----------
        all_devices : dataframe
            Dataframe of all devices available, returned by get_devices().
        selected_devices : dataframe
            Dataframe filtered spatially in select_devices().
        bounds : list
            Geographic bounds, in form of [min_lon, max_lon, min_lat, max_lat].
        backend : str, optional
            Matplotlib backend for rendering figures; for interactive maps, use 'qt'. The default is 'inline'.

        Returns
        -------
        fig : figure
            Matplotlib figure handle.
        ax : axes
            Matplotlib axes handle.

        """
        # extract out bounds
        min_lon, max_lon, min_lat, max_lat = bounds
        # for interactive/zoomable map, choose backend to be 'qt'
        get_ipython().run_line_magic('matplotlib',backend)
        
        # plot data
        fig = plt.figure(figsize=(12,6), dpi=300)
        map_proj = ccrs.PlateCarree()
        map_proj._threshold /= 100 # increases map resolution for drawing polygons
        ax = fig.add_subplot(111, projection=map_proj)
        
        # plot all data within geographical bounds
        ax.scatter(all_devices.loc[:,'lon'],
                    all_devices.loc[:,'lat'],
                    s=1 ,
                    c='r',
                    label='Available devices',
                    zorder=4,
                    transform=ccrs.PlateCarree())
        
        # plot filtered devices
        ax.scatter(selected_devices.loc[:,'lon'],
                    selected_devices.loc[:,'lat'],
                    s=1 ,
                    c='g',
                    label='User selected devices',
                    zorder=4,
                    transform=ccrs.PlateCarree())
        
        # plot and label nodes of polygon, if used
        if hasattr(self, 'polygon'):
            lons = self.polygon.exterior.xy[0]
            lats = self.polygon.exterior.xy[1]
            ax.plot(lons,
                    lats,
                    marker='s',
                    ms=2,
                    lw=0.2,
                    c='k',
                    label='Search polygon',
                    zorder=3,
                    transform=ccrs.PlateCarree())
            for (lon,lat) in zip(lons,lats):
                ax.text(lon,
                        lat+((max_lat-min_lat)*0.02),
                        f'({lon},{lat})',
                        ha='right',
                        va='center',
                        fontsize=5,
                        transform=ccrs.PlateCarree())
        
        # add some features and formatting
        gl = ax.gridlines(draw_labels=True, lw=0.1)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2, lw=0.1)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax.legend(fontsize=8, loc='upper right')
        return fig, ax
    
    def order_data(self, selected_devices, dateFrom, dateTo, token, device_keyword=None, extension='csv', use_subfolders=False):
        """
        Send a data order query to Oceans 3.0 servers. This is done iteratively,
        as the ONC API requires only one set of device name / location code / filetype
        be queried in a single call to the servers. The "deviceCategoryCodes" are extracted and
        filtered by the user selected time range ("dateFrom" to "dateTo") to further reduce 
        query numbers and limit file sizes. Devices can also be further filtered by a keyword 
        (e.g. "CTD" or "ADCP").

        Parameters
        ----------
        selected_devices : Dataframe
            Dataframe of devices to iterate through and download.
        dateFrom : str
            Timestamp of start date and time. Example formatting is "2021-10-22T20:00:00.000Z".
        dateTo : str
            Timestamp of end date and time. Example formatting is "2021-10-22T20:00:00.000Z".
        token : str
            Specific download token number for user (listed in Oceans 3.0 account).
        device_keyword : str, optional
            Keyword to restrict the number of devices further (e.g. only find devices with "ADCP" in name). The default is None.
        extension : str, optional
            Filetype extension; some examples include 'mat','nc','txt', and 'csv'. The default is 'csv'.
        use_subfolders: bool, optional
            Select whether to write data to subfolders labelled by property within a main folder, or to the main directory. 
            Note that if subfolders are used, duplicate datafiles across folders may result. the default is 'False'.
        includeMetadataFile : Bool, optional
            Specify whether to also download metadata. The default is 'False'.

        Returns
        -------
        None.

        """
        # iteratively download by property in devices
        for prop_ in selected_devices.keys():
            if use_subfolders is True:
                # handle paths - create a folder for each property inside the root path, if it doesn't exist
                if os.path.isdir(self.rootPath+'/'+f'{prop_}') is False:
                    # create folder to download files to
                    os.makedirs(self.rootPath+'/'+f'{prop_}')
                # update path
                self.onc_.outPath = self.rootPath+'/'+f'{prop_}'
            
            # first remove any duplicate entries based on following subsetted columns
            device_codes = selected_devices[prop_].loc[:,['begin','end','locationCode','deviceCategoryCode', 'depth', 'lat', 'lon']].copy().drop_duplicates()
            device_codes = device_codes.reset_index().drop('index', axis=1)
            
            # first pass filter (optional) - use keyword to find relevent devices
            if device_keyword is not None:
                device_codes = device_codes[device_codes.loc[:,'deviceCategoryCode'].str.contains(device_keyword)==True]
            
            # check that user-selected timestamps are within available date range
            device_codes = device_codes[(pd.Timestamp(dateFrom) >= device_codes.loc[:,'begin'])\
                                & (pd.Timestamp(dateTo) <= device_codes.loc[:,'end'])]
            device_codes = device_codes.reset_index().drop('index', axis=1)
            
            # quick bypass if devices fall outside temporal window (or devices do not contain keyword)
            if len(device_codes) != 0:
                for i in range(len(device_codes)):
                # i=1
                    download_params = {
                        'locationCode': device_codes.loc[:,'locationCode'].iloc[i],
                        'deviceCategoryCode': device_codes.loc[:,'deviceCategoryCode'].iloc[i],
                        'extension': extension,
                        'dateFrom':dateFrom,
                        'dateTo':dateTo,
                        "token": token,
                    }
                    
                    #### get the data product code
                    params = {
                        'locationCode': device_codes.loc[:,'locationCode'].iloc[i],
                        'deviceCategoryCode': device_codes.loc[:,'deviceCategoryCode'].iloc[i],
                        "extension": extension,
                    }
                    # if extension does not exist for date range, will return an error
                    try:
                        data_prods = self.onc_.getDataProducts(params)
                        
                        #### download data
                        download_params['dataProductCode'] = [j['dataProductCode'] for j in data_prods][0] # get product code
                        # use try/except to bypass internal server error (500) raised sometimes
                        try:
                            # order data
                            order = self.onc_.orderDataProduct(download_params, includeMetadataFile=False)
                            # write metadata geodataframe if order is complete
                            if len(order['downloadResults'][0]) > 0 and order['downloadResults'][0]['status'] != 'skipped':
                                filename_parts = order['downloadResults'][0]['file'].split('.')
                                meta_filename = filename_parts[0]+'_metadata.csv'
                                self._create_metadata_from_order(filename=meta_filename,
                                                                i=i,
                                                                device_codes=device_codes,
                                                                dateFrom=dateFrom,
                                                                dateTo=dateTo)
                        except:
                            pass
                        time.sleep(1) # this seems to catch errors associated with breaking and starting a new server call
                    except:
                        pass
                else:
                    print(f'{prop_}: no files found for selected data range')
        
    def _find_metadata_vals(self, df, keyword, sep=' ', dtype=str, ix=1):
        """
        Internal function to parse and extract metadata from ONC csv headers. 

        Parameters
        ----------
        df : Dataframe
            Dataframe read by pandas read_fwf() function.
        keyword : str
            The leading keyword of the metadata string that is the search target.
        sep : str, optional
            String seperator to split keyword from rest of metadata in searched string. The default is ' '.
        dtype : dtype, optional
            Specify a datatype to convert metadata output string to (e.g. float). The default is str.
        ix : int, optional
            Indexing integer of parsed metadata string. The default is 1.

        Returns
        -------
        val : str, float or other dtype
            Parsed metadata.

        """
        val = df[df.iloc[:,0].str.contains(keyword)==True].iloc[0,0]
        val = val.split(sep)
        val = list(filter(None,val))[ix]
        if dtype is float:
            val = dtype(val)
        return val

    def create_metadata_from_csv(self, inDir, inFilename):
        """
        Generate a GeoDataFrame containing metadata extracted from data csv file headers.

        Parameters
        ----------
        inDir : str
            Directory path to data file.
        inFilename : str
           Data filename.

        Returns
        -------
        metadata : GeoDataFrame
            GeoDataFrame of metadata.

        """
        # quick check directories include seperator
        if (inDir[-1] != '/') or (inDir[-2:] != '\\'):
            inDir = inDir+'/'
        # read in datafile to get header info
        df = pd.read_fwf(inDir+inFilename)
        # extract coords
        lat = self._find_metadata_vals(df, keyword='#LAT', dtype=float)
        lon = self._find_metadata_vals(df, keyword='#LON', dtype=float)
        # create metadata dict
        metadata = {
            'geometry': gpd.points_from_xy([lon], [lat], crs='WGS84'),
            'stn': self._find_metadata_vals(df, keyword='#STNCODE'),
            'depth': self._find_metadata_vals(df, keyword='#DEPTH', dtype=float),
            'datefrom': self._find_metadata_vals(df, keyword='#DATEFROM'),
            'dateto': self._find_metadata_vals(df, keyword='#DATETO'),
            'devcat': self._find_metadata_vals(df, keyword='#DEVCAT', sep=': '),
            'QC': self._find_metadata_vals(df, keyword='#"QC Flag', sep=': '),
            'data_loc': inDir+inFilename,
            # 'summary_plot_location':,
            }
        
        # build geopandas df
        metadata = gpd.GeoDataFrame(metadata)
        return metadata
    
    def _create_metadata_from_order(self, filename, i, device_codes, dateFrom, dateTo):
        """
        Internal function to generate, and write to file, a metadata GeoDataFrame corresponding to the downloaded data from Oceans 3.0.

        Parameters
        ----------
        filename : str
            filename.
        i : int
            Current index of device codes.
        device_codes : Dataframe
            Dataframe of devices to download.
        dateFrom : str
            Timestamp for start of search range.
        dateTo : str
            Timestamp for end of search range.

        Returns
        -------
        None.

        """
        # quick check that metadata folder exists
        savepath = os.path.join(self.onc_.outPath, 'metadata')
        if os.path.isdir(savepath) is False:
            # if not, create folder to download files to
            os.makedirs(savepath)
        
        # write path
        path = os.path.join(self.onc_.outPath, 'metadata/'+filename)
        
        # create metadata dict
        metadata = {
            'geometry': gpd.points_from_xy([device_codes.loc[:,'lon'].iloc[i]],
                                           [device_codes.loc[:,'lat'].iloc[i]], crs='WGS84'),
            'stn': device_codes.loc[:,'locationCode'].iloc[i],
            'depth': device_codes.loc[:,'depth'].iloc[i],
            'datefrom': dateFrom,
            'dateto': dateTo,
            'devcat':device_codes.loc[:,'deviceCategoryCode'].iloc[i],
            'data_loc': path,
            }
        
        # build geopandas df
        metadata = gpd.GeoDataFrame(metadata)
        metadata.to_csv(path)
        
def write_shp(metadata, outDir, outFilename):
    """
    write shapefile from metadata GeoDataFrame.

    Parameters
    ----------
    metadata : GeoDataFrame
        metadata extracted from csv file headers using create_metadata().
    outDir : str
        Directory path to write file to.
    outFilename : str
        Filename for shapefiles.

    Returns
    -------
    None.

    """
    # quick check directories include seperator
    if (outDir[-1] != '/') or (outDir[-2:] != '\\'):
        outDir = outDir+'/'
     # handle paths - create a folder for each property inside the root path, if it doesn't exist
    if os.path.isdir(outDir) is False:
         # create folder to download files to
         os.makedirs(outDir)
    # write shpfile
    metadata.to_file(outDir+outFilename, driver='ESRI Shapefile')

    