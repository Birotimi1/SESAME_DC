# %%
import pathlib
import obspy

# %%
import numpy as np
from obspy import read
import os
import math
from math import exp
from matplotlib import cm

import pandas as pd
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import hvsrpy
from hvsrpy import utils

# %%
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from pathlib import Path

# %%
plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

# %%
preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
preprocessing_settings.detrend = "linear"
preprocessing_settings.window_length_in_seconds = 300
preprocessing_settings.orient_to_degrees_from_north = 0.0
preprocessing_settings.filter_corner_frequencies_in_hz=(None, None)

print("Preprocessing Summary")
print("-"*300)
preprocessing_settings.psummary()

# %%
processing_settings = hvsrpy.settings.HvsrDiffuseFieldProcessingSettings()
processing_settings.window_type_and_width = ("tukey", 0.1)
processing_settings.smoothing=dict(operator="log_rectangular",
                                   bandwidth=0.1,
                                #    center_frequencies_in_hz=np.geomspace(0.2, 50, 256))
                                   center_frequencies_in_hz=np.geomspace(0.2, 25, 256))

print("Processing Summary")
print("-"*60)
processing_settings.psummary()

# %%
file_path = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/"
file_name = file_path + "Z9.D18.2013-05-11.2013-05-12.mseed"

pathlist = Path(file_path).glob('**/*.mseed')
for path in pathlist:
    print (path)

    st = read (path)
    sr = st[0].stats.sampling_rate
    print(sr)



# %%


# %%


file_path = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/"

pathlist = list(Path(file_path).glob('**/*.mseed'))

print(f"Number of recordings: {len(pathlist)}")
for path in pathlist:
    if not path.exists():
        raise FileNotFoundError(f"file {path} not found; check spelling.")
print("All files exist.")


# %%


# %%


# %%
srecords = hvsrpy.read(str(path))
srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
hvsr = hvsrpy.process(srecords, processing_settings)

# %%


# %%


# %%
## New version

import os
from pathlib import Path
import matplotlib.pyplot as plt
import hvsrpy
from obspy import read
import numpy as np

# Define the base directories for figures and HVSR results
fig_base_out = './outputhvsrloop/figures/'
hvsrpy_base_out = './outputhvsrloop/hvsrpy/'

# Ensure the base directories exist
os.makedirs(fig_base_out, exist_ok=True)
os.makedirs(hvsrpy_base_out, exist_ok=True)

file_path = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/"

# Get all mseed files
pathlist = sorted(Path(file_path).glob('**/*.mseed'))

 # Skip download if file already exists


for path in pathlist:
    print(path)
    st = read(path)
    sr = st[0].stats.sampling_rate
    print(sr)


    
    # Check and correct channel names
    isoverwrite = False
    for tr in st:
        if tr.stats.channel.endswith('1'):
            tr.stats.channel = tr.stats.channel.replace('1', 'N')
            tr.meta.channel = tr.meta.channel.replace('1', 'N')
            isoverwrite = True
        elif tr.stats.channel.endswith('2'):
            tr.stats.channel = tr.stats.channel.replace('2', 'E')
            tr.meta.channel = tr.meta.channel.replace('2', 'E')
            isoverwrite = True
    if isoverwrite:
        st.write(str(path), format="MSEED")

    # Define HVSR processing settings
    processing_settings = hvsrpy.settings.HvsrDiffuseFieldProcessingSettings()
    processing_settings.window_type_and_width = ("tukey", 0.1)
    processing_settings.smoothing = dict(
        operator="log_rectangular",
        bandwidth=0.1,
        center_frequencies_in_hz=np.geomspace(0.2, sr / 2, 256)
    )

    print("Processing Summary")
    print("-" * 60)
    processing_settings.psummary()

    # Read and process the seismic records
    srecords = hvsrpy.read(str(path))
    hvsr = hvsrpy.process(srecords, processing_settings)

    print("\nStatistical Summary:")
    print("-" * 20)
    hvsrpy.summarize_hvsr_statistics(hvsr)

    fig, ax = hvsrpy.plot_single_panel_hvsr_curves(hvsr)
    plt.show()

    # Use full path to get network.station name and day number
    file_toks = str(path).split('/')[-1].split('.')
    station_str = file_toks[0]+'.'+file_toks[1]
    day_num = file_toks[2]
    
    stadir_fig = fig_base_out+'/'+station_str+'/'
    stadir_hvsrpy = hvsrpy_base_out+'/'+station_str+'/'
    
    # Ensure all necessary folders exist
    os.makedirs(stadir_fig, exist_ok=True)
    os.makedirs(stadir_hvsrpy, exist_ok=True)

    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping download.")
        continue

    # Figure and file output names
    figure_name_out = stadir_fig+station_str+'_'+day_num+'_hvsr_figure_az.png'
    file_name_out_hvsrpy = stadir_hvsrpy+'/'+station_str+'_'+day_num+'_output_hvsrpy_az.csv'

    # Save figure in the station's day figure folder
    fig.savefig(figure_name_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved successfully to {figure_name_out}!")

    # Save HVSR results in the station's day HVSR folder
    hvsrpy.object_io.write_hvsr_object_to_file(hvsr, file_name_out_hvsrpy)
    print(f"Results saved successfully to {file_name_out_hvsrpy}!")

print("Processing complete.")

# %%
sta = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/Z9.W15B/hour14/Z9.W15B.hour14.mseed"
#sta= "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/Z9.W15B/hour13/Z9.W15B.hour13.mseed"

# %%
from obspy import read

st = read(sta)
sr = st[0].stats.sampling_rate

st.plot()

# %%
sta= "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata/Z9.W15B/hour13/Z9.W15B.hour13.mseed"

# %%
from obspy import read

st = read(sta)
sr = st[0].stats.sampling_rate

st.plot()

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import hvsrpy

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

plt.rcParams.update({'font.size': 16}) # set a larger font size

# %%
csv_files= '/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/outputhvsrloop/hvsrpy/'

# %%
print (len(csv_files))

# %%
# Load topography data
#path2grdfile = '/Users/russell/Lamont/GMT/grids/etopo1.grd'
path2grdfile = '/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/etopo1.grd'
etopodata = Dataset(path2grdfile)
lons_e = np.linspace(etopodata.variables['x_range'][0],
                    etopodata.variables['x_range'][1],
                    etopodata.variables['dimension'][0])
lats_e = np.linspace(etopodata.variables['y_range'][0],
                    etopodata.variables['y_range'][1],
                    etopodata.variables['dimension'][1])
etopo = etopodata.variables['z'][:]
etopo = np.reshape(etopo,etopodata.variables['dimension'][::-1])

# %%
# Load sediment thickness data
#path2grdfile = '/Users/russell/Lamont/GMT/grids/Mooney_Kaban_2010_NAsediments.nc'
path2gridfile = '/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/Mooney_Kaban_2010_NAsediments.nc'
sedsdata = Dataset(path2grdfile)
lons_s = np.linspace(sedsdata.variables['x_range'][0],
                    sedsdata.variables['x_range'][1],
                    sedsdata.variables['dimension'][0])
lats_s = np.linspace(sedsdata.variables['y_range'][0],
                    sedsdata.variables['y_range'][1],
                    sedsdata.variables['dimension'][1])
seds = sedsdata.variables['z'][:]
seds = np.reshape(seds,sedsdata.variables['dimension'][::-1])

# %%


csv_files = "./Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/outputhvsrloop/hvsrpy/"
pathlist = sorted(Path(csv_files).glob('**/*.csv'))


peak_frequency =np.array([])
peak_amplitude = np.array([])
lats=np.array([])
lons=np.array([])
stas=np.array([], dtype=object)

print(f"Number of recordings: {len(pathlist)}")
for path in pathlist:
    print(path)

    path_in_str = str(path)  
    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('_')[1]
    day_nums = path_in_str.split('/')[-1].split('_')[2]
    #loc = path.split('/')[-1].split('.')[2]
    #starttime = path_in_str.split('/')[-1].split('.')[2]
    #endtime = path_in_str.split('/')[-1].split('.')[3].split('_')[0]

     # Read station lat lon
    # inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    inventory = client.get_stations(network=net, station=sta)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)
    stas = np.append(stas, sta)
    #print(lat)

    # Load results csv file using pandas
    
    df = pd.read_csv(path,comment='#', header=None, names=['frequency','amplitude'])
    # print(df)

    # Determine peak amplitude and frequency
    peak_amplitude = np.append(peak_amplitude, df['amplitude'].max())
    peak_frequency = np.append(peak_frequency, df['frequency'][df['amplitude'].idxmax()])


# %%


# %%



