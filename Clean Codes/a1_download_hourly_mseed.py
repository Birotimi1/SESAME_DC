# %% 
# ## HVSR Quality Control Algorithm for Greenland Data
# ## Complete script with debugged file naming and loading

# %%
# Load all necessary modules
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.filter import bandpass
from datetime import datetime
import warnings 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import hvsrpy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from itertools import cycle
from scipy.fftpack import hilbert 
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter, remez)
try:
    from scipy.signal import sosfilt, zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos
import io
import contextlib
from collections import defaultdict

plt.rcParams.update({'font.size': 16})
warnings.filterwarnings('ignore')

# %%
# ==================== CENTRALIZED PATH CONFIGURATION ====================
BASE_PATH = Path("/Users/kcummi03/Desktop/full_HVSR")
#BASE_PATH = Path("/Users/birotimi/Downloads/Kenna")
DOWNLOAD_PATH = Path("/Volumes/Drive/AS-Filer/EES/jbrussel/SharedData/Greenland/full_HVSR")

# Input data directory
INPUT_DATA = DOWNLOAD_PATH / "hourlongdata"

# Filter parameters
FREQMIN = 0.1  # Hz
FREQMAX = 10.0  # Hz

print(f"Base path configured: {BASE_PATH}")
print(f"Input data directory: {INPUT_DATA}")

# %%
# ==================== HELPER FUNCTIONS ====================

def parse_mseed_filename(filename):
    """Parse YYYY-MM-DD-HH.mseed format"""
    try:
        stem = filename.replace('.mseed', '')
        dt = datetime.strptime(stem, "%Y-%m-%d-%H")
        date_str = dt.strftime("%Y%m%d")
        hour_str = dt.strftime("%H")
        return date_str, hour_str
    except:
        return None, None

def format_output_filename(station, date_str, hour_str, suffix):
    """Create consistent output filenames: STATION_YYYYMMDD_HH_suffix"""
    return f"{station}_{date_str}_{hour_str}_{suffix}"

# %%
# ==================== DATA DOWNLOAD ====================
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from pathlib import Path

# ==================== CENTRALIZED PATH CONFIGURATION ====================
#BASE_PATH = Path("/Users/kcummi03/Desktop/full_HVSR")
#INPUT_DATA = BASE_PATH / "hourlongdata"

# Create input directory
INPUT_DATA.mkdir(parents=True, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def parse_mseed_filename(filename):
    """Parse YYYY-MM-DD-HH.mseed format"""
    try:
        stem = filename.replace('.mseed', '')
        dt = datetime.strptime(stem, "%Y-%m-%d-%H")
        date_str = dt.strftime("%Y%m%d")
        hour_str = dt.strftime("%H")
        return date_str, hour_str
    except:
        return None, None

def format_output_filename(station, date_str, hour_str, suffix):
    """Create consistent output filenames: STATION_YYYYMMDD_HH_suffix"""
    return f"{station}_{date_str}_{hour_str}_{suffix}"

# ==================== DOWNLOAD SEISMIC DATA ====================

print("\n" + "="*60)
print("DOWNLOADING SEISMIC DATA - NETWORK DK")
print("="*60)

DOWNLOAD_DATA = True  # 

if DOWNLOAD_DATA:
    client = Client("IRIS")
    #client = Client("GFZ")
    
    # Geographic bounds (Greenland region)
    minlatitude = 55.0
    minlongitude = -100
    maxlatitude = 85.0
    maxlongitude = -7
    
    # Station parameters
    net = "DK" #"XF"
    sta = "**"
    loc = "**"
    chan = "BH*"
    
    # Time range
    starttime = UTCDateTime("2015-01-01")
    endtime = UTCDateTime("2016-01-01")  # 1 year full dataset
    
    n_days = int((endtime - starttime) / 86400)
    print(f"Download period: {starttime} to {endtime} ({n_days} days)")
    
    try:
        # Get station inventory
        print("\n[INFO] Fetching station inventory...")
        inv = client.get_stations(
            network=net, station=sta, location=loc, channel=chan,
            level="station", starttime=starttime, endtime=endtime,
            minlatitude=minlatitude, maxlatitude=maxlatitude,
            minlongitude=minlongitude, maxlongitude=maxlongitude
        )
        
        stations_list = inv.get_contents()['stations']
        print(f"[OK] Found {len(stations_list)} station(s)")
        for stn in stations_list:
            print(f"     - {stn}")
        
        # Process each station
        for net_obj in inv.networks:
            for sta_obj in net_obj.stations:
                sta_code = sta_obj.code
                net_code = net_obj.code
                
                print(f"\n{'='*60}")
                print(f"Processing station: {net_code}.{sta_code}")
                print(f"{'='*60}")
                
                # Create station directory
                station_dir = INPUT_DATA / f"{net_code}.{sta_code}"
                station_dir.mkdir(parents=True, exist_ok=True)
                
                # Track statistics
                downloaded = 0
                skipped_exists = 0
                skipped_no_data = 0
                skipped_gaps = 0
                errors = 0
                
                # Download hourly data
                for day in range(n_days):
                    for hour in range(24):
                        current_time = starttime + (day * 86400) + (hour * 3600)
                        next_hour = current_time + 3600
                        
                        # Stop if we've reached the end time
                        if current_time >= endtime:
                            continue
                        
                        # Create filename in YYYY-MM-DD-HH.mseed format
                        file_str = current_time.strftime("%Y-%m-%d-%H")
                        outfile = station_dir / f"{file_str}.mseed"
                        
                        # Skip if file already exists
                        if outfile.exists():
                            skipped_exists += 1
                            if hour % 6 == 0:  # Print every 6 hours to reduce clutter
                                print(f"[SKIP] {file_str}.mseed - already exists")
                            continue
                        
                        print(f"[INFO] Downloading: {file_str}")
                        
                        try:
                            # Download waveforms for this hour
                            st = client.get_waveforms(
                                network=net_code, station=sta_code,
                                location=loc, channel=chan,
                                starttime=current_time, endtime=next_hour
                            )
                            
                            # Check for empty stream
                            if len(st) == 0:
                                print(f"[SKIP] No data available")
                                skipped_no_data += 1
                                continue
                            
                            # Check for gaps
                            gaps = st.get_gaps()
                            if len(gaps) > 0:
                                print(f"[SKIP] Data has {len(gaps)} gap(s)")
                                skipped_gaps += 1
                                continue
                            
                            # Save continuous, gap-free data
                            st.write(str(outfile), format="MSEED")
                            print(f"[OK] Saved successfully")
                            downloaded += 1
                            
                        except Exception as e:
                            print(f"[ERROR] {e}")
                            errors += 1
                            continue
                
                # Print station summary
                print(f"\n{'='*60}")
                print(f"Station {net_code}.{sta_code} Summary:")
                print(f"  Downloaded: {downloaded}")
                print(f"  Already existed: {skipped_exists}")
                print(f"  No data: {skipped_no_data}")
                print(f"  Gaps detected: {skipped_gaps}")
                print(f"  Errors: {errors}")
                print(f"{'='*60}")
        
        print("\n✅ Download completed successfully.")
        
    except Exception as e:
        print(f"\n❌ Error during inventory fetch or processing: {e}")
        
else:
    print("Download disabled. Using existing data...")
    print(f"Data directory: {INPUT_DATA}")