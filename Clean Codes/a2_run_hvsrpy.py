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
BASE_PATH = Path("/Users/kcummi03/Desktop/HVSR")
#BASE_PATH = Path("/Users/birotimi/Downloads/Kenna")

# Input data directory
INPUT_DATA = BASE_PATH / "hourlongdata"

# HVSR processing outputs
THIRTY_DAYS_FIGURES = BASE_PATH / "hvsrpy" / "figures"
THIRTY_DAYS_HVSRPY = BASE_PATH / "hvsrpy" / "output_files"
HVSRPY_SUMMARY = BASE_PATH / "hvsrpy" / "hvsrpysummary"


# Filter parameters
FREQMIN = 0.1  # Hz
FREQMAX = 10.0  # Hz

print(f"Base path configured: {BASE_PATH}")
print(f"Input data directory: {INPUT_DATA}")

# Create all necessary directories
print("\nCreating directory structure...")
directories_to_create = [
    INPUT_DATA, THIRTY_DAYS_FIGURES, THIRTY_DAYS_HVSRPY, HVSRPY_SUMMARY
]

for directory in directories_to_create:
    directory.mkdir(parents=True, exist_ok=True)

print("All directories created successfully!")

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

#%%

 # ---------------- Preprocessing settings ----------------
pre_settings = hvsrpy.settings.HvsrPreProcessingSettings()
pre_settings.detrend = "linear"
pre_settings.window_length_in_seconds = 300
pre_settings.orientation_correction_in_deg = 0
pre_settings.filter = None  # no bandpass

# %%

## Download data and perform hvsr analysis

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read
from pathlib import Path
import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import hvsrpy

# ==================== HELPER FUNCTIONS ====================

def parse_mseed_filename(filename):
    """Parse YYYY-MM-DD-HH.mseed format"""
    from datetime import datetime
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

# ==================== HVSR PROCESSING ====================

print("\n" + "="*60)
print("HVSR PROCESSING FOR DOWNLOADED DATA")
print("="*60)

# Get all downloaded MiniSEED files from INPUT_DATA
pathlist = sorted(INPUT_DATA.rglob("*.mseed"))
total_files = len(pathlist)

print(f"\n[INFO] Found {total_files} MiniSEED files to process")
print(f"[INFO] Input directory: {INPUT_DATA}")
print(f"[INFO] Output directories:")
print(f"  - Figures: {THIRTY_DAYS_FIGURES}")
print(f"  - HVSR CSV: {THIRTY_DAYS_HVSRPY}")
print(f"  - Statistics: {HVSRPY_SUMMARY}")

failed_files = []
successful_files = []
skipped_files = []

for idx, path in enumerate(pathlist, 1):
    if not path.is_file():
        continue
    
    # Parse filename
    date_str, hour_str = parse_mseed_filename(path.name)
    if date_str is None:
        print(f"[{idx}/{total_files}] Skipping: {path.name} - invalid filename format")
        skipped_files.append(str(path))
        continue
    
    # Get station name from directory structure
    station_str = path.parts[-2]
    
    # Create output directories for this station
    stadir_fig = THIRTY_DAYS_FIGURES / station_str
    stadir_hvsrpy = THIRTY_DAYS_HVSRPY / station_str
    stadir_stats = HVSRPY_SUMMARY / station_str
    
    for d in [stadir_fig, stadir_hvsrpy, stadir_stats]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Consistent output naming
    base_name = format_output_filename(station_str, date_str, hour_str, "output_hvsrpy_az")
    figure_out = stadir_fig / f"{station_str}_{date_str}_{hour_str}_hvsr_figure_az.png"
    hvsrpy_out = stadir_hvsrpy / f"{base_name}.csv"
    stats_out = stadir_stats / f"{station_str}_{date_str}_{hour_str}_hvsr_statistics.txt"
    
    # Skip if already processed
    if hvsrpy_out.exists():
        print(f"[{idx}/{total_files}] Already processed: {path.name}")
        skipped_files.append(str(path))
        continue
    
    print(f"\n[{idx}/{total_files}] Processing: {station_str}/{path.name}")
    
    try:
        # Read seismic data with ObsPy
        st = read(str(path))
        sr = st[0].stats.sampling_rate
        print(f"  📏 Sampling rate: {sr} Hz")
        
        # Check if sampling rate is sufficient
        if sr < 1.0:
            print(f"  ⚠️  Skipping: SR too low ({sr} Hz) for HVSR analysis")
            skipped_files.append(str(path))
            continue
        
        # (JBR) Ensure same start and end times (same number of samples) per trace
        npts = [tr.stats.npts for tr in st]
        if len(set(npts)) > 1:
            latest_start = max([tr.stats.starttime for tr in st])
            earliest_end = min([tr.stats.endtime for tr in st])
            sr = st[0].stats.sampling_rate
            try:
                st.interpolate(sr, npts=min(npts),starttime=latest_start)
            except Exception as e:
                st.interpolate(sr, npts=min(npts)-1,starttime=latest_start)
                
            # overwrite old mseed file
            st.write(str(path), format="MSEED")

        # (JBR) If more than one location code, keep only first one
        unique_loc_codes = sorted(list({trace.stats.location for trace in st}))
        if len(unique_loc_codes) > 1:
            st = st.select(location=unique_loc_codes[0])


        # Check channel names and components BEFORE any modification
        components = {tr.stats.channel[-1].upper() for tr in st}
        print(f"  📊 Components found: {components}")
        
        # Fix channel names if needed - save to TEMP file first
        isoverwrite = False
        for tr in st:
            if tr.stats.channel.endswith('1'):
                tr.stats.channel = tr.stats.channel.replace('1', 'N')
                isoverwrite = True
            elif tr.stats.channel.endswith('2'):
                tr.stats.channel = tr.stats.channel.replace('2', 'E')
                isoverwrite = True
        
        if isoverwrite:
            print("  🔧 Channel names need correction")
            # Save to temporary file
            temp_file = path.parent / f"{path.stem}_temp.mseed"
            st.write(str(temp_file), format="MSEED")
            print(f"  💾 Saved corrected version to temp file")
            
            # Try to read with hvsrpy from temp file
            try:
                srecords_test = hvsrpy.read(str(temp_file))
                print("  ✅ hvsrpy can read temp file - replacing original")
                # If successful, replace original
                temp_file.replace(path)
            except Exception as e:
                print(f"  ❌ hvsrpy cannot read temp file: {e}")
                temp_file.unlink()  # Delete temp file
                failed_files.append(str(path))
                continue
        
        # Ensure we have E, N, Z components
        components = {tr.stats.channel[-1].upper() for tr in st}
        if not {'E', 'N', 'Z'}.issubset(components):
            print(f"  ⚠️  Skipping: Missing required components (need E, N, Z)")
            skipped_files.append(str(path))
            continue
        
        # HVSR processing settings
        processing_settings = hvsrpy.settings.HvsrDiffuseFieldProcessingSettings()
        processing_settings.window_type_and_width = ("tukey", 0.1)
        processing_settings.smoothing = dict(
            operator="log_rectangular",
            bandwidth=0.1,
            center_frequencies_in_hz=np.geomspace(0.2, sr / 2, 256)
        )
        
        # Process using HVSRpy
        srecords = hvsrpy.read(str(path))
        hvsr = hvsrpy.process(srecords, processing_settings)
        
        # Capture statistics
        stats_buffer = io.StringIO()
        with contextlib.redirect_stdout(stats_buffer):
            hvsrpy.summarize_hvsr_statistics(hvsr)
        
        # Plot and save figure
        fig, ax = hvsrpy.plot_single_panel_hvsr_curves(hvsr)
        fig.savefig(figure_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save HVSR object as CSV
        hvsrpy.object_io.write_hvsr_object_to_file(hvsr, hvsrpy_out)
        
        # Save statistics to text
        with open(stats_out, 'w') as f:
            f.write(stats_buffer.getvalue())
        
        print(f"  ✅ Saved: {hvsrpy_out.name}")
        successful_files.append(str(path))
    
    except Exception as e:
        print(f"  ❌ Error processing {path.name}: {e}")
        failed_files.append(str(path))

# ==================== SUMMARY ====================

print("\n" + "="*60)
print("🎉 HVSR PROCESSING COMPLETE")
print("="*60)
print(f"\n📊 SUMMARY:")
print(f"  Total files found: {total_files}")
print(f"  ✅ Successfully processed: {len(successful_files)}")
print(f"  ⏭️  Skipped (already done/invalid): {len(skipped_files)}")
print(f"  ❌ Failed: {len(failed_files)}")

if failed_files:
    print(f"\n❌ FAILED FILES ({len(failed_files)}):")
    for f in failed_files[:20]:  # Show first 20
        print(f"  - {f}")
    if len(failed_files) > 20:
        print(f"  ... and {len(failed_files) - 20} more")

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"  Figures: {THIRTY_DAYS_FIGURES}")
print(f"  HVSR CSV: {THIRTY_DAYS_HVSRPY}")
print(f"  Statistics: {HVSRPY_SUMMARY}")
print("\n" + "="*60)
