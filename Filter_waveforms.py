# %%
import warnings 
import numpy as np
from scipy.fftpack import hilbert 
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter, remez)
try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos
    

    

# %%
import os
import pathlib
from pathlib import Path
from obspy import read
from obspy.signal.filter import bandpass

# %%
#define the base directory

base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata")

# %%
otput_path= Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/outputhourlonghvsr")

# %%
##output directory
output_direc = otput_path / "filtered_data"
output_direc.mkdir(exist_ok=True)

# %%
#filter parameters
freqmin = 0.1
freqmax = 10.0


# %%
mseed_file.name

# %%
##loop through stations and folders iteratively 

for station in sorted(base_path.iterdir()):
    if station.is_dir():
        station_name = station.name
        print(f"Processing this current station: {station_name}")
        
        # Create the output directory
        station_output_dir = output_direc / station_name
        if station_output_dir.exists():
            continue
        else:
            station_output_dir.mkdir(exist_ok=True)

        for hour_folder in sorted(station.iterdir()):
            if hour_folder.is_dir():
                print(f"Processing {hour_folder.name} in {station_name}")
                
                # Create output directory for the hour folder
                hour_output_directory = station_output_dir / hour_folder.name
                if hour_output_directory.exists():
                    continue
                else:
                    hour_output_directory.mkdir(exist_ok=True)
                    
                for mseed_file in sorted(hour_folder.glob("*.mseed")):
                    print (f"i am currently filtering them {mseed_file.name}")
                    
                    
                    ##read in the files 
                    st = read(str(mseed_file))
                    
                    
                    #apply the filter
                    for tr in st:
                        tr.data = bandpass (tr.data, freqmin. freqmax, tr.stats.sampling_rate, corners=4, zerophase=TRUE)
                        
                        
                        ##save it out fam
                        
                        output_file = hour_output_dir/f"filtered_{mseed_file.name}"
                        st.write (str(output_file), format ="MSEED")
    
    
print (F"I am done with the filtering and i need to go home")
                        
                    
                    
                    


# %%
from obspy import read
from obspy.signal.filter import bandpass
from pathlib import Path
import os

# Define base path and output directory
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata")
output_direc = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/outputhourlonghvsr")
freqmin = 0.1  #  minimum frequency for bandpass filter
freqmax = 10.0  #  maximum frequency for bandpass filter

# Loop through stations and folders iteratively
for station in sorted(base_path.iterdir()):
    if station.is_dir():
        station_name = station.name
        print(f"Processing this current station: {station_name}")
        
        # Create the output directory
        station_output_dir = output_direc / station_name
        if station_output_dir.exists():
            continue
        else:
            station_output_dir.mkdir(exist_ok=True)

        for hour_folder in sorted(station.iterdir()):
            if hour_folder.is_dir():
                print(f"Processing {hour_folder.name} in {station_name}")
                
                # Create output directory for the hour folder
                hour_output_directory = station_output_dir / hour_folder.name
                if hour_output_directory.exists():
                    continue
                else:
                    hour_output_directory.mkdir(exist_ok=True)
                    
                for mseed_file in sorted(hour_folder.glob("*.mseed")):
                    print(f"Trying to filter: {mseed_file.name}")
                    
                    # Check if the file exists and is a valid MiniSEED file
                    if not mseed_file.exists():
                        print(f"Warning: {mseed_file} does not exist.")
                        continue
                    
                    try:
                        # Read the mseed files
                        st = read(str(mseed_file))
                    except Exception as e:
                        print(f"Error reading {mseed_file}: {e}")
                        continue
                    
                    # Apply the bandpass filter to each trace
                    for tr in st:
                        try:
                            tr.data = bandpass(tr.data, freqmin, freqmax, tr.stats.sampling_rate, corners=4, zerophase=True)
                        except Exception as e:
                            print(f"Error filtering trace {tr.id}: {e}")
                            continue
                        
                    # Save the filtered data
                    try:
                        output_file = hour_output_directory / f"filtered_{mseed_file.name}"
                        st.write(str(output_file), format="MSEED")
                        print(f"Saved filtered data to {output_file}")
                    except Exception as e:
                        print(f"Error saving filtered data for {mseed_file}: {e}")
                        
print("I am done with the filtering, and I need to go home!")


# %%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.filter import bandpass

# Turn on interactive mode
plt.ion()

# Define base directory
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/hourlongdata")

# Filter parameters
freqmin = 0.1  # Low cutoff frequency (Hz)
freqmax = 10.0  # High cutoff frequency (Hz)

# Process first station and first hour for testing
for station in sorted(base_path.iterdir()):
    if station.is_dir():
        print(f"Processing station: {station.name}")

        for hour_folder in sorted(station.iterdir()):
            if hour_folder.is_dir():
                print(f"  Processing {hour_folder.name}")

                # Process only the first MiniSEED file for preview
                for mseed_file in sorted(hour_folder.glob("*.mseed")):
                    print(f"    Filtering {mseed_file.name}")

                    # Read MiniSEED file
                    st = read(str(mseed_file))

                    # Process the first trace only for visualization
                    tr = st[0]
                    tr_filt = tr.copy()
                    tr_filt.data = bandpass(tr.data, freqmin, freqmax, tr.stats.sampling_rate, corners=4, zerophase=True)

                    # Create time vector
                    t = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

                    # Check if data exists
                    print(f"Data sample: {tr.data[:10]}")  # Show first 10 points
                    print(f"Sampling rate: {tr.stats.sampling_rate} Hz")
                    print(f"Number of points: {tr.stats.npts}")

                    # Plot raw and filtered data
                    plt.figure(figsize=(10, 6))
                    plt.subplot(211)
                    plt.plot(t, tr.data, 'k', label="Raw Data")
                    plt.ylabel("Amplitude")
                    plt.legend()

                    plt.subplot(212)
                    plt.plot(t, tr_filt.data, 'r', label="Filtered (0.1-10 Hz)")
                    plt.ylabel("Amplitude")
                    plt.xlabel("Time [s]")
                    plt.legend()

                    plt.suptitle(f"Station: {station.name}, File: {mseed_file.name}")
                    plt.show(block=True)  # Force display

                    break  # Stop after first file for preview
            break  # Stop after first hour for preview
    break  # Stop after first station for preview


# %%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.filter import bandpass

# Define base and output directories
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/hourlongdata")
output_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/24hours/filtered_data")

# Filter parameters
freqmin = 0.1  # Hz
freqmax = 10.0  # Hz

# Loop through each station folder
for station_folder in sorted(base_path.iterdir()):
    if station_folder.is_dir():
        station_name = station_folder.name
        print(f"Processing station: {station_name}")

        # Make sure output directory for station exists
        station_output_dir = output_base / station_name
        station_output_dir.mkdir(parents=True, exist_ok=True)

        # Loop through all MiniSEED files in the station folder
        for mseed_file in sorted(station_folder.glob("*.mseed")):
            output_filename = f"{mseed_file.stem}_3comp.png"
            output_path = station_output_dir / output_filename

            # Skip if figure already exists
            if output_path.exists():
                print(f"    Skipping {output_filename} (already exists)")
                continue

            print(f"  Filtering file: {mseed_file.name}")
            try:
                # Read the 3-component MiniSEED file
                st = read(str(mseed_file))

                if len(st) < 3:
                    print(f"    Warning: Less than 3 components in {mseed_file.name}")
                    continue

                fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
                fig.subplots_adjust(hspace=0.4)

                # Metadata from first trace
                starttime = st[0].stats.starttime
                endtime = st[0].stats.endtime
                metadata_title = f"{station_name} | {mseed_file.name} | {starttime} to {endtime}"
                fig.suptitle(metadata_title, fontsize=14, y=0.98)

                for i, tr in enumerate(st[:3]):
                    tr_filt = tr.copy()

                    # === Preprocessing ===
                    tr_filt.detrend("demean")
                    tr_filt.taper(max_percentage=0.05, type="cosine")

                    # === Bandpass Filter ===
                    tr_filt.data = bandpass(
                        tr_filt.data, freqmin, freqmax,
                        tr.stats.sampling_rate, corners=4, zerophase=True
                    )

                    # === Time axis ===
                    t = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

                    # === Plot raw ===
                    axes[i, 0].plot(t, tr.data, color='k')
                    axes[i, 0].set_ylabel("Amplitude")
                    axes[i, 0].set_title(f"{tr.stats.channel} - Raw", fontsize=10)
                    axes[i, 0].grid(True)

                    # === Plot filtered ===
                    axes[i, 1].plot(t, tr_filt.data, color='r')
                    axes[i, 1].set_ylabel("Amplitude")
                    axes[i, 1].set_title(f"{tr.stats.channel} - Filtered (0.1–10 Hz)", fontsize=10)
                    axes[i, 1].grid(True)

                axes[-1, 0].set_xlabel("Time [s]")
                axes[-1, 1].set_xlabel("Time [s]")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(output_path)
                plt.close()
                print(f"    Saved plot: {output_path.name}")

            except Exception as e:
                print(f"    Error processing {mseed_file.name}: {e}")


# %%
#%%

# test on asingle station folder 
import os
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.filter import bandpass
from datetime import datetime

# -------------------- CONFIGURATION --------------------
station_code = "D09"
station_folder = Path(f"/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/hourlongdata/Z9.{station_code}")
output_rms_dir = Path(f"/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/24hours/RMS/{station_code}")
output_rms_dir.mkdir(parents=True, exist_ok=True)

freqmin = 0.1  # Hz
freqmax = 10.0  # Hz

# -------------------- VALIDATE --------------------
if not station_folder.exists():
    raise FileNotFoundError(f"Station folder not found: {station_folder}")

# -------------------- PROCESSING --------------------
rms_records = []

for mseed_file in sorted(station_folder.glob("*.mseed")):
    try:
        st = read(str(mseed_file))
        if len(st) < 3:
            print(f"Skipping {mseed_file.name}: less than 3 components")
            continue

        # Extract date and hour from filename: 2013-08-01-00.mseed
        base = mseed_file.stem  # '2013-08-01-00'
        try:
            dt = datetime.strptime(base, "%Y-%m-%d-%H")
        except ValueError:
            print(f"Filename format invalid: {mseed_file.name}")
            continue

        date_str = dt.strftime("%Y%m%d")
        hour_str = dt.strftime("%H")

        record = {
            "filename": mseed_file.name,
            "date": date_str,
            "hour": hour_str
        }

        for tr in st[:3]:
            tr_filt = tr.copy()
            tr_filt.detrend("demean")
            tr_filt.taper(max_percentage=0.05, type="cosine")
            tr_filt.data = bandpass(
                tr_filt.data, freqmin, freqmax,
                tr.stats.sampling_rate, corners=4, zerophase=True
            )

            rms = np.sqrt(np.mean(tr_filt.data ** 2))
            comp = tr.stats.channel[-1].upper()
            record[f'RMS_{comp}'] = rms

        rms_records.append(record)

    except Exception as e:
        print(f"Error processing {mseed_file.name}: {e}")

# -------------------- SAVE MASTER CSV --------------------
if rms_records:
    date_grouped = pd.DataFrame(rms_records).groupby("date")
    for date, group in date_grouped:
        group = group.sort_values("hour")
        output_path = output_rms_dir / f"{station_code}_{date}_rms_summary.csv"
        group.to_csv(output_path, index=False)
        print(f"Saved: {output_path.name}")
else:
    print("⚠️ No RMS records found.")


# %%



