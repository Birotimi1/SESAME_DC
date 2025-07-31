# %% [markdown]
# ## This script will contain our latest updated quality control algorithm to improve the 
# # inversion of HVSR under the DFA, it will include a 3 steps check which will aim to remove
# # bad hours defined by having any channel of the seismometer faulty for that period of the day 
# # of there is an earthquake or blast that occured during an hour of the day. 
# # step one will calculate the rms of the filtered waveform and drop any hour of the day with value of 0 which implies the
# # station channel isn't recording at this timeframe, this hours will be dropped before the HVSR is computed 
# # the step 2 will check for the max_amplitude of the of the computed HVSR, if condition max_amp >1 <100 is met,
# # the hour passed, but an hour having its max _amp <1 and max_amp > 99 is dropped.
# # the step 3 will compute the misfit using reduced chi square and checks if the hourly hvsr is within acceptble range of <3
# # if the misfit is greater than 3, it is tagged a bad hour and will be dropped as well.
# 
# # this final step is to recompute the QC mean and std for the hours that passed and replot the original computed hvsr
# # of the hours that passed.
# 
# # The above workflow helps deal with anthropogenic noise, station instrument malfunctions, blasts, earthquakes, without
# # tampering with the computed HVSR curve for the day thus helping ensure more stable, consistent and correct inversion.

# %%


# %%
#load all neccesary modules 

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
import pandas as pd
plt.rcParams.update({'font.size': 16}) # set a larger font size

import numpy as np
import re
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.fftpack import hilbert 
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter, remez)
try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos

# %%


# %%
# Begin step 1: RMS
# Demean, detrend, and apply cosine filter, then compute the RMS of the filtered waveforms


# %%

# -------------------- CONFIGURATION --------------------
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/hourlongdata")
output_qc_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC")

good_dir = output_qc_path / "Goodminiseed"
bad_dir = output_qc_path / "Badminiseed"
summary_dir = output_qc_path / "RMS_summary"

freqmin = 0.1  # Hz
freqmax = 10.0  # Hz

# -------------------- PROCESSING --------------------
all_bad_records = []

for station_folder in sorted(base_path.glob("Z9.[DEW]*")):
    station_code = station_folder.name.split(".")[-1]
    print(f"Processing station: {station_code}")

    good_station_dir = good_dir / station_code
    bad_station_dir = bad_dir / station_code
    summary_station_dir = summary_dir / station_code

    good_station_dir.mkdir(parents=True, exist_ok=True)
    bad_station_dir.mkdir(parents=True, exist_ok=True)
    summary_station_dir.mkdir(parents=True, exist_ok=True)

    rms_records = {}

    for mseed_file in sorted(station_folder.glob("*.mseed")):
        # Skip if file already exists in either QC folder
        if (good_station_dir / mseed_file.name).exists() or (bad_station_dir / mseed_file.name).exists():
            print(f"Skipping {mseed_file.name}: already processed")
            continue

        try:
            st = read(str(mseed_file))
            components_present = {tr.stats.channel[-1].upper() for tr in st}

            if not {'E', 'N', 'Z'}.issubset(components_present):
                print(f"Skipping {mseed_file.name}: missing one or more components (E, N, Z)")
                continue

            base = mseed_file.stem
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
                "hour": hour_str,
                "RMS_E": 0.0,
                "RMS_N": 0.0,
                "RMS_Z": 0.0
            }

            rms_values = {}

            for tr in st:
                comp = tr.stats.channel[-1].upper()
                if comp not in {"E", "N", "Z"}:
                    continue

                tr_proc = tr.copy()
                tr_proc.detrend("demean")
                tr_proc.taper(max_percentage=0.05, type="cosine")
                tr_proc.data = bandpass(
                    tr_proc.data, freqmin, freqmax,
                    tr_proc.stats.sampling_rate, corners=4, zerophase=True
                )

                rms = np.sqrt(np.mean(tr_proc.data ** 2))
                rms_values[f'RMS_{comp}'] = rms

            record.update(rms_values)

            if any(rms_values.get(f'RMS_{comp}', 0) == 0 for comp in ["E", "N", "Z"]):
                shutil.copy(mseed_file, bad_station_dir / mseed_file.name)
                if date_str not in rms_records:
                    rms_records[date_str] = []
                rms_records[date_str].append(record)
            else:
                shutil.copy(mseed_file, good_station_dir / mseed_file.name)

        except Exception as e:
            print(f"Error processing {mseed_file.name}: {e}")

    # Save per-date summaries for bad files
    for date_str, records in rms_records.items():
        df = pd.DataFrame(records).sort_values("hour")
        summary_path = summary_station_dir / f"{station_code}_{date_str}_rms_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path.name}")
        df["station"] = station_code
        all_bad_records.append(df)

# -------------------- MASTER CSV OF ALL BAD HOURS --------------------
if all_bad_records:
    master_df = pd.concat(all_bad_records, ignore_index=True)
    master_csv_path = summary_dir / "ALL_bad_rms_summary.csv"
    master_df.to_csv(master_csv_path, index=False)
    print(f"📄 Master CSV of all bad hours saved: {master_csv_path}")
else:
    print("⚠️ No bad RMS data found to compile.")


# %%
# Step 2:
# for the stations having the hours that passed, we will proceed to compute the HVSR using hvsrpy and the goodminiseed datasets

# -------------------- BASE PATHS --------------------


base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days")
goodminiseed_path = base_path / "QC/Goodminiseed"

# Save everything under QC
qc_path = base_path / "QC"
fig_base_out = qc_path / "thirty_days_figures"
hvsrpy_base_out = qc_path / "thirty_days_hvsrpy"
summary_base_out = qc_path / "hvsrpysummary"

fig_base_out.mkdir(parents=True, exist_ok=True)
hvsrpy_base_out.mkdir(parents=True, exist_ok=True)
summary_base_out.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESSING --------------------
pathlist = sorted(goodminiseed_path.glob('**/*.mseed'))

for path in pathlist:
    print(f"Processing: {path}")
    
    # Determine station and file information from the path
    station_str = path.parent.name
    filename = path.name
    file_toks = filename.split('-')

    if len(file_toks) < 4:
        print(f"Skipping {filename} due to unexpected format.")
        continue

    year, month, day, hour_ext = file_toks
    hour = hour_ext.split('.')[0]

    # Define per-station output folders
    station_fig_dir = fig_base_out / station_str
    station_hvsrpy_dir = hvsrpy_base_out / station_str
    station_summary_dir = summary_base_out / station_str

    station_fig_dir.mkdir(parents=True, exist_ok=True)
    station_hvsrpy_dir.mkdir(parents=True, exist_ok=True)
    station_summary_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    fig_file = station_fig_dir / f"{station_str}_{year}{month}{day}_{hour}_hvsr_figure_az.png"
    csv_file = station_hvsrpy_dir / f"{station_str}_{year}{month}{day}_{hour}_output_hvsrpy_az.csv"
    summary_file = station_summary_dir / f"{station_str}_{year}{month}{day}_{hour}_summary.txt"

    # Skip if the hour has already been processed for both figures and hvsrpy outputs
    if fig_file.exists() and csv_file.exists():
        print(f"Skipping {filename}: hour miniseed has already been processed (figure and CSV exist).")
        continue

    # Read the miniseed file (assumed to contain 3-channel data)
    try:
        st = read(path)
    except Exception as read_err:
        print(f"Error reading {filename}: {read_err}")
        continue

    sr = st[0].stats.sampling_rate

    # Fix channels: 1 → N, 2 → E (update metadata if available)
    isoverwrite = False
    for tr in st:
        if tr.stats.channel.endswith('1'):
            tr.stats.channel = tr.stats.channel.replace('1', 'N')
            if hasattr(tr, "meta") and hasattr(tr.meta, "channel"):
                tr.meta.channel = tr.meta.channel.replace('1', 'N')
            isoverwrite = True
        elif tr.stats.channel.endswith('2'):
            tr.stats.channel = tr.stats.channel.replace('2', 'E')
            if hasattr(tr, "meta") and hasattr(tr.meta, "channel"):
                tr.meta.channel = tr.meta.channel.replace('2', 'E')
            isoverwrite = True
    if isoverwrite:
        st.write(str(path), format="MSEED")

    # HVSR Processing settings
    settings = hvsrpy.settings.HvsrDiffuseFieldProcessingSettings()
    settings.window_type_and_width = ("tukey", 0.1)
    settings.smoothing = dict(
        operator="log_rectangular",
        bandwidth=0.1,
        center_frequencies_in_hz=np.geomspace(0.2, sr / 2, 256)
    )

    # Read the file using hvsrpy
    try:
        srecords = hvsrpy.read(str(path))
    except Exception as err:
        print(f"Error in hvsrpy.read for {filename}: {err}")
        continue

    # Process HVSR
    try:
        hvsr = hvsrpy.process(srecords, settings)
    except Exception as proc_err:
        print(f"Error processing HVSR for {filename}: {proc_err}")
        continue

    # Plot and save figure
    try:
        fig, ax = hvsrpy.plot_single_panel_hvsr_curves(hvsr)
        # plt.show() is optional; remove if running in non-interactive mode.
        plt.show()  
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {fig_file}")
    except Exception as plot_err:
        print(f"Error during plotting/saving figure for {filename}: {plot_err}")
        continue

    # Save HVSR result CSV
    try:
        hvsrpy.object_io.write_hvsr_object_to_file(hvsr, csv_file)
        print(f"HVSR CSV saved: {csv_file}")
    except Exception as csv_err:
        print(f"Error saving HVSR CSV for {filename}: {csv_err}")
        continue

    # Save summary TXT (after summarizing)
    try:
        hvsrpy.summarize_hvsr_statistics(hvsr)  # attaches .statistics to the object
        peak_freq = hvsr.statistics["peak_frequency"]
        peak_amp = hvsr.statistics["peak_amplitude"]
        summary_text = (
            f"Statistical Summary:\n"
            f"{'-'*20}\n"
            f"The peak of the mean curve is at {peak_freq:.3f} Hz with amplitude {peak_amp:.3f}.\n"
        )
        with open(summary_file, "w") as f:
            f.write(summary_text)
        print(f"Summary saved: {summary_file}")
    except Exception as e:
        print(f"⚠️ Could not write summary for {filename}: {e}")
        continue

print("✅ HVSR Processing complete.")


# %%
##visualize the hvsrpy results 
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

# -------------------- PATH SETUP --------------------
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/thirty_days_hvsrpy")
output_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/plots")

# -------------------- COLOR PALETTE --------------------
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#d17a22', '#2171b5', '#7f7f7f', '#bcbd22', '#17becf',
    '#8c564b', '#e377c2', '#ffbf00', '#aa0b4b', '#009688',
    '#c2185b', '#7b1fa2', '#1976d2', '#388e3c', '#f57c00',
    '#e64a19', '#455a64', '#c0ca33', '#00acc1'
]
color_cycle = cycle(color_palette)

# -------------------- FILENAME REGEX --------------------
# Match: Z9.D07_20130801_00_output_hvsrpy_az.csv
filename_pattern = re.compile(r'_(\d{8})_(\d{2})_output_hvsrpy_az')

# -------------------- PROCESS STATIONS --------------------
netsta_list = next(os.walk(base_path))[1]

for netsta in netsta_list:
    netsta_path = base_path / netsta
    output_dir = output_base / netsta
    output_dir.mkdir(parents=True, exist_ok=True)

    path2file_list = sorted(netsta_path.glob('*.csv'))

    # Organize files by date
    daily_files = {}
    for path2file in path2file_list:
        filename = path2file.stem
        match = filename_pattern.search(filename)
        if not match:
            print(f"⚠️ Skipping file due to incorrect format: {filename}")
            continue

        date_part, hour_part = match.groups()

        if date_part not in daily_files:
            daily_files[date_part] = {}

        daily_files[date_part][hour_part] = path2file

    # -------------------- PLOTTING --------------------
    for date_part, hourly_files in sorted(daily_files.items()):
        fig, ax1 = plt.subplots(figsize=(9, 3))
        ax1.set_xscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('HVSR Amplitude')
        ax1.set_title(f"{netsta} - {date_part}")
        ax1.set_ylim(0, 12)

        color_cycle = cycle(color_palette)
        plotted = False

        for hour in sorted(hourly_files.keys(), key=lambda h: int(h)):
            path2file = hourly_files[hour]
            if path2file.exists():
                df = pd.read_csv(path2file, comment='#', header=None, names=['frequency', 'amplitude'])
                ax1.plot(df['frequency'], df['amplitude'], linewidth=2, color=next(color_cycle), label=f'Hour {hour}')
                plotted = True
            else:
                print(f"⚠️ File not found: {path2file}")

        if plotted:
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            output_file = output_dir / f"{netsta}_{date_part}_hourly_HVSR.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_file}")

        plt.close(fig)

print("🎉 HVSR plotting complete.")



# %%
# extract out the amplitudes to begin another check 


# -------------------- BASE PATHS --------------------
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/thirty_days_hvsrpy")
save_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/amplitudes")
save_path.mkdir(parents=True, exist_ok=True)

# -------------------- HELPERS --------------------
def extract_date_hour_from_filename(filename):
    match = re.search(r'_(\d{8})_(\d{2})_output_hvsrpy_az', filename)
    if match:
        return match.group(1), int(match.group(2))  # Return YYYYMMDD and hour as int
    return None, None

# -------------------- PROCESS STATIONS --------------------
station_dirs = [d for d in base_path.iterdir() if d.is_dir()]

if not station_dirs:
    print("No station directories found in base path.")
else:
    print(f"Found {len(station_dirs)} station directories.")

successful_stations = 0

for station_dir in station_dirs:
    station_name = station_dir.name
    csv_files = sorted(station_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found for station {station_name}.")
        continue

    print(f"Processing station {station_name} with {len(csv_files)} CSV files found.")

    # Create output directory for this station
    station_amplitude_dir = save_path / station_name
    station_amplitude_dir.mkdir(parents=True, exist_ok=True)

    daily_data = {}

    for csv_file in csv_files:
        date_str, hour = extract_date_hour_from_filename(csv_file.stem)

        if date_str is None or hour is None:
            print(f"⚠️ Skipping file with unexpected format: {csv_file.name}")
            continue

        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file, comment='#', header=None, usecols=[1], names=['amplitude'])
                print(f"✅ Loaded: {csv_file} (Date {date_str}, Hour {hour:02d})")

                if df.shape[0] != 256:
                    print(f"⚠️ Warning: {csv_file.name} has {df.shape[0]} rows (expected 256). Skipping.")
                    continue

                if date_str not in daily_data:
                    daily_data[date_str] = {}
                daily_data[date_str][hour] = df

            except pd.errors.EmptyDataError:
                print(f"⚠️ Empty CSV: {csv_file}")
            except pd.errors.ParserError as e:
                print(f"⚠️ Parser error in {csv_file}: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error in {csv_file}: {e}")

    # Save amplitudes
    for date_str, hourly_data in daily_data.items():
        for hour in range(24):
            if hour in hourly_data:
                output_filename = f"{station_name}_{date_str}_{hour:02d}_amplitudes.csv"
                output_path = station_amplitude_dir / output_filename
                hourly_data[hour].to_csv(output_path, index=False)
                print(f"💾 Saved: {output_path}")

    print(f"✅ Completed processing for station {station_name}")
    successful_stations += 1

print(f"\n🎉 Successfully processed {successful_stations} stations.")


# %%
#Step 3: 

# this is where we check for the maximum amplitude of the hvsr files and implement a control
# involving if max_amp is within >1 and <100.



# -------------------- BASE PATHS --------------------
base_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/amplitudes")
save_path = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/Good_hours")
bad_hours_path = save_path / "bad_hours"

# Ensure output directories exist
save_path.mkdir(parents=True, exist_ok=True)
bad_hours_path.mkdir(parents=True, exist_ok=True)

# -------------------- HELPERS --------------------
def extract_date_hour_from_filename(filename):
    match = re.search(r'_(\d{8})_(\d{2})_amplitudes', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

# -------------------- PROCESS --------------------
bad_hours = []
station_dirs = [d for d in base_path.iterdir() if d.is_dir()]

if not station_dirs:
    print("❌ No station directories found.")
else:
    print(f"📁 Found {len(station_dirs)} station folders.")

for station_dir in station_dirs:
    station_name = station_dir.name
    csv_files = sorted(station_dir.glob("*.csv"))

    if not csv_files:
        print(f"⚠️ No CSV files in {station_name}. Skipping.")
        continue

    # Create good and bad directories for the station
    good_station_path = save_path / station_name
    bad_station_path = bad_hours_path / station_name
    good_station_path.mkdir(parents=True, exist_ok=True)
    bad_station_path.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        date_str, hour = extract_date_hour_from_filename(csv_file.stem)

        if date_str is None or hour is None:
            print(f"⚠️ Skipping bad format: {csv_file.name}")
            continue

        try:
            df = pd.read_csv(csv_file, comment='#', header=None, skiprows=1, names=['amplitude'])

            if df.shape[0] != 256:
                print(f"⚠️ Skipping {csv_file.name} (expected 256 rows, got {df.shape[0]})")
                continue

            max_amp = df['amplitude'].max()
            output_filename = f"{station_name}_{date_str}_{hour:02d}_amplitudes.csv"

            if max_amp < 1 or max_amp > 99:
                # Bad hour
                df.to_csv(bad_station_path / output_filename, index=False)
                print(f"🚫 Bad hour saved: {output_filename} (max_amp = {max_amp:.2f})")
                bad_hours.append({
                    'station': station_name,
                    'date': date_str,
                    'hour': f"{hour:02d}",
                    'max_amplitude': max_amp,
                    'file': output_filename
                })
            else:
                # Good hour
                df.to_csv(good_station_path / output_filename, index=False)
                print(f"✅ Good hour saved: {output_filename}")

        except Exception as e:
            print(f"❌ Error reading {csv_file.name}: {e}")

# -------------------- SAVE BAD HOUR REPORT --------------------
if bad_hours:
    bad_hours_df = pd.DataFrame(bad_hours)
    report_path = save_path / "bad_hours_report.csv"
    bad_hours_df.to_csv(report_path, index=False)
    print(f"\n📄 Bad hour report saved to: {report_path}")
    print(f"🧾 Total bad hours: {len(bad_hours)}")
else:
    print("\n✅ No bad hours found. All amplitudes within expected range.")

print("🎉 Amplitude quality control complete.")


# %%
# next we will compute the mean and standard deviation of the good hours 


# -------------------- BASE PATHS --------------------
amplitude_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/Good_hours")
mean_output_base = amplitude_base.parent / "mean_amplitudes"
std_output_base = amplitude_base.parent / "std_amplitudes"

mean_output_base.mkdir(parents=True, exist_ok=True)
std_output_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS EACH STATION --------------------
for station_dir in sorted(amplitude_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    csv_files = sorted(station_dir.glob("*.csv"))

    # Group files by date (YYYYMMDD)
    daily_files = {}
    for file in csv_files:
        parts = file.stem.split("_")
        if len(parts) < 3:
            continue
        date = parts[1]
        daily_files.setdefault(date, []).append(file)

    # Process each day's 24 files
    for date, files in daily_files.items():
        stack = []

        for file in files:
            try:
                df = pd.read_csv(file)
                if df.shape[0] == 256 and 'amplitude' in df.columns:
                    stack.append(df['amplitude'].values)
                else:
                    print(f"⚠️ Skipping {file.name} — incorrect shape or missing 'amplitude'")
            except Exception as e:
                print(f"❌ Error reading {file.name}: {e}")

        if not stack:
            print(f"⚠️ No valid amplitude data for {station_name} on {date}")
            continue

        data = np.vstack(stack)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)

        # Output paths
        mean_dir = mean_output_base / station_name
        std_dir = std_output_base / station_name
        mean_dir.mkdir(parents=True, exist_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        mean_file = mean_dir / f"{station_name}_{date}_mean_amplitudes.csv"
        std_file = std_dir / f"{station_name}_{date}_std_amplitudes.csv"

        pd.DataFrame({'amplitude': mean_curve}).to_csv(mean_file, index=False)
        pd.DataFrame({'amplitude': std_curve}).to_csv(std_file, index=False)

        print(f"✅ Saved mean and std for {station_name} {date}")


# %%
# the next is to compute the hvsr diff, but ensuring we skip the hour amplitude that is not 
# available is not computed 


# -------------------- BASE PATHS --------------------
amplitude_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/Good_hours")
mean_base = amplitude_base.parent / "mean_amplitudes"
diff_output_base = amplitude_base.parent / "hvsr_diff"
diff_output_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS EACH STATION --------------------
for station_dir in sorted(amplitude_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    mean_station_dir = mean_base / station_name
    diff_station_dir = diff_output_base / station_name
    diff_station_dir.mkdir(parents=True, exist_ok=True)

    amp_files = sorted(station_dir.glob("*.csv"))

    for amp_file in amp_files:
        parts = amp_file.stem.split("_")
        if len(parts) < 3:
            print(f"⚠️ Skipping bad filename: {amp_file.name}")
            continue

        date, hour = parts[1], parts[2]
        mean_file = mean_station_dir / f"{station_name}_{date}_mean_amplitudes.csv"

        if not mean_file.exists():
            print(f"⚠️ Missing mean file: {mean_file.name} → Skipping {amp_file.name}")
            continue

        try:
            df_amp = pd.read_csv(amp_file)
            df_mean = pd.read_csv(mean_file)

            if df_amp.shape[0] != 256 or df_mean.shape[0] != 256:
                print(f"⚠️ Skipping {amp_file.name} — unexpected row count")
                continue

            hvsr_diff = df_amp['amplitude'] - df_mean['amplitude']
            df_diff = pd.DataFrame({'hvsr_diff': hvsr_diff})

            out_file = diff_station_dir / f"{station_name}_{date}_{hour}_hvsr_diff.csv"
            df_diff.to_csv(out_file, index=False)
            print(f"✅ HVSR diff saved: {out_file.name}")

        except Exception as e:
            print(f"❌ Error processing {amp_file.name}: {e}")

print("🎉 HVSR difference computation complete.")


# %%
# now we will square the hvsr differences 


# -------------------- PATHS --------------------
diff_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/hvsr_diff")
squared_base = diff_base.parent / "hvsrdiff_squared"
squared_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS EACH STATION --------------------
for station_dir in sorted(diff_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    output_station_dir = squared_base / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)

    for diff_file in station_dir.glob("*.csv"):
        try:
            df = pd.read_csv(diff_file)

            if "hvsr_diff" not in df.columns or df.shape[0] != 256:
                print(f"⚠️ Skipping {diff_file.name} — bad format or row count")
                continue

            df_squared = pd.DataFrame({
                'hvsrdiff_squared': df['hvsr_diff'] ** 2
            })

            output_file = output_station_dir / diff_file.name.replace("hvsr_diff", "hvsrdiff_squared")
            df_squared.to_csv(output_file, index=False)
            print(f"✅ Squared diff saved: {output_file.name}")

        except Exception as e:
            print(f"❌ Error processing {diff_file.name}: {e}")

print("🎉 HVSR squared difference computation complete.")


# %%
# now square the standard deviation before we start computing the hvsr misfit 


# -------------------- PATHS --------------------
std_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/std_amplitudes")
squared_base = std_base.parent / "std_squared"
squared_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS EACH STATION --------------------
for station_dir in sorted(std_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    output_station_dir = squared_base / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)

    for std_file in station_dir.glob("*.csv"):
        try:
            df = pd.read_csv(std_file)

            if "amplitude" not in df.columns or df.shape[0] != 256:
                print(f"⚠️ Skipping {std_file.name} — bad format or row count")
                continue

            df_squared = pd.DataFrame({
                'std_squared': df['amplitude'] ** 2
            })

            output_file = output_station_dir / std_file.name.replace("std_amplitudes", "std_squared")
            df_squared.to_csv(output_file, index=False)
            print(f"✅ Std squared saved: {output_file.name}")

        except Exception as e:
            print(f"❌ Error processing {std_file.name}: {e}")

print("🎉 Standard deviation squaring complete.")


# %%
# now we will compute the misfit which is the squared hvsr diff divided by the squared standard deviation


# -------------------- INPUT & OUTPUT PATHS --------------------
hvsrdiff_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/hvsrdiff_squared")
std_squared_base = hvsrdiff_base.parent / "std_squared"
misfit_base = hvsrdiff_base.parent / "misfit"
misfit_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS STATIONS --------------------
for station_dir in sorted(hvsrdiff_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    std_station_dir = std_squared_base / station_name
    output_station_dir = misfit_base / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)

    for diff_file in station_dir.glob("*.csv"):
        parts = diff_file.stem.split("_")
        if len(parts) < 4:
            print(f"⚠️ Skipping bad filename: {diff_file.name}")
            continue

        date = parts[1]
        hour = parts[2]
        std_file = std_station_dir / f"{station_name}_{date}_std_squared.csv"

        if not std_file.exists():
            print(f"⚠️ Missing std_squared file: {std_file.name}")
            continue

        try:
            df_diff = pd.read_csv(diff_file)
            df_std = pd.read_csv(std_file)

            if df_diff.shape[0] != 256 or df_std.shape[0] != 256:
                print(f"⚠️ Skipping {diff_file.name} — row mismatch")
                continue

            misfit = df_diff['hvsrdiff_squared'] / df_std['std_squared']
            df_misfit = pd.DataFrame({'misfit': misfit})

            output_file = output_station_dir / f"{station_name}_{date}_{hour}_misfit.csv"
            df_misfit.to_csv(output_file, index=False)

            print(f"✅ Misfit saved: {output_file.name}")

        except Exception as e:
            print(f"❌ Error computing misfit for {diff_file.name}: {e}")

print("🎉 HVSR misfit computation complete.")


# %%
# now we will average the misfit 


# Input & Output paths
misfit_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/misfit")
averaged_base = misfit_base.parent / "averaged_misfit"
averaged_base.mkdir(parents=True, exist_ok=True)

# Loop through stations
for station_dir in sorted(misfit_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    output_station_dir = averaged_base / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)

    for misfit_file in station_dir.glob("*.csv"):
        parts = misfit_file.stem.split("_")
        if len(parts) < 4:
            print(f"⚠️ Skipping improperly named file: {misfit_file.name}")
            continue

        date = parts[1]
        hour = parts[2]

        try:
            df = pd.read_csv(misfit_file)

            if "misfit" not in df.columns or df.shape[0] != 256:
                print(f"⚠️ Skipping {misfit_file.name} — invalid format")
                continue

            avg_misfit = df["misfit"].mean()
            out_file = output_station_dir / f"{station_name}_{date}_hour{hour}_averaged_misfit_function.csv"
            pd.DataFrame({'averaged_misfit': [avg_misfit]}).to_csv(out_file, index=False)

            print(f"✅ Averaged misfit saved: {out_file.name}")

        except Exception as e:
            print(f"❌ Error processing {misfit_file.name}: {e}")

print("🎯 Averaged hourly misfit computation complete.")


# %%
# now we will plot each hourly misfit across the 24 hours for each day 
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Base input/output paths
avg_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/averaged_misfit")
plot_base = avg_base.parent / "averaged_misfit_plot"
line_base = plot_base / "line_plot"
scatter_base = plot_base / "scatter_plot"

# Ensure root output folders exist
line_base.mkdir(parents=True, exist_ok=True)
scatter_base.mkdir(parents=True, exist_ok=True)

# Process stations
for station_dir in sorted(avg_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))

    # Group by day
    daily_data = defaultdict(dict)
    for file in files:
        parts = file.stem.split("_")
        if len(parts) >= 4:
            date = parts[1]
            hour = int(parts[2].replace("hour", ""))
            try:
                df = pd.read_csv(file)
                if 'averaged_misfit' in df.columns and df.shape[0] == 1:
                    daily_data[date][hour] = df['averaged_misfit'].values[0]
            except Exception as e:
                print(f"⚠️ Error reading {file.name}: {e}")

    # Create station output folders
    line_dir = line_base / station_name
    scatter_dir = scatter_base / station_name
    line_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)

    for date, hour_dict in sorted(daily_data.items()):
        hours = list(range(24))
        values = [hour_dict.get(h, None) for h in hours]
        available_hours = [h for h in hours if h in hour_dict]
        available_vals = [hour_dict[h] for h in available_hours]

        # -------------------- Line Plot --------------------
        plt.figure(figsize=(10, 3))
        plt.plot(hours, values, marker='o', linestyle='-', label='Averaged Misfit', color='blue')
        for h in hours:
            if h not in hour_dict:
                plt.scatter(h, 0, color='red', label='Missing Hour' if 'Missing Hour' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.title(f"{station_name} — {date} (Line Plot)")
        plt.xlabel("Hour")
        plt.ylabel("Average Misfit")
        plt.xticks(range(24))
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend()
        line_file = line_dir / f"{station_name}_{date}_hourly_misfit_line.pdf"
        plt.savefig(line_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved line plot: {line_file}")

        # -------------------- Scatter Plot --------------------
        plt.figure(figsize=(10, 3))
        plt.scatter(available_hours, available_vals, color='green', label='Available Hour')
        missing_hours = [h for h in hours if h not in hour_dict]
        if missing_hours:
            plt.scatter(missing_hours, [0]*len(missing_hours), color='red', label='Missing Hour')
        plt.title(f"{station_name} — {date} (Scatter Plot)")
        plt.xlabel("Hour")
        plt.ylabel("Average Misfit")
        plt.xticks(range(24))
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend()
        scatter_file = scatter_dir / f"{station_name}_{date}_hourly_misfit_scatter.pdf"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved scatter plot: {scatter_file}")

print("✅ All hourly misfit plots generated.")


# %%
# now we will plot the hvsrpy by introducing the threshold of if averaged misfit is greater than 3, color the hour  hvsrpy line red but if less than 3 or equal to 3 , color the hour blue 
#added a new step to extract out the csvs 

# -------------------- PATH SETUP --------------------
hvsr_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/thirty_days_hvsrpy")
misfit_base = hvsr_base.parent / "averaged_misfit"
output_base = hvsr_base.parent / "misfit_colored_hvsrpy"
blue_csv_base = hvsr_base.parent / "QC_amplitudes"

# -------------------- FILENAME PATTERN --------------------
filename_pattern = re.compile(r'_(\d{8})_(\d{2})_output_hvsrpy_az')

# -------------------- PROCESS STATIONS --------------------
netsta_list = next(os.walk(hvsr_base))[1]

for netsta in netsta_list:
    netsta_path = hvsr_base / netsta
    misfit_path = misfit_base / netsta
    output_dir = output_base / netsta
    blue_csv_dir = blue_csv_base / netsta

    output_dir.mkdir(parents=True, exist_ok=True)
    blue_csv_dir.mkdir(parents=True, exist_ok=True)

    hvsr_files = sorted(netsta_path.glob('*.csv'))

    # Group files by date
    daily_files = {}
    for file in hvsr_files:
        match = filename_pattern.search(file.stem)
        if not match:
            print(f"⚠️ Skipping bad filename: {file.name}")
            continue
        date_str, hour_str = match.groups()
        daily_files.setdefault(date_str, {})[hour_str] = file

    # -------------------- PLOT BY DAY --------------------
    for date_str, hourly_dict in sorted(daily_files.items()):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('HVSR Amplitude')
        ax.set_title(f"{netsta} — {date_str}")
        ax.set_ylim(0, 12)

        for hour_str in sorted(hourly_dict.keys(), key=lambda h: int(h)):
            hvsr_file = hourly_dict[hour_str]
            misfit_file = misfit_path / f"{netsta}_{date_str}_hour{hour_str}_averaged_misfit_function.csv"

            try:
                df = pd.read_csv(hvsr_file, comment='#', header=None, names=['frequency', 'amplitude'])

                if misfit_file.exists():
                    mf = pd.read_csv(misfit_file)
                    avg_misfit = mf['averaged_misfit'].values[0]
                    if avg_misfit > 3:
                        color = 'red'
                    else:
                        color = 'blue'
                        # Copy CSV for blue hours (good quality)
                        out_csv = blue_csv_dir / hvsr_file.name
                        shutil.copy(hvsr_file, out_csv)
                    label = f"Hour {hour_str} (misfit={avg_misfit:.2f})"
                else:
                    color = 'green'
                    label = f"Hour {hour_str} (no misfit)"

                ax.plot(df['frequency'], df['amplitude'], linewidth=1.5, color=color, label=label)

            except Exception as e:
                print(f"⚠️ Error plotting {hvsr_file.name}: {e}")

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plot_file = output_dir / f"{netsta}_{date_str}_misfit_colored_HVSR.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {plot_file}")

print("🎉 HVSR misfit-colored plotting + CSV extraction complete.")


# %%
# now we will compute and extract the mean and std as bounds for the QC hvsrpy 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -------------------- PATHS --------------------
qc_amp_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/QC_amplitudes")
qc_mean_base = qc_amp_base.parent / "QC_mean"
qc_std_base = qc_amp_base.parent / "QC_std"
plot_base = qc_amp_base.parent / "QC_hvsr_meanstd"

# Create output folders
for base in [qc_mean_base, qc_std_base, plot_base]:
    base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS STATIONS --------------------
for station_dir in sorted(qc_amp_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))

    # Group by date
    daily_files = defaultdict(list)
    for file in files:
        parts = file.stem.split("_")
        if len(parts) >= 3:
            date = parts[1]
            daily_files[date].append(file)

    mean_dir = qc_mean_base / station_name
    std_dir = qc_std_base / station_name
    plot_dir = plot_base / station_name
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- DAILY PROCESSING --------------------
    for date, file_list in sorted(daily_files.items()):
        amplitudes = []

        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, names=['frequency', 'amplitude'])
                if df.shape[0] == 256:
                    amplitudes.append(df['amplitude'].values)
            except Exception as e:
                print(f"⚠️ Skipping {file.name}: {e}")

        if not amplitudes:
            print(f"⚠️ No valid QC files for {station_name} on {date}")
            continue

        data = np.vstack(amplitudes)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)

        # Save mean & std
        pd.DataFrame({'amplitude': mean_curve}).to_csv(mean_dir / f"{station_name}_{date}_mean.csv", index=False)
        pd.DataFrame({'amplitude': std_curve}).to_csv(std_dir / f"{station_name}_{date}_std.csv", index=False)
        print(f"✅ Saved mean & std for {station_name} {date}")

        # -------------------- PLOT --------------------
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_ylim(0, 12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("HVSR Amplitude")
        ax.set_title(f"{station_name} — {date} QC HVSR")

        # Plot each hourly curve
        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, names=['frequency', 'amplitude'])
                ax.plot(df['frequency'], df['amplitude'], color='gray', alpha=0.6, linewidth=0.8)
            except Exception as e:
                print(f"⚠️ Could not plot {file.name}: {e}")

        freq = df['frequency'].values  # Use last loaded file (they all share the same freq)
        # Add dashed bounds only
        ax.plot(freq, mean_curve + 2 * std_curve, color='black', linestyle='--', linewidth=1.5, label='Mean ± 2 Std')
        ax.plot(freq, mean_curve - 2 * std_curve, color='black', linestyle='--', linewidth=1.5)

        ax.legend(loc='upper left')
        fig_file = plot_dir / f"{station_name}_{date}_QC_HVSR.pdf"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Saved QC plot: {fig_file}")

print("🎉 Done computing mean/std and plotting QC HVSR curves with ±2 std bounds only.")




# %%
## lets see mean +- 3std
# now we will compute and extract the mean and std as bounds for the QC hvsrpy 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -------------------- PATHS --------------------
qc_amp_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/QC_amplitudes")
qc_mean_base = qc_amp_base.parent / "QC_mean"
qc_std_base = qc_amp_base.parent / "QC_std"
plot_base = qc_amp_base.parent / "QC_hvsr_mean3std"

# Create output folders
for base in [qc_mean_base, qc_std_base, plot_base]:
    base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS STATIONS --------------------
for station_dir in sorted(qc_amp_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))

    # Group by date
    daily_files = defaultdict(list)
    for file in files:
        parts = file.stem.split("_")
        if len(parts) >= 3:
            date = parts[1]
            daily_files[date].append(file)

    mean_dir = qc_mean_base / station_name
    std_dir = qc_std_base / station_name
    plot_dir = plot_base / station_name
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- DAILY PROCESSING --------------------
    for date, file_list in sorted(daily_files.items()):
        amplitudes = []

        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, names=['frequency', 'amplitude'])
                if df.shape[0] == 256:
                    amplitudes.append(df['amplitude'].values)
            except Exception as e:
                print(f"⚠️ Skipping {file.name}: {e}")

        if not amplitudes:
            print(f"⚠️ No valid QC files for {station_name} on {date}")
            continue

        data = np.vstack(amplitudes)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)

        # Save mean & std
        pd.DataFrame({'amplitude': mean_curve}).to_csv(mean_dir / f"{station_name}_{date}_mean.csv", index=False)
        pd.DataFrame({'amplitude': std_curve}).to_csv(std_dir / f"{station_name}_{date}_std.csv", index=False)
        print(f"✅ Saved mean & std for {station_name} {date}")

        # -------------------- PLOT --------------------
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_ylim(0, 12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("HVSR Amplitude")
        ax.set_title(f"{station_name} — {date} QC HVSR")

        # Plot each hourly curve
        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, names=['frequency', 'amplitude'])
                ax.plot(df['frequency'], df['amplitude'], color='gray', alpha=0.6, linewidth=0.8)
            except Exception as e:
                print(f"⚠️ Could not plot {file.name}: {e}")

        freq = df['frequency'].values  # Use last loaded file (they all share the same freq)
        # Add dashed bounds only
        ax.plot(freq, mean_curve + 3 * std_curve, color='black', linestyle='--', linewidth=1.5, label='Mean ± 3 Std')
        ax.plot(freq, mean_curve - 3 * std_curve, color='black', linestyle='--', linewidth=1.5)

        ax.legend(loc='upper left')
        fig_file = plot_dir / f"{station_name}_{date}_QC_HVSR.pdf"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Saved QC plot: {fig_file}")

print("🎉 Done computing mean/std and plotting QC HVSR curves with ±3 std bounds only.")




# %%
# we will reiterate for a seocnd time to ensure we have a convergence and just a mean and std of the good hours 
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- BASE PATHS --------------------
amplitude_base = Path("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/thirty_days/QC/2nd_good_amplitudes")
output_base = Path("/Users/birotimi/Downloads/2nd_stationwide_meanstd")
output_base.mkdir(parents=True, exist_ok=True)

# -------------------- PROCESS EACH STATION --------------------
for station_dir in sorted(amplitude_base.iterdir()):
    if not station_dir.is_dir():
        continue

    station_name = station_dir.name
    print(f"\n📡 Processing station: {station_name}")

    csv_files = sorted(station_dir.glob(f"{station_name}_*_output_hvsrpy_az.csv"))
    if not csv_files:
        print(f"⚠️ No CSV files for station {station_name}")
        continue

    freq = None
    stack = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if df.shape[0] != 256 or "Frequency" not in df.columns or "Amplitude" not in df.columns:
                print(f"⚠️ Skipping {file.name} — incorrect shape or missing columns")
                continue

            if freq is None:
                freq = df["Frequency"].values
            stack.append(df["Amplitude"].values)
        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    if not stack or freq is None:
        print(f"⚠️ No valid data for {station_name}")
        continue

    data = np.vstack(stack)
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)

    # Save output CSV
    output_file = output_base / f"{station_name}_meanstd.csv"
    pd.DataFrame({
        "Frequency": freq,
        "Mean": mean_curve,
        "Std": std_curve
    }).to_csv(output_file, index=False)

    print(f"✅ Saved mean/std: {output_file}")



