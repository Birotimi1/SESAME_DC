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

# QC output directories
QC_BASE = BASE_PATH / "QC_out"
GOOD_MINISEED = QC_BASE / "Goodminiseed"
BAD_MINISEED = QC_BASE / "Badminiseed"
RMS_SUMMARY = QC_BASE / "RMS_summary"

# HVSR processing outputs
# THIRTY_DAYS_FIGURES = BASE_PATH / "hvsrpy" / "figures"
THIRTY_DAYS_HVSRPY = BASE_PATH / "hvsrpy" / "output_files"
# HVSRPY_SUMMARY = BASE_PATH / "hvsrpy" / "hvsrpysummary"

# Plots and analysis
PLOTS_BASE = QC_BASE / "figures"
AMPLITUDES = QC_BASE / "amplitudes"
GOOD_HOURS = QC_BASE / "good_hours"
BAD_HOURS = QC_BASE / "bad_hours"

# Statistical outputs
MEAN_AMPLITUDES = QC_BASE / "mean_amplitudes"
STD_AMPLITUDES = QC_BASE / "std_amplitudes"
HVSR_DIFF = QC_BASE / "hvsr_diff"
HVSRDIFF_SQUARED = QC_BASE / "hvsrdiff_squared"
STD_SQUARED = QC_BASE / "std_squared"
MISFIT = QC_BASE / "misfit"
AVERAGED_MISFIT = QC_BASE / "averaged_misfit"

# Plotting outputs
AVERAGED_MISFIT_PLOT = PLOTS_BASE / "averaged_misfit_plot"
LINE_PLOT = PLOTS_BASE / "line_plot"
SCATTER_PLOT = PLOTS_BASE / "scatter_plot"
MISFIT_COLORED_HVSRPY = PLOTS_BASE / "misfit_colored_hvsrpy"

# QC final outputs
QC_AMPLITUDES = QC_BASE / "QC_amplitudes"
QC_MEAN = QC_BASE / "QC_mean"
QC_STD = QC_BASE / "QC_std"
QC_HVSR_MEANSTD = QC_BASE / "QC_hvsr_meanstd"
QC_HVSR_MEAN3STD = QC_BASE / "QC_hvsr_mean3std"

# Second iteration outputs
SECOND_GOOD_AMPLITUDES = BASE_PATH / "2nd_good_amplitudes"
SECOND_STATIONWIDE_MEANSTD = BASE_PATH / "2nd_stationwide_meanstd"

# Filter parameters
FREQMIN = 0.1  # Hz
FREQMAX = 10.0  # Hz

print(f"Base path configured: {BASE_PATH}")

# Create all necessary directories
print("\nCreating directory structure...")
directories_to_create = [
    QC_BASE, GOOD_MINISEED, BAD_MINISEED, RMS_SUMMARY, 
    THIRTY_DAYS_HVSRPY,
    PLOTS_BASE, AMPLITUDES, GOOD_HOURS, BAD_HOURS,
    MEAN_AMPLITUDES, STD_AMPLITUDES, HVSR_DIFF, HVSRDIFF_SQUARED,
    STD_SQUARED, MISFIT, AVERAGED_MISFIT, AVERAGED_MISFIT_PLOT,
    LINE_PLOT, SCATTER_PLOT, MISFIT_COLORED_HVSRPY,
    QC_AMPLITUDES, QC_MEAN, QC_STD, QC_HVSR_MEANSTD, QC_HVSR_MEAN3STD,
    SECOND_GOOD_AMPLITUDES, SECOND_STATIONWIDE_MEANSTD
]

for directory in directories_to_create:
    directory.mkdir(parents=True, exist_ok=True)

print("All directories created successfully!")

# %%
# ==================== PLOT DAILY HVSR ====================
print("\n" + "="*60)
print("Plotting Daily HVSR")
print("="*60)

color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#d17a22', '#2171b5', '#7f7f7f', '#bcbd22', '#17becf',
    '#8c564b', '#e377c2', '#ffbf00', '#aa0b4b', '#009688'
]

# Pattern: STATION_YYYYMMDD_HH_output_hvsrpy_az.csv
filename_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_output_hvsrpy_az\.csv$')

for station_dir in sorted(THIRTY_DAYS_HVSRPY.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    output_dir = PLOTS_BASE / station_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(station_dir.glob("*.csv"))
    daily_files = {}
    
    for csv_file in csv_files:
        match = filename_pattern.match(csv_file.name)
        if not match:
            continue
        
        _, date_part, hour_part = match.groups()
        daily_files.setdefault(date_part, {})[hour_part] = csv_file
    
    for date_part, hourly_files in sorted(daily_files.items()):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('HVSR Amplitude')
        ax.set_title(f"{station_name} - {date_part}")
        ax.set_ylim(0, 12)
        
        color_cycle = cycle(color_palette)
        
        for hour in sorted(hourly_files.keys()):
            try:
                df = pd.read_csv(hourly_files[hour], comment='#', header=None, 
                               names=['frequency', 'amplitude'])
                ax.plot(df['frequency'], df['amplitude'], linewidth=2, 
                       color=next(color_cycle), label=f'Hour {hour}')
            except Exception as e:
                print(f"Error plotting {hourly_files[hour].name}: {e}")
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        output_file = output_dir / f"{station_name}_{date_part}_hourly_HVSR.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Daily HVSR plotting complete.")

# %%
# ==================== EXTRACT AMPLITUDES ====================
print("\n" + "="*60)
print("Extracting Amplitudes")
print("="*60)

for station_dir in sorted(THIRTY_DAYS_HVSRPY.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    station_amplitude_dir = AMPLITUDES / station_name
    station_amplitude_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(station_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        match = filename_pattern.match(csv_file.name)
        if not match:
            continue
        
        _, date_str, hour_str = match.groups()
        
        try:
            df = pd.read_csv(csv_file, comment='#', header=None, usecols=[1], 
                           names=['amplitude'])
            
            if df.shape[0] != 256:
                print(f"Skipping {csv_file.name}: {df.shape[0]} rows")
                continue
            
            output_filename = f"{station_name}_{date_str}_{hour_str}_amplitudes.csv"
            output_path = station_amplitude_dir / output_filename
            df.to_csv(output_path, index=False)
        
        except Exception as e:
            print(f"Error: {e}")

print("Amplitude extraction complete.")

# %%
# ==================== STEP 3: MAX AMPLITUDE CHECK ====================
print("\n" + "="*60)
print("STEP 3: Maximum Amplitude Check")
print("="*60)

amp_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_amplitudes\.csv$')
bad_hours = []

for station_dir in sorted(AMPLITUDES.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    good_station_path = GOOD_HOURS / station_name
    bad_station_path = BAD_HOURS / station_name
    good_station_path.mkdir(parents=True, exist_ok=True)
    bad_station_path.mkdir(parents=True, exist_ok=True)
    
    for csv_file in sorted(station_dir.glob("*.csv")):
        match = amp_pattern.match(csv_file.name)
        if not match:
            continue
        
        _, date_str, hour_str = match.groups()
        
        try:
            df = pd.read_csv(csv_file)
            
            if df.shape[0] != 256 or 'amplitude' not in df.columns:
                continue
            
            max_amp = df['amplitude'].max()
            
            if 1 < max_amp < 100:
                df.to_csv(good_station_path / csv_file.name, index=False)
                print(f"Good: {csv_file.name}")
            else:
                df.to_csv(bad_station_path / csv_file.name, index=False)
                print(f"Bad: {csv_file.name} (max={max_amp:.2f})")
                bad_hours.append({
                    'station': station_name,
                    'date': date_str,
                    'hour': hour_str,
                    'max_amplitude': max_amp,
                    'file': csv_file.name
                })
        
        except Exception as e:
            print(f"Error: {e}")

if bad_hours:
    bad_hours_df = pd.DataFrame(bad_hours)
    report_path = GOOD_HOURS / "bad_hours_report.csv"
    bad_hours_df.to_csv(report_path, index=False)
    print(f"\nBad hour report: {report_path}")

print("Amplitude QC complete.")

# %%
# ==================== COMPUTE MEAN AND STD ====================
print("\n" + "="*60)
print("Computing Mean and Std")
print("="*60)

for station_dir in sorted(GOOD_HOURS.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    csv_files = sorted(station_dir.glob("*.csv"))
    
    daily_files = {}
    for file in csv_files:
        match = amp_pattern.match(file.name)
        if match:
            date = match.group(2)
            daily_files.setdefault(date, []).append(file)
    
    mean_dir = MEAN_AMPLITUDES / station_name
    std_dir = STD_AMPLITUDES / station_name
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    
    for date, files in daily_files.items():
        stack = []
        
        for file in files:
            try:
                df = pd.read_csv(file)
                if df.shape[0] == 256 and 'amplitude' in df.columns:
                    stack.append(df['amplitude'].values)
            except:
                pass
        
        if not stack:
            continue
        
        data = np.vstack(stack)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)
        
        mean_file = mean_dir / f"{station_name}_{date}_mean_amplitudes.csv"
        std_file = std_dir / f"{station_name}_{date}_std_amplitudes.csv"
        
        pd.DataFrame({'amplitude': mean_curve}).to_csv(mean_file, index=False)
        pd.DataFrame({'amplitude': std_curve}).to_csv(std_file, index=False)
        
        print(f"Saved mean/std: {station_name} {date}")

print("Mean/Std computation complete.")



# %%
# ==================== COMPUTE HVSR DIFFERENCE ====================
print("\n" + "="*60)
print("Computing HVSR Difference")
print("="*60)

for station_dir in sorted(GOOD_HOURS.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    mean_station_dir = MEAN_AMPLITUDES / station_name
    diff_station_dir = HVSR_DIFF / station_name
    diff_station_dir.mkdir(parents=True, exist_ok=True)
    
    amp_files = sorted(station_dir.glob("*.csv"))
    
    for amp_file in amp_files:
        match = amp_pattern.match(amp_file.name)
        if not match:
            continue
        
        _, date, hour = match.groups()
        mean_file = mean_station_dir / f"{station_name}_{date}_mean_amplitudes.csv"
        
        if not mean_file.exists():
            continue
        
        try:
            df_amp = pd.read_csv(amp_file)
            df_mean = pd.read_csv(mean_file)
            
            if df_amp.shape[0] != 256 or df_mean.shape[0] != 256:
                continue
            
            hvsr_diff = df_amp['amplitude'] - df_mean['amplitude']
            df_diff = pd.DataFrame({'hvsr_diff': hvsr_diff})
            
            out_file = diff_station_dir / f"{station_name}_{date}_{hour}_hvsr_diff.csv"
            df_diff.to_csv(out_file, index=False)
            print(f"Saved: {out_file.name}")
        
        except Exception as e:
            print(f"Error: {e}")

print("HVSR difference complete.")

# %%
# Create directories
HVSRDIFF_SQUARED.mkdir(parents=True, exist_ok=True)

# ==================== SQUARE HVSR DIFFERENCES ====================
print("\n" + "="*60)
print("Squaring HVSR Differences")
print("="*60)


diff_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_hvsr_diff\.csv$')

total_processed = 0
total_skipped = 0
total_errors = 0

for station_dir in sorted(HVSR_DIFF.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    print(f"\n[INFO] Processing station: {station_name}")
    
    output_station_dir = HVSRDIFF_SQUARED / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)
    
    station_processed = 0
    station_skipped = 0
    station_errors = 0
    
    for diff_file in station_dir.glob("*.csv"):
        match = diff_pattern.match(diff_file.name)
        if not match:
            print(f"  [SKIP] {diff_file.name} - doesn't match pattern")
            station_skipped += 1
            continue
        
        _, date, hour = match.groups()
        
        try:
            df = pd.read_csv(diff_file)
            
            # Validate data
            if "hvsr_diff" not in df.columns:
                print(f"  [SKIP] {diff_file.name} - missing 'hvsr_diff' column")
                station_skipped += 1
                continue
            
            if df.shape[0] != 256:
                print(f"  [SKIP] {diff_file.name} - expected 256 rows, got {df.shape[0]}")
                station_skipped += 1
                continue
            
            # Square the differences
            df_squared = pd.DataFrame({'hvsrdiff_squared': df['hvsr_diff'] ** 2})
            
            # Save output
            output_file = output_station_dir / f"{station_name}_{date}_{hour}_hvsrdiff_squared.csv"
            df_squared.to_csv(output_file, index=False)
            print(f"  ✅ Saved: {output_file.name}")
            station_processed += 1
        
        except Exception as e:
            print(f"  ❌ Error processing {diff_file.name}: {e}")
            station_errors += 1
    
    # Station summary
    print(f"  Station {station_name} summary: {station_processed} processed, {station_skipped} skipped, {station_errors} errors")
    
    total_processed += station_processed
    total_skipped += station_skipped
    total_errors += station_errors

# Final summary
print("\n" + "="*60)
print("✅ HVSR Squared Difference Complete")
print("="*60)
print(f"\n📊 SUMMARY:")
print(f"  Total processed: {total_processed}")
print(f"  Total skipped: {total_skipped}")
print(f"  Total errors: {total_errors}")
print(f"\n📁 Output directory: {HVSRDIFF_SQUARED}")

# %%
# ==================== SQUARE STANDARD DEVIATION ====================

# Create directories
STD_SQUARED.mkdir(parents=True, exist_ok=True)

# ==================== SQUARE STANDARD DEVIATION ====================
print("\n" + "="*60)
print("Squaring Standard Deviation")
print("="*60)

# 
std_pattern = re.compile(r'^(.+)_(\d{8})_std_amplitudes\.csv$')

total_processed = 0
total_skipped = 0
total_errors = 0

for station_dir in sorted(STD_AMPLITUDES.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    print(f"\n[INFO] Processing station: {station_name}")
    
    output_station_dir = STD_SQUARED / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)
    
    station_processed = 0
    station_skipped = 0
    station_errors = 0
    
    for std_file in station_dir.glob("*.csv"):
        match = std_pattern.match(std_file.name)
        if not match:
            print(f"  [SKIP] {std_file.name} - doesn't match pattern")
            station_skipped += 1
            continue
        
        _, date = match.groups()
        
        try:
            df = pd.read_csv(std_file)
            
            # Validate data
            if "amplitude" not in df.columns:
                print(f"  [SKIP] {std_file.name} - missing 'amplitude' column")
                station_skipped += 1
                continue
            
            if df.shape[0] != 256:
                print(f"  [SKIP] {std_file.name} - expected 256 rows, got {df.shape[0]}")
                station_skipped += 1
                continue
            
            # Square the standard deviation
            df_squared = pd.DataFrame({'std_squared': df['amplitude'] ** 2})
            
            # Save output
            output_file = output_station_dir / f"{station_name}_{date}_std_squared.csv"
            df_squared.to_csv(output_file, index=False)
            print(f"  ✅ Saved: {output_file.name}")
            station_processed += 1
        
        except Exception as e:
            print(f"  ❌ Error processing {std_file.name}: {e}")
            station_errors += 1
    
    # Station summary
    print(f"  Station {station_name} summary: {station_processed} processed, {station_skipped} skipped, {station_errors} errors")
    
    total_processed += station_processed
    total_skipped += station_skipped
    total_errors += station_errors

# Final summary
print("\n" + "="*60)
print("✅ Standard Deviation Squaring Complete")
print("="*60)
print(f"\n📊 SUMMARY:")
print(f"  Total processed: {total_processed}")
print(f"  Total skipped: {total_skipped}")
print(f"  Total errors: {total_errors}")
print(f"\n📁 Output directory: {STD_SQUARED}")

# %%
# ==================== COMPUTE MISFIT ====================
print("\n" + "="*60)
print("Computing Misfit")
print("="*60)

squared_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_hvsrdiff_squared\.csv')

for station_dir in sorted(HVSRDIFF_SQUARED.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    std_station_dir = STD_SQUARED / station_name
    output_station_dir = MISFIT / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)
    
    for diff_file in station_dir.glob("*.csv"):
        match = squared_pattern.match(diff_file.name)
        if not match:
            continue
        
        _, date, hour = match.groups()
        std_file = std_station_dir / f"{station_name}_{date}_std_squared.csv"
        
        if not std_file.exists():
            continue
        
        try:
            df_diff = pd.read_csv(diff_file)
            df_std = pd.read_csv(std_file)
            
            if df_diff.shape[0] != 256 or df_std.shape[0] != 256:
                continue
            
            misfit = df_diff['hvsrdiff_squared'] / df_std['std_squared']
            df_misfit = pd.DataFrame({'misfit': misfit})
            
            output_file = output_station_dir / f"{station_name}_{date}_{hour}_misfit.csv"
            df_misfit.to_csv(output_file, index=False)
            print(f"Saved: {output_file.name}")
        
        except Exception as e:
            print(f"Error: {e}")

print("Misfit computation complete.")

# %%
# ==================== AVERAGE MISFIT ====================
print("\n" + "="*60)
print("Averaging Misfit")
print("="*60)

misfit_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_misfit\.csv')

for station_dir in sorted(MISFIT.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    output_station_dir = AVERAGED_MISFIT / station_name
    output_station_dir.mkdir(parents=True, exist_ok=True)
    
    for misfit_file in station_dir.glob("*.csv"):
        match = misfit_pattern.match(misfit_file.name)
        if not match:
            continue
        
        _, date, hour = match.groups()
        
        try:
            df = pd.read_csv(misfit_file)
            
            if "misfit" not in df.columns or df.shape[0] != 256:
                continue
            
            avg_misfit = df["misfit"].mean()
            out_file = output_station_dir / f"{station_name}_{date}_{hour}_averaged_misfit.csv"
            pd.DataFrame({'averaged_misfit': [avg_misfit]}).to_csv(out_file, index=False)
            print(f"Saved: {out_file.name}")
        
        except Exception as e:
            print(f"Error: {e}")

print("Averaged misfit complete.")

# %%
# ==================== PLOT HOURLY MISFIT ====================
print("\n" + "="*60)
print("Plotting Hourly Misfit")
print("="*60)

avg_misfit_pattern = re.compile(r'^(.+)_(\d{8})_(\d{2})_averaged_misfit\.csv')

for station_dir in sorted(AVERAGED_MISFIT.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))
    
    daily_data = defaultdict(dict)
    for file in files:
        match = avg_misfit_pattern.match(file.name)
        if not match:
            continue
        
        _, date, hour = match.groups()
        
        try:
            df = pd.read_csv(file)
            if 'averaged_misfit' in df.columns and df.shape[0] == 1:
                daily_data[date][int(hour)] = df['averaged_misfit'].values[0]
        except:
            pass
    
    line_dir = LINE_PLOT / station_name
    scatter_dir = SCATTER_PLOT / station_name
    line_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    for date, hour_dict in sorted(daily_data.items()):
        hours = list(range(24))
        values = [hour_dict.get(h, None) for h in hours]
        available_hours = [h for h in hours if h in hour_dict]
        available_vals = [hour_dict[h] for h in available_hours]
        
        # Line Plot
        plt.figure(figsize=(10, 3))
        plt.plot(hours, values, marker='o', linestyle='-', color='blue')
        for h in hours:
            if h not in hour_dict:
                plt.scatter(h, 0, color='red', s=50)
        plt.title(f"{station_name} - {date} (Line)")
        plt.xlabel("Hour")
        plt.ylabel("Average Misfit")
        plt.xticks(range(24))
        plt.grid(True)
        plt.ylim(bottom=0)
        line_file = line_dir / f"{station_name}_{date}_hourly_misfit_line.pdf"
        plt.savefig(line_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter Plot
        plt.figure(figsize=(10, 3))
        plt.scatter(available_hours, available_vals, color='green', s=50)
        missing_hours = [h for h in hours if h not in hour_dict]
        if missing_hours:
            plt.scatter(missing_hours, [0]*len(missing_hours), color='red', s=50)
        plt.title(f"{station_name} - {date} (Scatter)")
        plt.xlabel("Hour")
        plt.ylabel("Average Misfit")
        plt.xticks(range(24))
        plt.grid(True)
        plt.ylim(bottom=0)
        scatter_file = scatter_dir / f"{station_name}_{date}_hourly_misfit_scatter.pdf"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()

print("Hourly misfit plots complete.")

# %%
# ==================== PLOT MISFIT-COLORED HVSR & EXTRACT QC ====================
print("\n" + "="*60)
print("Plotting Misfit-Colored HVSR")
print("="*60)

for station_dir in sorted(THIRTY_DAYS_HVSRPY.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    misfit_dir = AVERAGED_MISFIT / station_name
    output_dir = MISFIT_COLORED_HVSRPY / station_name
    qc_csv_dir = QC_AMPLITUDES / station_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_csv_dir.mkdir(parents=True, exist_ok=True)
    
    hvsr_files = sorted(station_dir.glob("*.csv"))
    
    daily_files = {}
    for file in hvsr_files:
        match = filename_pattern.match(file.name)
        if match:
            _, date_str, hour_str = match.groups()
            daily_files.setdefault(date_str, {})[hour_str] = file
    
    for date_str, hourly_dict in sorted(daily_files.items()):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('HVSR Amplitude')
        ax.set_title(f"{station_name} - {date_str}")
        ax.set_ylim(0, 12)
        
        for hour_str in sorted(hourly_dict.keys()):
            hvsr_file = hourly_dict[hour_str]
            misfit_file = misfit_dir / f"{station_name}_{date_str}_{hour_str}_averaged_misfit.csv"
            
            try:
                df = pd.read_csv(hvsr_file, comment='#', header=None, 
                               names=['frequency', 'amplitude'])
                
                if misfit_file.exists():
                    mf = pd.read_csv(misfit_file)
                    avg_misfit = mf['averaged_misfit'].values[0]
                    
                    if avg_misfit > 3:
                        color = 'red'
                    else:
                        color = 'blue'
                        shutil.copy(hvsr_file, qc_csv_dir / hvsr_file.name)
                    
                    label = f"H{hour_str} (m={avg_misfit:.2f})"
                else:
                    color = 'green'
                    label = f"H{hour_str} (no misfit)"
                
                ax.plot(df['frequency'], df['amplitude'], linewidth=1.5, 
                       color=color, label=label)
            
            except Exception as e:
                print(f"Error: {e}")
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plot_file = output_dir / f"{station_name}_{date_str}_misfit_colored_HVSR.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Misfit-colored HVSR plots complete.")

# %%
# ==================== COMPUTE QC MEAN AND STD (±2 STD) ====================
print("\n" + "="*60)
print("Computing QC Mean/Std (±2 Std)")
print("="*60)

for station_dir in sorted(QC_AMPLITUDES.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))
    
    daily_files = defaultdict(list)
    for file in files:
        match = filename_pattern.match(file.name)
        if match:
            date = match.group(2)
            daily_files[date].append(file)
    
    mean_dir = QC_MEAN / station_name
    std_dir = QC_STD / station_name
    plot_dir = QC_HVSR_MEANSTD / station_name
    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for date, file_list in sorted(daily_files.items()):
        amplitudes = []
        freq = None
        
        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, 
                               names=['frequency', 'amplitude'])
                if df.shape[0] == 256:
                    amplitudes.append(df['amplitude'].values)
                    if freq is None:
                        freq = df['frequency'].values
            except:
                pass
        
        if not amplitudes:
            continue
        
        data = np.vstack(amplitudes)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)
        
        mean_path = mean_dir / f"{station_name}_{date}_mean.csv"
        std_path = std_dir / f"{station_name}_{date}_std.csv"
        pd.DataFrame({'amplitude': mean_curve}).to_csv(mean_path, index=False)
        pd.DataFrame({'amplitude': std_curve}).to_csv(std_path, index=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xscale('log')
        ax.set_ylim(0, 12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("HVSR Amplitude")
        ax.set_title(f"{station_name} - {date} QC HVSR")
        
        for file in file_list:
            try:
                df = pd.read_csv(file, comment='#', header=None, 
                               names=['frequency', 'amplitude'])
                ax.plot(df['frequency'], df['amplitude'], color='gray', 
                       alpha=0.6, linewidth=0.8)
            except:
                pass
        
        if freq is not None:
            ax.plot(freq, mean_curve + 2 * std_curve, 'k--', linewidth=1.5, 
                   label='Mean ± 2 Std')
            ax.plot(freq, mean_curve - 2 * std_curve, 'k--', linewidth=1.5)
        
        ax.legend()
        fig_file = plot_dir / f"{station_name}_{date}_QC_HVSR_2std.pdf"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved QC: {station_name} {date}")

print("QC Mean/Std (±2) complete.")

# %%
# ==================== COMPUTE QC MEAN AND STD (±3 STD) ====================
print("\n" + "="*60)
print("Computing QC Mean/Std (±3 Std)")
print("="*60)

for station_dir in sorted(QC_AMPLITUDES.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    files = sorted(station_dir.glob("*.csv"))
    
    daily_files = defaultdict(list)
    for file in files:
        match = filename_pattern.match(file.name)
        if match:
            date = match.group(2)
            daily_files[date].append(file)
    
    plot_dir = QC_HVSR_MEAN3STD / station_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    mean_dir = QC_MEAN / station_name
    std_dir = QC_STD / station_name
    
    for date, file_list in sorted(daily_files.items()):
        mean_file = mean_dir / f"{station_name}_{date}_mean.csv"
        std_file = std_dir / f"{station_name}_{date}_std.csv"
        
        if not mean_file.exists() or not std_file.exists():
            continue
        
        try:
            mean_df = pd.read_csv(mean_file)
            std_df = pd.read_csv(std_file)
            mean_curve = mean_df['amplitude'].values
            std_curve = std_df['amplitude'].values
            
            # Get frequency from first file
            df = pd.read_csv(file_list[0], comment='#', header=None, 
                           names=['frequency', 'amplitude'])
            freq = df['frequency'].values
            
            # Plot
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.set_xscale('log')
            ax.set_ylim(0, 12)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("HVSR Amplitude")
            ax.set_title(f"{station_name} - {date} QC HVSR")
            
            for file in file_list:
                try:
                    df = pd.read_csv(file, comment='#', header=None, 
                                   names=['frequency', 'amplitude'])
                    ax.plot(df['frequency'], df['amplitude'], color='gray', 
                           alpha=0.6, linewidth=0.8)
                except:
                    pass
            
            ax.plot(freq, mean_curve + 3 * std_curve, 'k--', linewidth=1.5, 
                   label='Mean ± 3 Std')
            ax.plot(freq, mean_curve - 3 * std_curve, 'k--', linewidth=1.5)
            
            ax.legend()
            fig_file = plot_dir / f"{station_name}_{date}_QC_HVSR_3std.pdf"
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved QC: {station_name} {date}")
        
        except Exception as e:
            print(f"Error: {e}")

print("QC Mean/Std (±3) complete.")

# %%
# ==================== STATIONWIDE MEAN AND STD ====================
print("\n" + "="*60)
print("Computing Stationwide Mean/Std")
print("="*60)

SECOND_STATIONWIDE_MEANSTD.mkdir(parents=True, exist_ok=True)

for station_dir in sorted(QC_AMPLITUDES.iterdir()):
    if not station_dir.is_dir():
        continue
    
    station_name = station_dir.name
    csv_files = sorted(station_dir.glob("*.csv"))
    
    if not csv_files:
        continue
    
    freq = None
    stack = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file, comment='#', header=None, 
                           names=['frequency', 'amplitude'])
            if df.shape[0] == 256:
                if freq is None:
                    freq = df['frequency'].values
                stack.append(df['amplitude'].values)
        except:
            pass
    
    if not stack or freq is None:
        continue
    
    data = np.vstack(stack)
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)
    
    output_file = SECOND_STATIONWIDE_MEANSTD / f"{station_name}_meanstd.csv"
    pd.DataFrame({
        "Frequency": freq,
        "Mean": mean_curve,
        "Std": std_curve
    }).to_csv(output_file, index=False)
    
    print(f"Saved: {output_file.name}")

print("\n" + "="*60)
print("ALL PROCESSING COMPLETE")
print("="*60)