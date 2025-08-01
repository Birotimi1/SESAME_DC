# %%
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hvsrpy
from hvsrpy import utils
from obspy import read

from pathlib import Path
import os

from matplotlib import cm
# from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature



# %%


# %%
# from obspy import read
#from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

#client = Client("IRIS")
#net = "Z9"  # network of the station
#sta = "E27"  # station code
## sta = "E18"  # station code
#loc = "**"  # to specify the instrument at the station
#chan = "BH*"

#starttime = UTCDateTime("2012-09-01T00:00:00")
#endtime = starttime + 60 * 60 * 24  # 24 hours
## endtime = starttime + 60 * 60  # 24 hours

#st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime, endtime=endtime, 
                        #   attach_response=True
                          
#print(st)


client = Client ("IRIS")
minlatitude = 26.0
maxlatitude = 49.0

minlongitude =  -120
maxlongitude =  -60

net = "LI"
sta = "**"
loc = "**"
chan = "BH*"
client = Client("IRIS")
starttime = UTCDateTime("2010-01-01")
endtime = UTCDateTime("2014-12-02")
#startafter= UTCDateTime ("2024-01-20")
#endafter= UTCDateTime ("2024-06-30")


clt = client.get_stations(network="LI", station="**",
                                starttime=starttime,minlatitude=minlatitude, maxlatitude=maxlatitude,
                                minlongitude=minlongitude, maxlongitude=maxlongitude,
                                endtime=endtime) #matchtimeseries = True)

print (clt)

#st.plot(projection="local",label=False,color_per_network=True) 


# # Remove instrument response
# sr = st[0].stats.sampling_rate
# st.remove_response(output='ACC', zero_mean=True, taper=True, taper_fraction=0.05, pre_filt=[0.001, 0.005, sr/3, sr/2], water_level=600)

# Save waveform
#st.write("Apriltoday.mseed", format="MSEED")

#st.plot()

# %%
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os

Ndays = 1  # Duration of each download period (1 day)
total_days = 4  # Total number of successive days to download

client = Client("IRIS")
net = "LI"
sta = "**"
loc = "**"
chan = "BH?"

# Ensure the base output directory exists
base_output_dir = "outputloopdaysnew"
os.makedirs(base_output_dir, exist_ok=True)

for network in clt:
    for station in network:
        # Get station and network codes
        sta = station.code
        net = network.code
        
        try:
            # Initial start and end times
            initial_starttime = UTCDateTime((station.creation_date + 365*24*60*60).strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Error in date conversion for station {station.code}: {e}")
            continue
        
        for day_offset in range(total_days):
            # Calculate start and end time for the current day
            starttime = initial_starttime + day_offset * 24 * 60 * 60
            endtime = starttime + Ndays * 24 * 60 * 60

            # Create a separate folder for each day within the station's directory
            station_output_dir = os.path.join(base_output_dir, f"{net}.{sta}")
            day_output_dir = os.path.join(station_output_dir, starttime.strftime('%Y-%m-%d'))
            os.makedirs(day_output_dir, exist_ok=True)

            output_file = os.path.join(day_output_dir, f"{net}.{sta}.{starttime.strftime('%Y-%m-%d')}.mseed")

            # Skip download if file already exists
            if os.path.exists(output_file):
                print(f"File {output_file} already exists, skipping download.")
                continue

            print(f"Processing station: {sta}, Day: {day_offset + 1}")

            try:
                st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime, endtime=endtime)
            except Exception as e:
                print(f"Error downloading data for station {sta}: {e}. Trying alternative dates...")
                try:
                    st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
                except Exception as e2:
                    print(f"Error downloading data for alternative dates for station {sta}: {e2}")
                    continue

            # Check for gaps and re-download if necessary
            if len(st.get_gaps()) > 0:
                print(f"Gaps exist in waveforms for station {sta}... trying alternative dates again")
                try:
                    st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
                except Exception as e:
                    print(f"Error re-downloading data for station {sta}: {e}")
                    continue

                if len(st.get_gaps()) > 0:
                    print(f"Still gaps exist for station {sta}, skipping")
                    continue

            # Write the waveforms to a file
            st.write(output_file, format="MSEED")
            print(f"Waveforms written to {output_file}")

print("Processing complete.")


# %%
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os

Ndays = 1  # Duration of each download period (1 day)
total_days = 10  # Total number of successive days to download

client = Client("IRIS")
net = "LI"
sta = "**"
loc = "**"
chan = "BH?"

# Ensure the base output directory exists
base_output_dir = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/LIGO/hourlongdata"
os.makedirs(base_output_dir, exist_ok=True)

for network in clt:
    for station in network:
        # Get station and network codes
        sta = station.code
        net = network.code
        
        try:
            # Initial start and end times
            initial_starttime = UTCDateTime((station.creation_date + 365*24*60*60).strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Error in date conversion for station {station.code}: {e}")
            continue
        
        for day_offset in range(total_days):
            # Calculate start and end time for the current day
            starttime = initial_starttime + day_offset * 24 * 60 * 60
            endtime = starttime + Ndays * 24 * 60 * 60

            # Create a separate folder for each day within the station's directory
            station_output_dir = os.path.join(base_output_dir, f"{net}.{sta}")
            day_output_dir = os.path.join(station_output_dir, f"day{day_offset + 1}")
            os.makedirs(day_output_dir, exist_ok=True)

            output_file = os.path.join(day_output_dir, f"{net}.{sta}.day{day_offset + 1}.mseed")

            # Skip download if file already exists
            if os.path.exists(output_file):
                print(f"File {output_file} already exists, skipping download.")
                continue

            print(f"Processing station: {sta}, Day: {day_offset + 1}")

            try:
                st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime, endtime=endtime)
            except Exception as e:
                print(f"Error downloading data for station {sta}: {e}. Trying alternative dates...")
                try:
                    st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
                except Exception as e2:
                    print(f"Error downloading data for alternative dates for station {sta}: {e2}")
                    continue

            # Check for gaps and re-download if necessary
            if len(st.get_gaps()) > 0:
                print(f"Gaps exist in waveforms for station {sta}... trying alternative dates again")
                try:
                    st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
                except Exception as e:
                    print(f"Error re-downloading data for station {sta}: {e}")
                    continue

                if len(st.get_gaps()) > 0:
                    print(f"Still gaps exist for station {sta}, skipping")
                    continue

            # Write the waveforms to a file
            st.write(output_file, format="MSEED")
            print(f"Waveforms written to {output_file}")

print("Processing complete.")


# %%
# st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime, endtime=endtime)
t1 = UTCDateTime("2010-12-27T00:00:00.000")
# t1 = UTCDateTime("2010-02-27T06:30:00.000")
t2 = t1 + 5
# st = client.get_waveforms("IU", "ANMO", "**", "LHZ", t1, t2)
st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=t1, endtime=t2)


# %%
starttime

# %%
Ndays = 1


client = Client("IRIS")
#t1 = (station.creation_date + 1*365*24*60*60).strftime('%Y-%m-%d')
#t2 = (station.creation_date + 1*365*24*60*60 + Ndays*24*60*60).strftime('%Y-%m-%d')
net = "Z9"
sta = "**"
loc = "**"
chan = "BH?"
#t1 = UTCDateTime ("2012-10-01")
#t2 = UTCDateTime ("2012-10-02")

#starttime  = UTCDateTime("2012-02-27T00:00:00.000")
#endtime = starttime +60 *60 *24

   
for network in clt:
    for station in network:
        # Compute HVSR for each station
        
        starttime =  UTCDateTime((station.creation_date + 1*365*24*60*60).strftime('%Y-%m-%d'))
        endtime =  UTCDateTime((station.creation_date + 1*365*24*60*60 + Ndays*24*60*60).strftime('%Y-%m-%d'))
        
        sta = station.code
        #loc = '--'
        net = network.code
        print (sta)

        try:
            st = client.get_waveforms(network= net , station= sta, location= loc, channel= chan, starttime= starttime, endtime= endtime) #attach_response=True)    
        except:
            print('Error downloading data... try subtracting 6 months from start/end time')
            st = client.get_waveforms(network= net , station= sta, location= loc, channel= chan, starttime= starttime-60*60*24*30*6, endtime= endtime-60*60*24*30*6) #attach_response=True)
        
        # Merge waveforms if gaps exist
        if len(st.get_gaps())>0:
            print('Gaps exist in waveforms... try subtracting 6 months from start/end time')
            # st.merge(method=0, fill_value=None)
            st = client.get_waveforms(network= net , station= sta, location= loc, channel= chan, starttime= starttime-60*60*24*30*6, endtime= endtime-60*60*24*30*6) #attach_response=True)
            
            if len(st.get_gaps())>0:
                continue

        st.write (net+'.'+sta+'.'+starttime.strftime('%Y-%m-%d')+'.'+endtime.strftime('%Y-%m-%d')+".mseed", format ="MSEED")
            
print (st)






# %%
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os

Ndays = 1

client = Client("IRIS")
net = "Z9"
sta = "**"
loc = "**"
chan = "BH?"

# Ensure the output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for network in clt:
    for station in network:
        # Compute HVSR for each station
        
        try:
            # Convert creation_date to datetime, add one year and then the number of days
            starttime = UTCDateTime((station.creation_date + 365*24*60*60).strftime('%Y-%m-%d'))
            endtime = UTCDateTime((station.creation_date + 365*24*60*60 + Ndays*24*60*60).strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Error in date conversion for station {station.code}: {e}")
            continue

        sta = station.code
        net = network.code
        output_file = os.path.join(output_dir, f"{net}.{sta}.{starttime.strftime('%Y-%m-%d')}.{endtime.strftime('%Y-%m-%d')}.mseed")

        # Skip download if file already exists
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping download.")
            continue

        print(f"Processing station: {sta}")

        try:
            st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime, endtime=endtime)
        except Exception as e:
            print(f"Error downloading data for station {sta}: {e}. Trying alternative dates...")
            try:
                st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
            except Exception as e2:
                print(f"Error downloading data for alternative dates for station {sta}: {e2}")
                continue
        
        # Check for gaps and re-download if necessary
        if len(st.get_gaps()) > 0:
            print(f"Gaps exist in waveforms for station {sta}... trying alternative dates again")
            try:
                st = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=starttime-60*60*24*30*6, endtime=endtime-60*60*24*30*6)
            except Exception as e:
                print(f"Error re-downloading data for station {sta}: {e}")
                continue

            if len(st.get_gaps()) > 0:
                print(f"Still gaps exist for station {sta}, skipping")
                continue

        # Write the waveforms to a file
        st.write(output_file, format="MSEED")
        print(f"Waveforms written to {output_file}")

print("Processing complete.")


# %%
starttime


# %%
# Input file name (may be a relative or full path).

# file_name = "UT.STN11.A2_C50.miniseed"
# file_name = "UT.STN11.A2_C150.miniseed"
# file_name = "UT.STN12.A2_C50.miniseed"
# file_name = "UT.STN12.A2_C150.miniseed"
#file_name= "april.mseed"
#file_path = "/Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/sesameapril/"
file_path = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/"
#file_name= "/Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/SESAMEAPRIL.mseed"

# Minimum frequency after resampling
resample_fmin = 0.1  # Default value
# Maximum frequency after resampling
# resample_fmax = 50  # Default value
# resample_fmax = sr/2

# Window length in seconds. In general low frequency peaks require longer window lengths.
# See the SESAME guidelines for specific window length recommendations.
# windowlength = 60 # Default value
    
# windowlength = 250
windowlength = 10 / resample_fmin

# Boolean to control whether Butterworth filter is applied. 
# Geopsy does not apply a bandpass filter.

filter_bool = False        

# Low-cut frequency for bandpass filter.

# filter_flow = 0.1 
filter_flow = resample_fmin               

# High-cut frequency for bandpass filter.
filter_fhigh = 30                   
# Filter order.
filter_order = 5

# Width of cosine taper {0. - 1.}. Geopsy default of 0.05 is equal to 0.1 -> 0.1 is recommended
width = 0.1

# %%
# Konno and Ohmachi smoothing constant. 40 is recommended.
bandwidth = 40

# Number of frequencies after resampling
resample_fnum = 200
# Type of resampling {'log', 'linear'}
resample_type = 'log'

# Upper and lower frequency limits to restrict peak selection. To use the entire range use `None`.
peak_f_lower = None
peak_f_upper = None

# %%
##Azimutal settings##

# Rotation of horizontal components
# azimuthal_inverval defines the spacing in degrees between considerd azimuths -> 15 is recommended.
azimuthal_interval = 15
azimuth = np.arange(0, 180, azimuthal_interval)

# Boolean to control whether frequency-domain rejection-rejection algorithm is applied.
# Geopsy does not offer this functionality.
rejection_bool = True
# Number of standard deviations to consider during rejection. Smaller values will reject more windows -> 2 is recommended.
n = 2
# Maximum number of iterations to perform for rejection -> 50 is recommended.
max_iterations = 50

# Distribution of f0 {"lognormal", "normal"}. Geopsy default "normal" -> "lognormal" is recommended.
distribution_f0 = "lognormal"
# Distribution of mean curve {"lognormal", "normal"}. Geopsy default "lognormal" -> "lognormal" is recommended.
distribution_mc = "lognormal"

# %%
##Plot Settings##
#Manually set the ylimits of the HVSR figures. Default is None so limits will be set automatically.
ymin, ymax = 0, 15

# %%
from pathlib import Path
file_path = "/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/"
file_name = file_path+"Z9.D06.2013-05-04.2013-05-05.mseed"

pathlist = Path(file_path).glob('**/*.mseed')
for path in pathlist:
    print(path)

    st = read(path)
    sr  = st[0].stats.sampling_rate
    print(sr)

st_sta = read(file_name)

#fig_out = './results/'+azimuth+'/figures/'
#hvsrpy_out = './results/'+azimuth+'/hvsrpy/'
#geopsy_out = './results/'+azimuth+'/geopsy/'

#if not os.path.exists(fig_out):
 #      os.makedirs(fig_out)
#if not os.path.exists(hvsrpy_out):
 #      os.makedirs(hvsrpy_out)
#if not os.path.exists(geopsy_out):
 #      os.makedirs(geopsy_out)

# %%
st
# sensor
# sensor = hvsrpy.Sensor3c.from_mseed(file_name)
# file_name
hvsrpy.Sensor3c.from_mseed("/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/Z9.D02.2012-05-17.2012-05-18.mseed")

# %%
##Calculations##

# Generate output directories
method = 'multiple-azimuths'
fig_out = './July/'+method+'/figures/'
hvsrpy_out = './July/'+method+'/hvsrpy/'
geopsy_out = './July/'+method+'/geopsy/'

if not os.path.exists(fig_out):
    os.makedirs(fig_out)
if not os.path.exists(hvsrpy_out):
    os.makedirs(hvsrpy_out)
if not os.path.exists(geopsy_out):
    os.makedirs(geopsy_out)

# Loop over all mseed files in file_path

pathlist = Path(file_path).glob('**/*.mseed')
for path in pathlist:

    file_name = path #file_path+"Z9.D03.2013-05-04.2013-05-05.mseed"
    station_str = str(path).split('/')[-1].split('.mseed')[0]

    print('Working on: '+station_str)

    st = read (file_name)
    sr  = st[0].stats.sampling_rate
    start = time.time()
    sensor = hvsrpy.Sensor3c.from_mseed(file_name)
    resample_fmax = sr/2
    bp_filter = {"flag":filter_bool, "flow":filter_flow, "fhigh":filter_fhigh, "order":filter_order}
    resampling = {"minf":resample_fmin, "maxf":resample_fmax, "nf":resample_fnum, "res_type":resample_type}
    hv = sensor.hv(windowlength, bp_filter, width, bandwidth, resampling, "multiple-azimuths", f_low=peak_f_lower, f_high=peak_f_upper, azimuth=azimuth)

    # Check for bad data with a single value at all points
    if np.sum(np.diff(st[0].data)) == 0:
        print("Problem with "+st[0].stats.channel+" ... skipping")
        continue
    elif np.sum(np.diff(st[1].data)) == 0:
        print("Problem with "+st[1].stats.channel+" ... skipping")
        continue
    elif np.sum(np.diff(st[2].data)) == 0:
        print("Problem with "+st[2].stats.channel+" ... skipping")
        continue

    if rejection_bool:
        hv.reject_windows(n=n, max_iterations=max_iterations, distribution_f0=distribution_f0, distribution_mc=distribution_mc)

    azimuths = [*hv.azimuths, 180.]
    mesh_frq, mesh_azi = np.meshgrid(hv.frq, azimuths)
    mesh_amp = hv.mean_curves(distribution=distribution_mc)
    mesh_amp = np.vstack((mesh_amp, mesh_amp[0]))

    end = time.time()
    print(f"Elapsed Time: {str(end-start)[0:4]} seconds")

    # Layout
    fig = plt.figure(figsize=(6,5), dpi=150)
    gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.3, hspace=0.1, width_ratios=(1.2,0.8))
    ax0 = fig.add_subplot(gs[0:2, 0:1], projection='3d')
    ax1 = fig.add_subplot(gs[0:1, 1:2])
    ax2 = fig.add_subplot(gs[1:2, 1:2])
    fig.subplots_adjust(bottom=0.21) 

    # Settings
    individual_width = 0.3
    median_width = 1.3

    ## 3D Median Curve
    ax = ax0
    ax.plot_surface(np.log10(mesh_frq), mesh_azi, mesh_amp, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
    for coord in list("xyz"):
        getattr(ax, f"w_{coord}axis").set_pane_color((1, 1,1))    
    ax.set_xticks(np.log10(np.array([0.01, 0.1, 1, 10, 100])))
    ax.set_xticklabels(["$10^{"+str(x)+"}$" for x in range(-2, 3)])
    ax.set_xlim(np.log10((0.1, 30)))
    ax.view_init(elev=30, azim=245)
    ax.dist=12
    ax.set_yticks(np.arange(0,180+45, 45))
    ax.set_ylim(0,180)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_zlabel("HVSR Amplitude")
    pfrqs, pamps = hv.mean_curves_peak(distribution=distribution_mc)
    pfrqs = np.array([*pfrqs, pfrqs[0]])
    pamps = np.array([*pamps, pamps[0]])
    ax.scatter(np.log10(pfrqs), azimuths, pamps*1.01, marker="s", c="w", edgecolors="k", s=9)

    ## 2D Median Curve
    ax = ax1
    contour = ax.contourf(mesh_frq, mesh_azi, mesh_amp, cmap=cm.plasma, levels=10)
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_ylabel("Azimuth (deg)")
    ax.set_yticks(np.arange(0,180+30, 30))
    ax.set_ylim(0,180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    fig.colorbar(contour, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")

    ax.plot(pfrqs, azimuths, marker="s", color="w", linestyle="", markersize=3, markeredgecolor="k",
            label=r"$f_{0,mc,\alpha}$")

    ## 2D Median Curve
    ax = ax2

    # Accepted Windows
    label="Accepted"
    for amps in hv.amp:
        for amp in amps:
            ax.plot(hv.frq, amp, color="#888888", linewidth=individual_width, zorder=2, label=label)
            label=None

    # Mean Curve
    label = r"$LM_{curve,AZ}$" if distribution_mc=="lognormal" else r"$Mean_{curve,AZ}$"
    ax.plot(hv.frq, hv.mean_curve(distribution_mc), color='k', label=label, linewidth=median_width, zorder=4)

    # Mean +/- Curve
    label = r"$LM_{curve,AZ}$"+" ± 1 STD" if distribution_mc=="lognormal" else r"$Mean_{curve,AZ}$"+" ± 1 STD"
    ax.plot(hv.frq, hv.nstd_curve(-1, distribution=distribution_mc), color="k", linestyle="--",
            linewidth=median_width, zorder=4, label=label)
    ax.plot(hv.frq, hv.nstd_curve(+1, distribution=distribution_mc), color="k", linestyle="--",
            linewidth=median_width, zorder=4)

    # Window Peaks
    label = r"$f_{0,i,\alpha}$"
    for frq, amp in zip(hv.peak_frq, hv.peak_amp):
        ax.plot(frq, amp, linestyle="", zorder=3, marker='o', markersize=2.5, markerfacecolor="#ffffff",
                markeredgewidth=0.5, markeredgecolor='k', label=label)
        label=None

    # Peak Mean Curve
    ax.plot(hv.mc_peak_frq(distribution_mc), hv.mc_peak_amp(distribution_mc), linestyle="", zorder=5,
            marker='D', markersize=4, markerfacecolor='#66ff33', markeredgewidth=1, markeredgecolor='k', 
            label = r"$f_{0,mc,AZ}$")

    # f0,az
    if ymin is not None and ymax is not None:
            ax.set_ylim((ymin, ymax))
    label = r"$LM_{f0,AZ}$"+" ± 1 STD" if distribution_f0=="lognormal" else "Mean "+r"$f_{0,AZ}$"+" ± 1 STD"
    _ymin, _ymax = ax.get_ylim()
    ax.plot([hv.mean_f0_frq(distribution_f0)]*2, [ymin, ymax], linestyle="-.", color="#000000", zorder=6)
    ax.fill([hv.nstd_f0_frq(-1, distribution_f0)]*2 + [hv.nstd_f0_frq(+1, distribution_f0)]*2, [_ymin, _ymax, _ymax, _ymin], 
            color = "#ff8080", label=label, zorder=1)
    ax.set_ylim((_ymin, _ymax))

    # Limits and labels
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    # Lettering
    xs, ys = [0.45, 0.85, 0.85], [0.81, 0.81, 0.47]  
    for x, y, letter in zip(xs, ys, list("abc")):
        fig.text(x, y, f"({letter})", fontsize=12)
        
    # Legend
    handles, labels = [], []
    for ax in [ax2, ax1, ax0]:
            _handles, _labels = ax.get_legend_handles_labels()
            handles += _handles
            labels += _labels
    new_handles, new_labels = [], []
    for index in [0, 5, 1, 2, 3, 4, 6]:
        new_handles.append(handles[index])
        new_labels.append(labels[index])
    fig.legend(new_handles, new_labels, loc="lower center", bbox_to_anchor=(0.47, 0), ncol=4,
            columnspacing=0.5, handletextpad=0.4)
    

                # Print stats
    print("\nStatistics after rejection considering azimuth:")
    hv.print_stats(distribution_f0)

    plt.show()


    ###################### Save figure ###################### 

    figure_name_out = fig_out+"/"+station_str+"_hvsr_figure_az.png"

    fig.savefig(figure_name_out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved successfully!")


    ###################### Save results text file ###################### 

    file_name_out_hvsrpy = hvsrpy_out+"/"+station_str+"_output_hvsrpy_az.hv"

    hv.to_file(file_name_out_hvsrpy, distribution_f0, distribution_mc, data_format="hvsrpy")
    print("Results saved successfully!")


    ###################### Save Geopsy text file ######################
    
   # file_name_out_geopsy = geopsy_out+"/"+station_str+"_output_geopsy_az.hv"

   # hv.to_file(file_name_out_geopsy, distribution_f0, distribution_mc, data_format="geopsy")
    #print("Results saved successfully!")


        



# %%
st

# %%
#sensor = hvsrpy.Sensor3c.from_mseed(file_name)
# st[2].plot()
# st.print_gaps()
# st.merge(method=0, fill_value=None).plot()
# st = read (file_name)
# st.merge(method=0, fill_value=None)
# st
# hvsrpy.Sensor3c.from_mseed(file_name)
file_name


# %%
pathlist = pathlist = Path(file_path).glob('**/*.mseed')

# %%
##save fig to file##



file_name_out = "example_output_hvsrpy_az.hv"

hv.to_file(file_name_out, distribution_f0, distribution_mc)
print("Results saved successfully!")

# %%


# %%
file = open ('/Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/results/multiple-azimuths/hvsrpy/.hv', 'r')
Lines = file.readlines()
    for line in Lines:
    
    
#print (Lines[13].split(',')[1])


# %%
pathlist = sorted(Path(hvsrpy_out).glob('**/*.hv'))




f0_hvsr= np.array ([])
peak_freq_hvsr =np.array([])
peak_amplitude_hvsr = np.array([])
lats=np.array([])
lons=np.array([])
for path in pathlist:
    path_in_str = str(path)  
   # print (path)
    file = open (path, 'r')#('/Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/results/multiple-azimuths/hvsrpy/.hv#file_path+"Z9.D03.2013-05-04.2013-05-05.mseed"
    #f0 = str(path).split('/')[-1].split('.hv')[0]
    Lines =file.readlines()

    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    #loc = path.split('/')[-1].split('.')[2]
    starttime = path_in_str.split('/')[-1].split('.')[2]
    endtime = path_in_str.split('/')[-1].split('.')[3].split('_output_hvsrpy_az')[0]

     # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)
    #print(lat)
    
    f0= float(Lines[13].split(',')[1])
    f0_hvsr =np.append (f0_hvsr, f0)
    #print(f0)
    #print(len(f0_hvsr))
    peak_freq = float(Lines[20].split (',')[1])
    peak_freq_hvsr = np.append (peak_freq_hvsr, peak_freq )

    peak_amplitude = float (Lines [21].split (',') [1])
    peak_amplitude_hvsr = np.append (peak_amplitude_hvsr, peak_amplitude)

    #print(peak_amplitude_hvsr)

    #print (peak_freq)
    #print (peak_freq_hvsr)

    #stan_dev = float (Lines[15].split(',')[1])
    #stan_dev_calculation = np.append (stan_dev_calculation, stan_dev)
    #print (stan_dev_calculation)
    
    LM_dev = float (Lines [17].split ('.')[1])
    LM_deviation = np.append (LM_deviation, LM_dev)




# %%
starttime

# %%
path2grdfile = '//Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/etopo1.grd'
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
 #PLOT PEAK FREQUENCY

# Map projection
data_projection = ccrs.PlateCarree()

# Mask the bad stations
#mask_bad_hvsr = (max_hvsr < 2) | (max_hvsr > 100) | (max_freq>18)
mask_bad_hvsr = (f0_hvsr >990) 

# Generate the map
ax = plt.axes(projection=data_projection)
ax.set_extent([minlongitude-0.5, maxlongitude+0.5, minlatitude-0.5, maxlatitude+0.5], crs=data_projection)
img_extent = (-180, 180, -90, 90)
pos = ax.imshow(etopo, origin='upper', extent=img_extent, transform=data_projection,
            cmap='gray', alpha=1, zorder=-1)
cb = plt.colorbar(pos, ax=ax, shrink=0.8)
cb.ax.set_title('Elev. (m)')
cmax = etopo.max()
cmin = etopo.min()
pos.set_clim(-200,1000)
ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'], zorder=0)
# ax.add_feature(cfeature.RIVERS)
ax.coastlines(zorder=0)

# Plot bad stations first
ax.scatter(lons[mask_bad_hvsr],lats[mask_bad_hvsr],marker='o',s=50,c='white',edgecolor='black', transform=data_projection)
# Now plot good stations
sc = ax.scatter(
    lons[~mask_bad_hvsr],lats[~mask_bad_hvsr], marker='o', s=50, 
    # c=max_hvsr[~mask_bad_hvsr], 
    c=np.log10(f0_hvsr[~mask_bad_hvsr]),
    cmap='Spectral',edgecolor='black', transform=data_projection)
cb = plt.colorbar(sc)
cb.set_label('Peak Frequency (Hz)')
# ax.set_xlim(minlongitude,maxlongitude)
# ax.set_ylim(minlatitude,maxlatitude)

gl = ax.gridlines(crs= data_projection, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
gl.xlabels_top = False
gl.ylabels_right = False

plt.savefig('hvsr_peak_frequency.pdf')

# %%
#plot HVSR median curve peak amplitude and median frequency


pathlist = sorted(Path(hvsrpy_out).glob('**/*.hv'))
df= pathlist
plt.figure ()
peak_amplitudes = np.array ([])
f_0 = np.array ([])
ii = 0
for path in pathlist:
   peak_amplitudes =  np.append (peak_amplitudes, peak_amplitude)
   f_0 =np.append (f_0, f0)
   plt.plot (['f_0'], ['peak_amplitudes']) 
   plt.xlim(0,20)
   plt.ylim(0,10)
   plt.xlabel('LogNormal Frequency Hz')
   plt.ylabel('Median Peak Amplitude')
   #plt.xscale('log')
   #plt.plot (f0,peak_amplitude,'ro')


   











# %%

           

 #PLOT PEAK FREQUENCY

# Map projection
data_projection = ccrs.PlateCarree()

# Mask the bad stations
#mask_bad_hvsr = (max_hvsr < 2) | (max_hvsr > 100) | (max_freq>18)
mask_bad_hvsr = (f0_hvsr >990) 

# Generate the map
ax = plt.axes(projection=data_projection)
ax.set_extent([minlongitude-0.5, maxlongitude+0.5, minlatitude-0.5, maxlatitude+0.5], crs=data_projection)
img_extent = (-180, 180, -90, 90)
pos = ax.imshow(etopo, origin='upper', extent=img_extent, transform=data_projection,
            cmap='gray', alpha=1, zorder=-1)
cb = plt.colorbar(pos, ax=ax, shrink=0.8)
cb.ax.set_title('Elev. (m)')
cmax = etopo.max()
cmin = etopo.min()
pos.set_clim(-200,1000)
ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'], zorder=0)
# ax.add_feature(cfeature.RIVERS)
ax.coastlines(zorder=0)

# Plot bad stations first
ax.scatter(lons[mask_bad_hvsr],lats[mask_bad_hvsr],marker='o',s=50,c='white',edgecolor='black', transform=data_projection)
# Now plot good stations
sc = ax.scatter(
    lons[~mask_bad_hvsr],lats[~mask_bad_hvsr], marker='o', s=50, 
    # c=max_hvsr[~mask_bad_hvsr], 
    c=np.log10(peak_freq_hvsr[~mask_bad_hvsr]),
    cmap='Spectral',edgecolor='black', transform=data_projection)
cb = plt.colorbar(sc)
cb.set_label('Peak Frequency (Hz)')
# ax.set_xlim(minlongitude,maxlongitude)
# ax.set_ylim(minlatitude,maxlatitude)

gl = ax.gridlines(crs= data_projection, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
gl.xlabels_top = False
gl.ylabels_right = False

plt.savefig('hvsr_peak_frequency.pdf')

# %%
plt.figure()
mask_bad_hvsr = (np.abs(f0_hvsr-peak_freq_hvsr) > 2)
plt.plot(f0_hvsr,peak_freq_hvsr,'o')
plt.plot(f0_hvsr[mask_bad_hvsr],peak_freq_hvsr[mask_bad_hvsr],'or')
plt.plot([0,25],[0,25],'--k')


# %%


# %%
pathlist = sorted(Path(hvsrpy_out).glob('**/*.hv'))

f0_hvsr= np.array ([])

for path in pathlist:
    print (path)
    file = open (path, 'r')#('/Users/birotimi/Desktop/PHD/HVSR-project/HVSR-master/hvsrpy/results/multiple-azimuths/hvsrpy/.hv#file_path+"Z9.D03.2013-05-04.2013-05-05.mseed"
    #f0 = str(path).split('/')[-1].split('.hv')[0]
    Lines =file.readlines()

    f0= float(Lines[13].split(',')[1])
    f0_hvsr =np.append (f0_hvsr, f0)
    #print(f0)
    print(len(f0_hvsr))
    

# %%
print (f0)

# %%
print (peak_freq)

# %%
#rec = 1/f0_hvsr
df = pathlist # = sorted(Path(hvsrpy_out).glob('**/*.hv'))
plt.figure ()

plt.plot (df ['f0_hvsr'], df ['peak_freq_hvsr']) #label = table)
          
plt.ylim (0,20)
plt.xlim (0,50)
plt.xlabel ('f0_hvsr Hz')
plt.ylabel ('peak_freq')
plt.yscale ('log')

plt.show


