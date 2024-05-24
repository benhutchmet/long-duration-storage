"""
David storage model code which Lainey will be using to try
and replicate plots from the Royal Society report on Large-Scale electricity storage.
"""


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta


##############################################################################
# Setup
##############################################################################

# Capacities --- these are for the 741 TWh/yr RE scenario for 570 TWh/yr demand
# - Generation capacity such that mean annual genration (sol+wind in relevant ratios) is 741 TWh
#   (from memory, I think I lump wind onshore and offshore together!)
# - Demand rescale by multiplication (so mean *and* variance increase to reach average 570 TWh/y) 
INSTALLED_WP_CAP = 184.29 # Please check!
INSTALLED_SOLAR_CAP = 158.26 # Please check!
DEMAND_SCALING = 1.63 # Please check!

# Storage parameters - please check!
smax = 123.*1000. # total store (in GWh)
effin = 0.75 # 25% loss on input (is this correct?)
effout = 0.55 # 45% loss on output (is this correct?)
effstore = 1.0 # 0% loss on the store insitu (i.e., is <1.0 then hydrogen leaks from store)
convmax = 98.*24 # Maximum conversion rate (electricity <=> hydrogen)


##############################################################################
# Load the demand data (one of the UREAD datasets)
# - I found I had to do a bit of hacking to get the data to work, but maybe this is me using pandas badly!
##############################################################################
DEMAND_EU = pd.read_csv('ERA5_full_demand_1979_2018.csv',parse_dates=['Unnamed: 0'])
DEMAND_UK = DEMAND_EU['United_Kingdom_full_demand_no_pop_weights_1979_2018.dat']
Dates = DEMAND_EU['Unnamed: 0']

# Strip off the surplus characters off hannah's date field
for ii in np.arange(0,len(Dates),1): Dates[ii]=(((Dates[ii].replace("(","")).replace(")","")).replace(",",""))

df_demand = {'Dates':Dates,'UK_Demand':DEMAND_UK} 
UK_dataframe_demand = pd.DataFrame(df_demand)
UK_dataframe_demand.set_index('Dates')
UK_dataframe_demand['date'] = pd.to_datetime(UK_dataframe_demand['Dates'])

# Convert to daily demand... (note *24 because demand data is GW reported daily)
daily_df_demand = UK_dataframe_demand.resample('D', on='date')['UK_Demand'].sum()*24.* DEMAND_SCALING# in GWh


##############################################################################
# Load the wind power data (one of the UREAD datasets)
##############################################################################
WIND_DATA = pd.read_csv('ERA5_wind_power_capacity_factor_all_countries_1979_2018_inclusive_3hourly.csv',parse_dates=['Unnamed: 0'])
WIND_UK = WIND_DATA['ERA5_native_grid_United_Kingdom_capacity_factor_at_each_site_1979_2018_v16.dat']
time_data_3hrly = WIND_DATA['Unnamed: 0']
df = {'timesteps':time_data_3hrly,'UK_CF':WIND_UK}
UK_df_wind = pd.DataFrame(df)
UK_df_wind['date'] = pd.to_datetime(UK_df_wind['timesteps'])
UK_df_wind['year'] = pd.DatetimeIndex(UK_df_wind['date']).year

# Convert to daily wind power... (note *3 because wind data is GW reported every 3h)
daily_df_wind = UK_df_wind.resample('D', on='date')['UK_CF'].sum()*3.*INSTALLED_WP_CAP# in GWh


##############################################################################
# Load the solar data (one of the UREAD datasets)
##############################################################################
SOLAR_DATA = pd.read_csv('ERA5_solar_power_capacity_factor_all_countries_1979_2018_inclusive_3hourly.csv',parse_dates=['Unnamed: 0'])
SOLAR_UK = SOLAR_DATA['United_Kingdom_cf_no_pop_weights_edit_eff_Phils_lims_1979_2019.dat']
time_data_3hrly_solar = SOLAR_DATA['Unnamed: 0']
df = {'timesteps':time_data_3hrly_solar,'UK_CF_solar':SOLAR_UK}
UK_df_solar = pd.DataFrame(df)
UK_df_solar['date'] = pd.to_datetime(UK_df_solar['timesteps'])
UK_df_solar['year'] = pd.DatetimeIndex(UK_df_solar['date']).year

# Convert to daily wind power... (note *3 because solar data is GW reported every 3h)
daily_df_solar = UK_df_solar.resample('D', on='date')['UK_CF_solar'].sum()*3.*INSTALLED_SOLAR_CAP# in GWh



##############################################################################
# Put it into some simpler variables, units are GWh/day
##############################################################################
dat = daily_df_demand.index
dem = daily_df_demand.values
wnd = daily_df_wind.values
sol = daily_df_solar.values
sur = wnd+sol-dem # instanstaneous surplus

##############################################################################
# Calculate time-integral of surplus (without any losses)
##############################################################################
sur_cum = np.zeros(len(sur)) # accumulated surplus (from start of timeseries)
for ii in np.arange(0,len(sur),1): sur_cum[ii] = sum(sur[0:ii+1])

##############################################################################
# Calculate realised storage (starting from a full store) including losses
##############################################################################
store = np.zeros(len(sur))
store[0] = smax
for ii in np.arange(0,len(sur)-1,1): 
	if (sur[ii] > 0.): # Surplus, add to store
		store[ii+1] = (effstore*store[ii]) + np.min([sur[ii]*effin,convmax])
	else: # Shortfall, take from store
		store[ii+1] = (effstore*store[ii]) + sur[ii]/effout
	if (store[ii+1] > smax): # Store overflow, cap at maximum level
		store[ii+1] = smax

		
##############################################################################
# Plot storage level over time
##############################################################################
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2,1,1)
xvals = dat
yvals = store/1000. # convert to TWh
minx = min(xvals)
maxx = max(xvals)
stepx = 10
miny = 0.
maxy = max(yvals)
stepy = 10.
ax1.axis([minx,maxx,miny,maxy])
ax1.tick_params(direction='out', which='both', labelsize='x-large')
ax1.set_xlabel('Date', fontsize='x-large')
ax1.set_ylabel('Storage level, TWh', fontsize='x-large')
ax1.set_xticks(np.arange(minx,maxx,stepx))
ax1.set_yticks(np.arange(miny,maxy,stepy))
ax1.plot(xvals,yvals,label='label',color="red",linewidth=1.)
#ax1.legend(loc='upper right', fontsize='medium')
plt.savefig('testA.png')
plt.close(fig)






##############################################################################
# Here, I was starting to calculate a different way of looking at the data
# Essentially, trying to work out what the maximum storage needed over 
# a time window of arbitrary length
# I'm not sure if this is useful (or even 'correct'!), but potentially worth thinking about?
# Key bit I haven't understood is why max(-sur_rt_runmean_min) doesn't match 
# the biggest drawdown on the store in the previous plot (between 2009, 12, 10 and 2011, 3, 31).
##############################################################################
hwindows = np.arange(0,1440,5) # half-windows for running window

# Calculate flow into and out of the store each day
# Allow for efficiency of input to / output from storage
# Limit the electrolyser to a maximum value on input only, note:
# - apply electrolyser limit to input only
# - apply electrolyser limit to the actual inflow (after efficiency loss)
# I don't know if these assumptions are correct, please check in the RS report!
sur_pos = sur > 0. # surplus days
sur_neg = sur < 0. # deficit days
sur_rt = np.zeros(len(sur))
sur_rt[sur_pos] = sur[sur_pos] * effin # putting into store
sur_rt[sur_rt > convmax] = convmax # max of electrolyser
sur_rt[sur_neg] = sur[sur_neg] / effout # retrieval from store

# Calculate window-storage curve
# sur_runmean == running mean (over 2*hwindow+1) of surplus if unconstrainted by efficiency losses
# sur_rt_runmean == constrainted by efficiencies and electrolyser (but no losses from store itself)
# min == storage size needed to meet deficits over the window
# max == storage size needed to hold all surpluses over the window
sur_runmean = np.zeros([len(hwindows),len(sur)])
sur_runmean_max = np.zeros(len(hwindows))
sur_runmean_min = np.zeros(len(hwindows))
sur_rt_runmean = np.zeros([len(hwindows),len(sur)])
sur_rt_runmean_max = np.zeros(len(hwindows))
sur_rt_runmean_min = np.zeros(len(hwindows))
for ii in np.arange(0,len(hwindows)):
	hwin = hwindows[ii]
	for jj in np.arange(hwin,len(sur)-hwin):
		sur_runmean[ii,jj]=np.sum(sur[jj-hwin:jj+hwin])
		sur_rt_runmean[ii,jj]=np.sum(sur_rt[jj-hwin:jj+hwin])
	sur_runmean_max[ii] = np.max(sur_runmean[ii,:])
	sur_rt_runmean_max[ii] = np.max(sur_rt_runmean[ii,:])
	sur_runmean_min[ii] = np.min(sur_runmean[ii,:])
	sur_rt_runmean_min[ii] = np.min(sur_rt_runmean[ii,:])
		

# plot window-storage
fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(2,1,1)
xvals = (hwindows*2+1)/365.
minx = 0.
maxx = 5.0
stepx = .5
miny = 0.
maxy = 400.
stepy = 50.
ax1.axis([minx,maxx,miny,maxy])
ax1.tick_params(direction='out', which='both', labelsize='x-large')
ax1.set_xlabel('Years (approx)', fontsize='x-large')
ax1.set_ylabel('Storage required, TWh', fontsize='x-large')
ax1.set_xticks(np.arange(minx,maxx,stepx))
ax1.set_yticks(np.arange(miny,maxy,stepy))
ax1.plot(xvals,-sur_runmean_min/1000.,'r-',label='Meets deficit',linewidth=1.)
ax1.plot(xvals,-sur_rt_runmean_min/1000.,'r.',label='Constrained, meets deficit',linewidth=1.)
ax1.plot(xvals,sur_runmean_max/1000.,'b-',label='Meets surplus',linewidth=1.)
ax1.plot(xvals,sur_rt_runmean_max/1000.,'b.',label='Constrained, meets surplus',linewidth=1.)
ax1.legend(loc='upper right', fontsize='medium')

plt.savefig('testC_usewithcare.png')
plt.close(fig)




import pdb; pdb.set_trace()
















