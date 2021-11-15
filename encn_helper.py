from tethysts import Tethys
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import salem
import metpy
import metpy.calc
import glob
from metpy.units import units


def get_dataset_id (parameter,product_code,owner,tathy_instance):
    datasets = tathy_instance.datasets
    dataset=[dataset for dataset in datasets if (dataset['parameter'] == parameter) and
              (dataset['product_code'] == product_code) and
              (dataset['owner'] == owner)][0]
    return dataset["dataset_id"]

def get_station_id (dataset_id,station_names,tathy_instance):
    all_stations = tathy_instance.get_stations(dataset_id)
    if isinstance(station_names,str):
        station =[s for s in all_stations if (s['name'] == station_names)][0]
        station_id = station["station_id"]
    elif isinstance(station_names,list):
        station =[s for s in all_stations if (s['name'] in station_names)]
        station_id =[s["station_id"] for s in station]
    return station_id

def plot_wind_rose(ws,wd,nsector=16, normed=True, opening=0.8, edgecolor='white',cmap='viridis',**kw):
    velocity_ds=xr.merge([ws,wd])
    if (type(ws) == xr.core.dataarray.Dataset) and (type(wd) == xr.core.dataarray.Dataset):
        ws_name = "wind_speed"
        wd_name = "wind_direction"
    else:
        ws_name = ws.name
        wd_name = wd.name
    ax = WindroseAxes.from_ax()
    ccmap = plt.get_cmap(cmap)
    ax.bar(velocity_ds[wd_name],velocity_ds[ws_name], normed=True, opening=0.8, edgecolor='white', bins=np.arange(0, 8, 1),cmap=ccmap)
    ax.set_xticklabels((90, 45, 0, 315, 270, 225, 180, 135))
    ax.set_xticklabels(['E', 'NE','N', 'NW', 'W', 'SW', 'S', 'SE'])
    ax.set_legend()
    fig=plt.gcf()
    plt.close()
    return fig

def wind_hourly_composite(ws,wd,save_files=False):
    velocity_ds=xr.merge([ws,wd])
    if (type(ws) == xr.core.dataarray.Dataset) and (type(wd) == xr.core.dataarray.Dataset):
        ws_name = "wind_speed"
        wd_name = "wind_direction"
    else:
        ws_name = ws.name
        wd_name = wd.name
    u_component,v_component=metpy.calc.wind_components(velocity_ds[ws_name].compute()*units.meter_per_second,velocity_ds[wd_name].compute()*units.degree)
    u_component_hourly=u_component.groupby("time.hour").mean()
    v_component_hourly=v_component.groupby("time.hour").mean()
    
    wspeed_hourly = velocity_ds[ws_name].groupby("time.hour").mean()
    wspeed_hourly.name = ws_name
    wdir_hourly = metpy.calc.wind_direction(u_component_hourly,v_component_hourly)
    wdir_hourly.name = wd_name
    fig, axes=plt.subplots(2,dpi=150)
    axes[0].plot(wspeed_hourly)
    axes[0].set_ylabel("Wind Speed (m/s)")
    axes[1].plot(wdir_hourly)
    axes[1].set_ylabel(r"Wind Direction ($^\circ$)")
    axes[1].set_xlabel(r"Time (hour)")
    plt.close()
    if save_files == True:
        wspeed_hourly.to_dataframe().to_csv("houlry_composite_windspeed.csv")
        wdir_hourly.to_dataframe().to_csv("houlry_composite_winddirection.csv")
    return fig

# def data_cleaner(data_array,time_zone='Pacific/Auckland'):
#     data_df=data_array.to_dataframe().drop(['geometry', 'height',],axis=1)
#     data_df=data_df.reset_index()
#     data_df["time"] = data_df["time"].dt.tz_localize('utc').dt.tz_convert('Pacific/Auckland')
#     return data_df

def ds_utc_local(ds,local_zone='Pacific/Auckland'):
    time_local=pd.to_datetime(ds.time.values).tz_localize('UTC').tz_convert(local_zone).tz_localize(None)
    #time_local = pd.to_datetime(ds.time.values).tz_localize(local_zone,ambiguous=True,nonexistent="shift_backward").tz_convert(None).values
    ds=ds.assign_coords(time=time_local,keep_attrs=True)
    return ds

def plot_hourly_composite(data_array,time_zone='Pacific/Auckland',average="day", dpi=150, save_files=False, **kw):
    df = data_array.to_dataframe().reset_index()
    df = (
        df.groupby(
            [
                df.time.dt.strftime("%Y"),
                df.time.dt.strftime("%Y-%m"),
                df.time.dt.strftime("%Y-%m-%d"),
                df.time.dt.strftime("%H:00"),
            ]
        )[df.columns[-1]]
        .mean()
        .rename_axis(index=["year", "month", "day", "hour"])
    )
    df_stack = df.groupby([average, "hour"]).mean(average)
    fig, ax = plt.subplots(dpi=dpi)
    sns.heatmap(df_stack.unstack(level=average), ax=ax, **kw)
    plt.xticks(rotation=90)
    if average == "day" or average == "month":
        for label in ax.get_xaxis().get_ticklabels()[::2]:
            label.set_visible(False)
    for label in ax.get_yaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    fig.autofmt_xdate()
    plt.close()
    if save_files==True:
        df_stack.unstack(level=average).to_csv("hourly_composite_"+average+".csv")
    return fig

def get_horizontal_idx_from_latlon(ds,lat,lon):
    x = ((ds.lat-lat)**2 +(ds.lon-lon)**2).values
    return np.argwhere(x == np.min(x))[0]
def get_horizontal_idx_from_snwe(ds,sn,we):
    sn_idx=np.argmin(np.abs(ds.south_north.values -sn))
    we_idx=np.argmin(np.abs(ds.west_east.values -we))
    return sn_idx,we_idx

def add_variables(ds,calculate_wdir=False):
    with xr.set_options(keep_attrs=True):
        ds["theta"] =      ds.T+300.0
        ds["pressure"]=    ds.P+ds.PB
    ds["temperature"]= metpy.calc.temperature_from_potential_temperature(ds.pressure*units.Pa,ds.theta*units.kelvin)
    ds["temperature"].values = ds["temperature"].metpy.magnitude
    ds["wind_speed"] = metpy.calc.wind_speed(ds.U*units.meter_per_second,ds.V*units.meter_per_second)
    ds["wind_speed"].values = ds["wind_speed"].metpy.magnitude
    ds["wind_speed_10m"] = metpy.calc.wind_speed(ds.U10*units.meter_per_second,ds.V10*units.meter_per_second)
    ds["wind_speed_10m"].values = ds["wind_speed_10m"].metpy.magnitude
    
    
    if calculate_wdir == True:
        with xr.set_options(keep_attrs=True):
            ds["wind_dir_10m"] = (270-xr.ufuncs.arctan2(ds.V10,ds.U10)* 180/ np.pi)%360
#             ds["wind_dir_10m"] = metpy.calc.wind_direction(ds.U10.compute()*units.meter_per_second,ds.V10.compute()*units.meter_per_second)
#             ds["wind_dir_10m"].values = ds["wind_dir_10m"].metpy.magnitude
            ds["wind_dir"] = (270-xr.ufuncs.arctan2(ds.V,ds.U)* 180/np.pi)%360
#             ds["wind_dir"] = metpy.calc.wind_direction(ds.U.compute()*units.meter_per_second,ds.V.compute()*units.meter_per_second)
#             ds["wind_dir"].values = ds["wind_dir"].metpy.magnitude
            ds["wind_dir"].attrs["long_name"] = "WD"
            ds["wind_dir_10m"].attrs["long_name"] = "WD10m"
#     ds["temporature"].attrs["pyproj_srs"] = ds.pyproj_srs
#     ds["pressure"].attrs["pyproj_srs"] = ds.pyproj_srs
#     ds["theta"].attrs["pyproj_srs"] = ds.pyproj_srs
    return ds


def get_point_dataset(ds,lat,lon,height_level=0):
    sn_idx,we_idx=get_horizontal_idx_from_latlon(ds,lat,lon)
    ds_point=ds.isel(south_north=sn_idx,west_east=we_idx,bottom_top=height_level)
#     if clean_height == True:
#         ds_point["bottom_top"] = ds_point.Z.mean(dim="time",keep_attrs=True)-ds_point.HGT.mean(dim="time",keep_attrs=True)
    return ds_point

def get_vertical_profile_dataset(ds,lat,lon,clean_height=True):
    sn_idx,we_idx=get_horizontal_idx_from_latlon(ds,lat,lon)
    ds_point=ds.isel(south_north=sn_idx,west_east=we_idx)
    if clean_height == True:
        ds_point["bottom_top"] = ds_point.Z.mean(dim="time",keep_attrs=True)-ds_point.HGT.mean(dim="time",keep_attrs=True)
    return ds_point

def add_proj_attrs_2d3dvar(ds):
    var_list = list(ds.var())
    for var_name in var_list:
        if len(ds[var_name].shape) >=3:
            ds[var_name].attrs["pyproj_srs"] = ds.pyproj_srs
    return ds

def clean_data(ds,calculate_wdir=False,convert_localtz=True):
    ds=ds.sel(time=~ds.indexes['time'].duplicated())
    if convert_localtz ==True:
        ds = ds_utc_local(ds)
    if "xtime" in list(ds.variables):
        ds = ds.drop("xtime")
    ds = add_variables(ds,calculate_wdir)
    ds = add_proj_attrs_2d3dvar(ds)
    return ds

def get_wrf_by_time(domain_number, start_date,end_date,folder="/srv/shared_space/"):
    file_list=sorted(glob.glob(folder+'**/wrfout_*',recursive=True))
    domain_file_list = [f for f in file_list if "d0"+str(int(domain_number)) in f]
    sub_file_list = [f for f in domain_file_list if compare_date_str(f.split("/")[-1].split("_")[-4],start_date) if compare_date_str(end_date,f.split("/")[-1].split("_")[-4])]
    return sub_file_list

def compare_date_str(date1,date2):
#     date1 = "-".join(date1.split("-")[:2])
#     date2 = "-".join(date2.split("-")[:2])
    if date1 >= date2:
        return True
    else:
        return False
    
def plot_vec_t(ds,level="surface",vmin=None,vmax=None,countries=False,vec_scale="auto",cord_interval=5,cord_lw=1.5,cmap='viridis',vec_gap_x=7,vec_gap_y=7):
    #plot the surface temperature map, with the vector on top
    #You can plot other height levels if you want by specify an integer for level (bottom_top index)
    if (len(ds.time.values.shape) != 0) and  (ds.time.values.shape[0] == 1):
        ds_plot = ds.isel(time=0)
    elif (len(ds.time.values.shape) != 0) and  (ds.time.values.shape[0] != 1):
        print ("You have multiple time index!!! Consider specify the time index you want to plot or use plot_multi_vec_t.")
        return
    else:
        ds_plot = ds
    #get scale factor for the quiver plot
    if vec_scale == "auto":
        vec_scale = vec_scale_vec_t(ds,level=level)
    #get the cross section slice
    if level != "surface":
        ds_plot = ds_plot.isel(bottom_top=level)
    

        
    if (vmin == None) and (vmax == None):
        if level == "surface":
            vmin = np.floor(ds_plot.T2.min().values -273.15)
            vmax = np.ceil(ds_plot.T2.max().values -273.15)
        else:
            vmin = np.floor(ds_plot.temperature.min().values -273.15)
            vmax = np.ceil(ds_plot.temperature.max().values -273.15)            
    mpl.rcdefaults()

    # get the data
    with xr.set_options(keep_attrs=True):
        if level == "surface": 
            u = ds_plot.U10
            v = ds_plot.V10
            temperature = ds_plot.T2 -273.15
        else:
            u = ds_plot.U
            v = ds_plot.V
            temperature = ds_plot.temperature -273.15

    # get the axes ready
    fig, ax = plt.subplots(tight_layout=True)

    # plot the salem map background, make countries in grey
    smap = ds_plot.salem.get_map(countries=countries)
    smap.set_shapefile(countries=countries, color='grey')
    smap.plot(ax=ax)
    smap.set_lonlat_contours(add_ytick_labels=False, interval=cord_interval, linewidths=cord_lw,
                         linestyles='-', colors='C1')
    # transform the coordinates to the map reference system and contour the data
    xx, yy = smap.grid.transform(temperature.west_east.values, temperature.south_north.values,
                                 crs=temperature.salem.grid.proj)
    clev = np.arange(vmin,vmax,0.1) 
    cs = ax.contourf(xx, yy, temperature, clev, cmap=cmap)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'T ($^\circ$C)')
    # Add the contour line levels to the colorbar
    # Quiver only every 7th grid point
    
    u = u[4::vec_gap_y, 4::vec_gap_x]
    v = v[4::vec_gap_y, 4::vec_gap_x]

    # transform their coordinates to the map reference system and plot the arrows
    xx, yy = smap.grid.transform(u.west_east.values, u.south_north.values,
                                 crs=u.salem.grid.proj)
    xx, yy = np.meshgrid(xx, yy)
    qu = ax.quiver(xx, yy, u.values, v.values,angles='xy', scale_units='xy', scale=vec_scale)
    qk = plt.quiverkey(qu, 0.8, 0.94, 
                       vec_scale*40, str(vec_scale*40)+' m s$^{-1}$',
                       labelpos='E', coordinates='figure')

    # done!
    plt.title(ds_plot.time.dt.strftime('%Y-%m-%d %X').values)
    plt.close()
    return fig

def get_vec_scale(da,scale_base=0.125):
    # get the vector scaling factor for quiver plot
    vec_scale = np.floor(da.max().values/5)*scale_base
    if vec_scale == 0:
        vec_scale = scale_base #set a minimum vec_scale
    return vec_scale

def vec_scale_vec_t(ds,level="surface"):
    #should not be directly used outside the helper
    if level == "surface":
        vec_scale = get_vec_scale(ds["wind_speed_10m"])
    else:
        vec_scale = get_vec_scale(ds.isel(bottom_top=level)["wind_speed"])
    return vec_scale
    
def plot_multi_vec_t(ds,save_dir="",fig_format="png",level="surface",countries=False,vec_scale="auto",cord_interval=5,cord_lw=1.5,cmap='viridis',vec_gap_x=7,vec_gap_y=7):
    # plot and save multiple timesteps of the the surface temperature map, with the vector on top
    # You can plot and save other height levels if you want by specify an integer for level (bottom_top index)
    if level == "surface":        
        vmin = np.floor(ds.T2.min().values -273.15)
        vmax = np.ceil(ds.T2.max().values -273.15)
    else:
        vmin = np.floor(ds.isel(bottom_top=level).temperature.min().values -273.15)
        vmax = np.ceil(ds.isel(bottom_top=level).temperature.max().values -273.15)
    #get the vector scale for the quiver plot
    if vec_scale == "auto":
        vec_scale = vec_scale_vec_t(ds,level=level)
    # get the data at the latest time step
    if (len(ds.time.values.shape) == 0):
        fig = plot_vec_t(ds,level=level,vmin=vmin,vmax=vmax,countries=countries,vec_scale=vec_scale,cord_interval=cord_interval,cord_lw=1.5,cmap=cmap,vec_gap_x=vec_gap_x,vec_gap_y=vec_gap_y)
        fig.savefig(save_dir+"vec_t"+"."+fig_format)
    else:
        for i in range(0,ds.time.shape[0]):
            mpl.rcdefaults()
            ds_sub = ds.isel(time=i)
            fig=plot_vec_t(ds_sub,level=level,vmin=vmin,vmax=vmax,countries=countries,vec_scale=vec_scale,cord_interval=cord_interval,cord_lw=1.5,cmap=cmap,vec_gap_x=vec_gap_x,vec_gap_y=vec_gap_y)
            fig.savefig(save_dir+"vec_t_"+"{0:02d}".format(i)+"."+fig_format)

            
def select_ds_time(ds,start_time,end_time):
    #essetially help select the time period
    #format can be "2020-01-07 12:00"
    time_index=(ds.time >= np.datetime64(start_time))&(ds.time <= np.datetime64(end_time))
    return ds.isel(time=time_index)

def get_avg_time_bins(ds,avg_period=3):
    time_bins = np.arange(0,ds.time.shape[0],avg_period)
    if time_bins[-1] != ds.time.shape[0]-1:
        time_bins=np.append(time_bins,ds.time.shape[0]-1)
    return time_bins

def avg_profiles(ds,avg_period="3H",calculate_wdir="auto"):
    #avg_period sets the averaging period, 3H means averaging every 3H, you can also use "5D" or "5M" for 5-daily or 5-monthly avg
    #calculate_wdir = "auto" means if the input ds has wdir varibles, then the wdir will be calculated also
    #you can set calculate_wdir to True or False to surpass.
    with xr.set_options(keep_attrs=True):
        ds_avg=ds.resample(time=avg_period).mean(dim="time")
        ds_avg["wind_speed"] = metpy.calc.wind_speed(ds_avg.U*units.meter_per_second,ds_avg.V*units.meter_per_second)
        ds_avg["wind_speed"].values = ds_avg["wind_speed"].metpy.magnitude     
        ds_avg["wind_speed_10m"] = metpy.calc.wind_speed(ds_avg.U10*units.meter_per_second,ds_avg.V10*units.meter_per_second)
        ds_avg["wind_speed_10m"].values = ds_avg["wind_speed_10m"].metpy.magnitude
        if calculate_wdir== True or (calculate_wdir == "auto" and "wind_dir" in list(ds.var())):               
            ds_avg["wind_dir_10m"] = metpy.calc.wind_direction(ds_avg.U10.compute()*units.meter_per_second,ds_avg.V10.compute()*units.meter_per_second)
            ds_avg["wind_dir_10m"].values = ds_avg["wind_dir_10m"].metpy.magnitude
            ds_avg["wind_dir"] = metpy.calc.wind_direction(ds_avg.U.compute()*units.meter_per_second,ds_avg.V.compute()*units.meter_per_second)
            ds_avg["wind_dir"].values = ds_avg["wind_dir"].metpy.magnitude
    return ds_avg

def crop_data(ds,north,south,west,east):
    # data cropping
    ds_filter_array=((ds.lat > south) & (ds.lat < north) & (ds.lon > west)& (ds.lon < east))
    return ds.where(ds_filter_array,drop=True)

def select_ds_by_hour(ds,start_hour,end_hour):
    ###
    #get datasets by specific hours.
    #for example you might put start_hour = 6 and end_hour = 18 to select all day time data
    ###
    select_matrix=((ds.time.dt.hour>=start_hour)&\
                   (ds.time.dt.hour<=end_hour))
    return ds.where(select_matrix,drop=True)