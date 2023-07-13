import pandas as pd
import xarray as xr

def convert_to_celsius(temperature_kelvin):
  """Converts a temperature from Kelvin to Celsius.

  Args:
    temperature_kelvin: The temperature in Kelvin.

  Returns:
    The temperature in Celsius.
  """

  return temperature_kelvin - 273.15

def fuse_rain_gauge_and_nwp(df_station: pd.DataFrame, station_latitude: float, station_longitude: float, ds_era5):
    df_station['datetime'] = df_station['datetime'].dt.tz_convert('UTC')
    df_station['datetime'] = df_station['datetime'].dt.tz_localize(None)
    df_station = df_station.set_index(pd.DatetimeIndex(df_station['datetime']))
    df_station = df_station.drop(['datetime'], axis = 1)
    
    # Select ERA5 data near the station
    era5_data_at_1000hPa = ds_era5.sel(level=1000, longitude=station_longitude, latitude=station_latitude, method="nearest")

    df_NWP_data_for_station = pd.DataFrame(
        {
            "time": era5_data_at_1000hPa.time.values,
            "pressure_1000": 1000,
            "Humidity_1000": era5_data_at_1000hPa.r,
            "Temperature_1000": era5_data_at_1000hPa.t,
            "WindU_1000": era5_data_at_1000hPa.u,
            "WindV_1000": era5_data_at_1000hPa.v
        }
    )

    # Create a datetime index for the ERA dataframe
    format_string = '%Y-%m-%d %H:%M:%S'
    df_NWP_data_for_station['datetime'] = pd.to_datetime(df_NWP_data_for_station['time'], format=format_string)
    df_NWP_data_for_station = df_NWP_data_for_station.set_index(pd.DatetimeIndex(df_NWP_data_for_station['datetime']))
    df_NWP_data_for_station = df_NWP_data_for_station.drop(['time', 'datetime'], axis = 1)
    
    # Temperature in ERA5 is provided in Kelvin; convert to Celsius.
    df_NWP_data_for_station["Temperature_1000_Celsius"] = df_NWP_data_for_station["Temperature_1000"].apply(convert_to_celsius)
    df_NWP_data_for_station.head()
    
    df_fusion = pd.merge(df_station, df_NWP_data_for_station, how='left', left_index=True, right_index=True)
    
    return df_fusion

def fuse_rain_gauges_and_nwp():
    era5_filename = "./data/NWP/ERA5/RJ_1997_2023.nc"
    ds_era5 = xr.open_dataset(era5_filename)
    alertario_stations_filename = "./data/ws/alertario_stations.parquet"
    df_alertario_stations = pd.read_parquet(alertario_stations_filename)
    for index, row in df_alertario_stations.iterrows():
        station_id = row["estacao_desc"]
        station_latitude = row["latitude"]
        station_longitude = row["longitude"]
        df_station = pd.read_parquet("./data/ws/alertario/rain_gauge/" + station_id + ".parquet")
        df_fusion = fuse_rain_gauge_and_nwp(df_station, station_latitude, station_longitude, ds_era5)
        df_fusion.to_parquet("./data/ws/alertario/rain_gauge/fused_" + station_id + ".parquet")

if __name__ == "__main__":
  fuse_rain_gauges_and_nwp()
