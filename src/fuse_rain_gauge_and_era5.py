from pathlib import Path
import argparse

import pandas as pd
import xarray as xr

import util

def fuse_rain_gauge_and_era5(df_station: pd.DataFrame, station_latitude: float, station_longitude: float, ds_era5):
    df_station['datetime'] = df_station['datetime'].dt.tz_convert('UTC')
    df_station['datetime'] = df_station['datetime'].dt.tz_localize(None)
    df_station = df_station.set_index(pd.DatetimeIndex(df_station['datetime']))
    df_station = df_station.drop(['datetime'], axis = 1)

    time_min = min(df_station.index)
    time_max = max(df_station.index)
    print(f"Range of timestamps: [{time_min}, {time_max}]")
    print(f"Number of observations (gauge station): {df_station.shape[0]}")

    # Select ERA5 data near the station
    era5_data_at_1000hPa = ds_era5.sel(level=1000, longitude=station_longitude, latitude=station_latitude, method="nearest")

    df_era5_data_for_station = pd.DataFrame(
        {
            "time": era5_data_at_1000hPa.time.values,
            "pressure_1000": 1000,
            "Humidity_1000": era5_data_at_1000hPa.r,
            "Temperature_1000": era5_data_at_1000hPa.t,
            "WindU_1000": era5_data_at_1000hPa.u,
            "WindV_1000": era5_data_at_1000hPa.v
        }
    )

    # Create a datetime index for the dataframe containing ERA5 simulated data.
    format_string = '%Y-%m-%d %H:%M:%S'
    df_era5_data_for_station['datetime'] = pd.to_datetime(df_era5_data_for_station['time'], format=format_string)
    df_era5_data_for_station = df_era5_data_for_station.set_index(pd.DatetimeIndex(df_era5_data_for_station['datetime']))
    df_era5_data_for_station = df_era5_data_for_station.drop(['time', 'datetime'], axis = 1)
    
    # Temperature in ERA5 is provided in Kelvin; convert to Celsius.
    df_era5_data_for_station["Temperature_1000_Celsius"] = df_era5_data_for_station["Temperature_1000"].apply(util.convert_to_celsius)
    
    assert (not df_era5_data_for_station.isnull().values.any().any())

    print("Index of df_station:", df_station.index)
    print("Index of df_era5_data_for_station:", df_era5_data_for_station.index)
    # alertario resolution is '2016-01-01 02:00:00', 2016-01-01 02:15:00', '2016-01-01 02:30:00', ...
    # UPDATE alertario during "data retrieval step" is 1 hour time resolution now
    # while ERA5 resolution is '1997-01-01 00:00:00'', '1997-01-01 01:00:00', '1997-01-01 02:00:00', ...
    # so at index 2:15, 2:30 and 2:45 barometric pressure will be NA
    df_fusion = pd.merge(df_station, df_era5_data_for_station, how='left', left_index=True, right_index=True)
    if df_fusion['pressure_1000'].isnull().values.any():
        print("Time mismatch between df_station and df_era5_data_for_station detected, dropping extra data")
        df_fusion = df_fusion.dropna(subset=['pressure_1000'])
    
    column_name_mapping = {
      "Temperature_1000_Celsius": "temperature",
      "Humidity_1000": "relative_humidity",
      "pressure_1000": "barometric_pressure",
      "WindU_1000": "wind_direction_u",
      "WindV_1000": "wind_direction_v",
      "precipitation_sum": "precipitation"
    }
    column_names = column_name_mapping.keys()

    df_fusion = util.get_dataframe_with_selected_columns(df_fusion, column_names)

    df_fusion = util.rename_dataframe_column_names(df_fusion, column_name_mapping)

    print(f"Number of observations (fused gauge station): {df_fusion.shape[0]}")

    return df_fusion

def fuse_rain_gauges_and_era5(dataset_file: str, entity: str = "alertario"):
    era5_filename = dataset_file
    ds_era5 = xr.open_dataset(era5_filename)
    time_min = ds_era5.time.min().values
    time_max = ds_era5.time.max().values
    print(f"Range of timestamps in the ERA5 data: [{time_min}, {time_max}]")

    entity_paths = {
        "alertario": {
            "stations_filename": "./data/ws/alertario_stations.parquet",
            "rain_gauge_folder": "./data/ws/alertario/rain_gauge/",
            "output_folder": "./data/ws/alertario/rain_gauge_era5_fused/"
        },
        "websirenes": {
            "stations_filename": "./data/ws/websirenes_stations.parquet",
            "rain_gauge_folder": "./data/ws/websirenes/rain_gauge/",
            "output_folder": "./data/ws/websirenes/rain_gauge_era5_fused/"
        }
    }

    entity_data = entity_paths.get(entity)

    stations_filename = entity_data["stations_filename"]
    rain_gauge_folder = entity_data["rain_gauge_folder"]
    output_folder = entity_data["output_folder"]

    df_stations = pd.read_parquet(stations_filename)
    for index, row in df_stations.iterrows():
        station_id = row["estacao_desc"]
        print(f"Fusing data for gauge station {station_id}")
        station_latitude = row["latitude"]
        station_longitude = row["longitude"]
        df_station = pd.read_parquet(rain_gauge_folder + station_id + ".parquet")
        df_fusion_result = fuse_rain_gauge_and_era5(df_station, station_latitude, station_longitude, ds_era5)
        df_fusion_result.to_parquet(output_folder + station_id + ".parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse rain gauge and ERA5 data')
    parser.add_argument('-ds', '--dataset_file', type=str, required=True, help='Path to the ERA5 dataset file (.nc)')
    entity_action = parser.add_argument(
            '-e',
            '--entity',
            type=str,
            required=True,
            help='Entity name', choices=["alertario", "websirenes"]
    )

    args = parser.parse_args()
    dataset_file = args.dataset_file
    entity = args.entity

    if entity not in entity_action.choices:
        parser.print_help()
        raise ValueError(f"Invalid entity name: {entity}")

    if not Path(dataset_file).is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    fuse_rain_gauges_and_era5(dataset_file, entity)
