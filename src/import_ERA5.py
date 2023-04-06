import pandas as pd
import sys, getopt
from datetime import datetime, timedelta
from util import is_posintstring
import time
import cdsapi
from pathlib import Path
import xarray as xr
import requests

'''
    For using the CDS API to download ERA-5 data consult: https://cds.climate.copernicus.eu/api-how-to
'''

def get_data(station_name, start_date, end_date):

    print(f"Downloading data from radiosonde {station_name}...")

    today = datetime.today()
    end_date = min([end_date, today.strftime("%Y")])

    file = 'RJ_'+str(start_date)+'_'+str(end_date)
    
    file_exist = Path('../data/'+file+'.nc')

    if file_exist.is_file():
        ds = xr.open_dataset('../data/'+ file +'.nc')
        ds2 = xr.open_dataset('../data/'+ file +'_200.nc')
    else:
        c = cdsapi.Client()

        unsuccesfully_downloaded_probes = 0

        year = list(map(str,range(start_date,end_date)))
        for i in year:
            try:
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            'geopotential', 'relative_humidity','temperature', 'u_component_of_wind', 'v_component_of_wind',
                        ],
                        'pressure_level': [
                            '1000', '700',
                        ],
                        'year': [
                            i,
                        ],
                        'month': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                        ],
                        'day': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': [
                            -22, -44, -23,
                            -42,
                        ],
                    },
                    '../data/ERA-5/RJ_'+ i +'.nc')
                first_probe = 'RJ_'+ i 
                print(f"Data successfully downloaded for {first_probe}.")
            except IndexError as e:
                print(f'{repr(e)}')
                unsuccesfully_downloaded_probes += 1
            except ValueError as e:
                print(f'{str(e)}')
                unsuccesfully_downloaded_probes += 1
            except requests.HTTPError as e:
                print(f'{repr(e)}')
                print("Server seems to be busy. Going to sleep for a while...")
                time.sleep(10) # Sleep for a moment
                print("Back to work!")
                continue
            except Exception as e:
                print(f'Unexpected error! {repr(e)}')
                sys.exit(2)
        
        ano = list(range(start_date,end_date,2))
        ano2 = list(range(start_date,end_date,2))

        year2 = []
        for i in range(0,len(ano2)):
            year2 = year2 + [[str(ano[i]),str(ano2[i])]]

        for i in year2:
            try:
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            'geopotential', 'relative_humidity','temperature', 'u_component_of_wind', 'v_component_of_wind',
                        ],
                        'pressure_level': [
                            '200',
                        ],
                        'year': [
                        i[0],i[1],
                        ],
                        'month': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                        ],
                        'day': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': [
                            -22, -44, -23,
                            -42,
                        ],
                    },
                    '../data/ERA-5/RJ_'+ i[0]+'_'+i[1] +'_200.nc')
                second_probe = 'RJ_'+ i[0]+'_'+i[1] +'_200'
                print(f"Data successfully downloaded for {second_probe}.")
            except IndexError as e:
                print(f'{repr(e)}')
                unsuccesfully_downloaded_probes += 1
            except ValueError as e:
                print(f'{str(e)}')
                unsuccesfully_downloaded_probes += 1
            except requests.HTTPError as e:
                print(f'{repr(e)}')
                print("Server seems to be busy. Going to sleep for a while...")
                time.sleep(10) # Sleep for a moment
                print("Back to work!")
                continue
            except Exception as e:
                print(f'Unexpected error! {repr(e)}')
                sys.exit(2)

        for i in year:
            if i == str(start_date):
                ds = xr.open_dataset('../data/ERA-5/RJ_'+ i +'.nc')
            else:
                ds_aux = xr.open_dataset('../data/ERA-5/RJ_'+ i +'.nc')
                ds = ds.merge(ds_aux) 
        
        for i in year2:
            if i[0] == str(start_date):
                ds2 = xr.open_dataset('../data/ERA-5/RJ_'+ i[0]+'_'+i[1] +'_200.nc')
            else:
                ds_aux2 = xr.open_dataset('../data/ERA-5/RJ_'+ i[0]+'_'+i[1] +'_200.nc')
                ds2 = ds2.merge(ds_aux2)

        print(f"Done! Number of unsuccesfully downloaded probes: {unsuccesfully_downloaded_probes}.")
        ds.to_netcdf('../data/'+file+'.nc')
        ds2.to_netcdf('../data/'+file+'_200.nc')
    
    df_stations = pd.read_csv('../data/estacoes_local.csv')
    df_stations = df_stations[df_stations['files'] == station_name]
    latitude_aux = df_stations['VL_LATITUDE'].iloc[0]
    longitude_aux = df_stations['VL_LONGITUDE'].iloc[0]

    era5_data = ds.sel(level = 1000, longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
    era5_data2 = ds.sel(level = 700, longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
    era5_data3 = ds2.sel(longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
    
    df_era = pd.DataFrame({'time': era5_data.time,
                           'Geopotential_1000': era5_data.z, 
                           'Humidity_1000': era5_data.r,
                           'Temperature_1000': era5_data.t, 
                           'WindU_1000': era5_data.u, 
                           'WindV_1000': era5_data.v,
                           'Geopotential_700': era5_data2.z, 
                           'Humidity_700': era5_data2.r,
                           'Temperature_700': era5_data2.t, 
                           'WindU_700': era5_data2.u, 
                           'WindV_700': era5_data2.v,
                           'Geopotential_200': era5_data3.z, 
                           'Humidity_200': era5_data3.r,
                           'Temperature_200': era5_data3.t, 
                           'WindU_200': era5_data3.u, 
                           'WindV_200': era5_data3.v})
    
    filename = '../data/ERA5_' + station_name + '_'+ start_date + '_' + end_date + '.csv'
    print(f"Saving dowloaded content to file {filename}.")
    df_era.to_csv(filename, index = False)

def main(argv):
    help_message = "Usage: {0} -s <station_name> -b <start_year> -e <end_year>".format(argv[0])

    station_name = 'SBGL'

    date_format_str = '%Y-%m-%d'
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:b:e:", ["help","station_name=","start_year=","end_year="])
    except:
        print("Invalid arguments. Use -h or --help for more information.")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit(2)
        elif opt in ("-s", "--station_name"):
            station_name = arg
        elif opt in ("-b", "--start_date"):
            try:
                start_date = datetime.strptime(arg, date_format_str)
            except ValueError:
                print("Invalid date format. Use -h or --help for more information.")
                sys.exit(2)
        elif opt in ("-e", "--end_date"):
            try:
                end_date = datetime.strptime(arg, date_format_str)
            except ValueError:
                print("Invalid date format. Use -h or --help for more information.")
                sys.exit(2)

    assert (station_name is not None) and (station_name != '')
    assert (start_date <= end_date)

    get_data(station_name, start_date, end_date)


if __name__ == "__main__":
    main(sys.argv)