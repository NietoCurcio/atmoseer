import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
import sys

def compute_indices(df_probe_original):
    df_probe = df_probe_original.drop_duplicates(subset='pressure', ignore_index=True)

    df_probe = df_probe.dropna()

    # df_probe.reset_index(inplace = True)
    
    pressure_values = df_probe['pressure'].to_numpy() * units.hPa
    temperature_values = df_probe['temperature'].to_numpy() * units.degC
    dewpoint_values = df_probe['dewpoint'].to_numpy() * units.degC

    parcel_profile = mpcalc.parcel_profile(pressure_values, 
                                           df_probe['temperature'][0] * units.degC, 
                                           df_probe['dewpoint'][0] * units.degC)
    parcel_profile =  parcel_profile.magnitude * units.degC

    indices = dict()

    CAPE = mpcalc.surface_based_cape_cin(pressure_values, temperature_values, dewpoint_values)
    indices['cape'] = CAPE[0].magnitude
    indices['cin'] = CAPE[1].magnitude

    lift = mpcalc.lifted_index(pressure_values, temperature_values, parcel_profile)
    indices['lift'] = lift[0].magnitude

    k = mpcalc.k_index(pressure_values, temperature_values, dewpoint_values)
    indices['k'] = k.magnitude

    total_totals = mpcalc.total_totals_index(pressure_values, temperature_values, dewpoint_values)
    indices['total_totals'] = total_totals.magnitude

    showalter = mpcalc.showalter_index(pressure_values, temperature_values, dewpoint_values)
    indices['showalter'] = showalter.magnitude[0]

    return indices
    
def main():
    input_file = '../data/radiosonde/SBGL_1997-01-01_2022-12-31.csv'
    output_file = '../data/radiosonde/SBGL_indices_1997-01-01_2022-12-31.csv'
    
    # input_file = '../data/radiosonde/2012-02-02.csv'
    # output_file = '../data/radiosonde/2012-02-02_indices.csv'
    
    dtype_dict = {'pressure': 'float',
                  'height': 'float',
                  'temperature': 'float',
                  'dewpoint': 'float',
                  'direction': 'float',
                  'speed': 'float',
                  'u_wind': 'float',
                  'v_wind': 'float',
                  'station': 'str',
                  'station_number': 'int',
                  'time': 'str',
                  'latitude': 'float',
                  'longitude': 'float',
                  'elevation': 'float',
                  'pw': 'float'}

    df_s = pd.read_csv(input_file, header=0, dtype=dtype_dict, on_bad_lines='skip')

    df_indices = pd.DataFrame(columns = ['time', 'cape', 'cin', 'lift', 'k', 'total_totals', 'showalter'])

    for probe_timestamp in pd.to_datetime(df_s.time).unique():
        try:
            df_probe = df_s[pd.to_datetime(df_s['time']) == probe_timestamp]
            indices_dict = compute_indices(df_probe)
            indices_dict['time'] = probe_timestamp
            df_indices = pd.concat([df_indices, pd.DataFrame.from_records([indices_dict])])
        except ValueError as e:
            print(f'Error processing probe at timestamp {probe_timestamp}')
            print(f'{repr(e)}')
        except IndexError as e:
            print(f'Error processing probe at timestamp {probe_timestamp}')
            print(f'{repr(e)}')
        except KeyError as e:
            print(f'Error processing probe at timestamp {probe_timestamp}')
            print(f'{repr(e)}')

    df_indices.to_csv(output_file, index = False)

if __name__ == "__main__":
    main()
