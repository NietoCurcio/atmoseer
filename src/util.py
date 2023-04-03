import pytz
from datetime import datetime, timedelta
import os
from metpy.calc import wind_components
from metpy.units import units
import numpy as np
import pandas as pd

def is_posintstring(s):
    try:
        temp = int(s)
        if temp > 0:
            return True
        else:
            return False
    except ValueError:
        return False

def format_time(input_str):
    '''
    This function first converts the input string to an integer using the int function. 
    It then extracts the hours and minutes from the input integer using integer division and modulus operations. 
    Finally, it formats the output string using f-strings to ensure that both hours and minutes are represented with two digits.
    Usage examples:
        print(format_time("100"))   # Output: "01:00"
        print(format_time("1200"))  # Output: "12:00"
        print(format_time("2300"))  # Output: "23:00"
    '''
    # Convert input string to integer
    input_int = int(input_str)
    
    # Extract hours and minutes from the input integer
    hours = input_int // 100
    minutes = input_int % 100
    
    # Format the output string
    output_str = f"{hours:02}:{minutes:02}"
    
    return output_str

def utc_to_local(utc_string, local_tz, format_string):
    '''
    This function first converts the UTC string to a datetime object with timezone information 
    using strptime() and replace(). Then it converts the datetime object to the local timezone 
    using astimezone(). Finally, it formats the resulting datetime as a string in the same
    format as the input string.

    Here's an example usage of the function:
    
        utc_string = '1972-09-13 12:00:00'
        local_tz = 'America/Sao_Paulo'
        format_string = '%Y-%m-%d %H:%M'
        local_string = utc_to_local(utc_string, local_tz, format_string)
        print(local_string)

    '''
    # convert utc string to datetime object
    utc_dt = datetime.strptime(utc_string, format_string).replace(tzinfo=pytz.UTC)
    
    # convert to local timezone
    local_tz = pytz.timezone(local_tz)
    local_dt = utc_dt.astimezone(local_tz)
    
    # format as string and return
    return local_dt.strftime(format_string)

def transform_wind(wind_speed, wind_direction, comp_idx):
    """
    This function calculates either the U or V wind vector component from the speed and direction.
    comp_idx = 0 --> computes the U component
    comp_idx = 1 --> computes the V component
    (see https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.wind_components.html)
    """
    assert(comp_idx == 0 or comp_idx == 1)
    return wind_components(wind_speed * units('m/s'), wind_direction * units.deg)[comp_idx].magnitude

def transform_hour(df):
    """
    Transforms a DataFrame's datetime index into two new columns representing the hour in sin and cosine form.
    (see https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes)

    Args:
    - df: A pandas DataFrame with a datetime index.

    Returns:
    - The input pandas DataFrame with two new columns named 'hour_sin' and 'hour_cos' representing the hour in sin and cosine form.
    """
    dt = df.index
    hourfloat = dt.hour + dt.minute/60.0
    df['hour_sin'] = np.sin(2. * np.pi * hourfloat/24.)
    df['hour_cos'] = np.cos(2. * np.pi * hourfloat/24.)
    return df

def get_filename_and_extension(filename):
    """
    Given a filename, returns a tuple with the base filename and extension.
    """
    basename = os.path.basename(filename)
    filename_parts = os.path.splitext(basename)
    return filename_parts

def find_contiguous_observation_blocks(df: pd.DataFrame):
    """
    Given a list of timestamps, finds contiguous blocks of timestamps that are exactly one hour apart of each other.

    Args:
    - timestamp_range: A list-like object of pandas Timestamps.

    Yields:
    - A tuple representing a contiguous block of timestamps: (start, end).

    Usage example:
    >>> period_under_study = df['2007-05-18':'2007-05-31']
    >>> contiguous_observations = list(find_contiguous_observation_blocks(period_under_study))
    >>> print(len(contiguous_observations))
    >>> print(contiguous_observations)
    5
    [(Timestamp('2007-05-18 18:00:00'), Timestamp('2007-05-18 19:00:00')), 
     (Timestamp('2007-05-19 11:00:00'), Timestamp('2007-05-19 13:00:00')), 
     (Timestamp('2007-05-20 12:00:00'), Timestamp('2007-05-21 00:00:00')), 
     (Timestamp('2007-05-21 02:00:00'), Timestamp('2007-05-21 08:00:00')), 
     (Timestamp('2007-05-21 10:00:00'), Timestamp('2007-05-31 23:00:00'))]

    In this example, `timestamp_range` is a list of pandas Timestamps extracted from a DataFrame's 'Datetime' index.
    The function finds contiguous blocks of timestamps that are exactly one hour apart, and yields tuples representing these blocks.
    The `yield` statement produces a generator object, which can be converted to a list using the `list()` function.

    Returns:
    - None
    """
    timestamp_range = df.index
    assert (len(timestamp_range) > 1)
    first = last = timestamp_range[0]
    for n in timestamp_range[1:]:
        previous = n - timedelta(hours=1, minutes=0)
        if previous == last: # Part of the current block, bump the end
            last = n
        else: # Not part of the current block; yield current block and start a new one
            yield first, last
            first = last = n
    yield first, last # Yield the last block
