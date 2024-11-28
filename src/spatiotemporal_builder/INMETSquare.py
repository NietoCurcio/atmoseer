from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from dask.distributed import Actor
from pydantic import BaseModel

from .get_neighbors import get_bottom_neighbor, get_right_neighbor, get_upper_neighbor
from .INMETKeys import INMETKeys
from .Logger import logger

log = logger.get_logger(__name__)


class Square(BaseModel):
    top_left: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]


class INMETSquare:
    def __init__(self, inmet_keys: INMETKeys) -> None:
        self.inmet_keys = inmet_keys

    def get_keys_in_square(
        self, square: Square, stations_inmet: Actor, verbose: bool = False
    ) -> list[str]:
        keys = [x.stem for x in Path(self.inmet_keys.inmet_keys_path).glob("*.parquet")]
        inmet_keys = []
        for key in keys:
            key_lat, key_lon = map(float, key.split("_"))

            if (
                verbose
                and not (key_lat < square.bottom_left[0] or key_lat > square.top_left[0])
                and not (key_lon < square.top_left[1] or key_lon > square.top_right[1])
            ):
                log.success(f"""
                    Lat and Lon Square:
                    {square.top_left[0]}{square.top_left[1]} - {square.top_right[0]}{square.top_right[1]}
                    |              {key_lat} {key_lon}                       |
                    {square.bottom_left[0]}{square.bottom_left[1]} - {square.bottom_right[0]}{square.bottom_right[1]}
                """)

            if key_lat < square.bottom_left[0] or key_lat > square.top_left[0]:
                continue
            if key_lon < square.top_left[1] or key_lon > square.top_right[1]:
                continue

            inmet_keys.append(key)

        if len(inmet_keys) > 0:
            stations_inmet.update(inmet_keys)

        return inmet_keys

    def _find_nearest_non_null(
        self, ds_time: xr.Dataset, lat: float, lon: float, data_var: str, max_radius=1.0, step=0.1
    ) -> float:
        radius = 0.0
        while radius <= max_radius:
            ds_lat_lon = ds_time.sel(latitude=lat, longitude=lon, method="nearest")
            value = ds_lat_lon[data_var].values
            if not np.isnan(value):
                return float(value)
            lat_range = np.arange(lat - radius, lat + radius, step)
            lon_range = np.arange(lon - radius, lon + radius, step)
            for new_lat in lat_range:
                for new_lon in lon_range:
                    ds_lat_lon = ds_time.sel(latitude=new_lat, longitude=new_lon, method="nearest")
                    value = ds_lat_lon[data_var].values
                    if not np.isnan(value):
                        return float(value)
            radius += step
        log.warning(
            f"Could not find a non-null value for lat={lat}, lon={lon}, returning median of the ds_time to avoid bias"
        )
        median_ds_time = ds_time[data_var].median().item()
        if np.isnan(median_ds_time):
            log.error("median_ds_time is NaN")
            exit(1)
        return median_ds_time

    def _get_era5_single_levels_precipitation_in_square(
        self, square: Square, era5land_at_time: xr.Dataset, data_var="tp"
    ) -> float:
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]

        corner_data = {
            corner: era5land_at_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }

        for corner in corners:
            assert (
                corner_data[corner][data_var].size == 1
            ), f"{corner}['{data_var}'].size: {corner_data[corner][{data_var}].size}"

        tp_values = [corner_data[corner][data_var].item() for corner in corners]
        max_tp = max(tp_values)

        if np.isnan(max_tp):
            lat_mean, lon_mean = np.mean(coords, axis=0)
            max_tp = self._find_nearest_non_null(era5land_at_time, lat_mean, lon_mean, data_var)
        m_to_mm = 1000
        return max_tp * m_to_mm

    def get_precipitation_in_square(
        self,
        square: Square,
        inmet_keys: list[str],
        timestamp: pd.Timestamp,
        ds_time: xr.Dataset,
    ) -> float:
        if len(inmet_keys) == 0:
            return self._get_era5_single_levels_precipitation_in_square(square, ds_time)
        precipitations: list[float] = []
        for key in inmet_keys:
            df_web = self.inmet_keys.load_key(key)
            df_web_filtered = df_web[df_web.index == timestamp]
            h1 = df_web_filtered["precipitation"]
            if h1.isnull().all():
                h1 = np.array(self._get_era5_single_levels_precipitation_in_square(square, ds_time))
            precipitations.append(h1.item())
        return max(precipitations)

    def get_square(
        self,
        lat: float,
        lon: float,
        sorted_latitudes_ascending: npt.NDArray[np.float32],
        sorted_longitudes_ascending: npt.NDArray[np.float32],
    ) -> Optional[Square]:
        """
        Get the square that contains the point (lat, lon)
        Example, given this grid:
              0   1   2   3
            0 *   *   *   *
            1 *   *   *   *
            2 *   *   *   *
        Given that lat long "*" is the top_left = (0,0):
        top_left's bottom_left neighbor = (1,0)
        bottom_left's bottom_right neighbor = (1,1)
        bottom_right's top_right neighbor = (0,1)

        With top_left, bottom_left, bottom_right, top_right we can create a square

        Note we can get out of bounds, that's when we return None.
        For example, there's no bottom neighbor for (3,3)
        """
        bottom_neighbor = get_bottom_neighbor(lat, lon, sorted_latitudes_ascending)
        if bottom_neighbor is None:
            return None
        lat_bottom, lon_bottom = bottom_neighbor

        right_neighbor = get_right_neighbor(lat_bottom, lon_bottom, sorted_longitudes_ascending)
        if right_neighbor is None:
            return None
        lat_right, lon_right = right_neighbor

        upper_neighbor = get_upper_neighbor(lat_right, lon_right, sorted_latitudes_ascending)
        if upper_neighbor is None:
            return None
        lat_upper, lon_upper = upper_neighbor

        return Square(
            top_left=(lat, lon),
            bottom_left=(lat_bottom, lon_bottom),
            bottom_right=(lat_right, lon_right),
            top_right=(lat_upper, lon_upper),
        )
