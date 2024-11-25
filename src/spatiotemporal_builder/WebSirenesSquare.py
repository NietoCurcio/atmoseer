from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from dask.distributed import Lock, Variable
from pydantic import BaseModel

from .get_neighbors import get_bottom_neighbor, get_right_neighbor, get_upper_neighbor
from .Logger import logger
from .WebSirenesKeys import WebSirenesKeys

log = logger.get_logger(__name__)


class Square(BaseModel):
    top_left: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]


class WebSirenesSquare:
    def __init__(self, websirenes_keys: WebSirenesKeys) -> None:
        self.websirenes_keys = websirenes_keys

    def get_keys_in_square(self, square: Square, verbose: bool = False) -> list[str]:
        """
        Get the keys of the websirenes datasets that are inside the square
        Args:
            square (Square): The square to check for keys
        """
        keys = [x.stem for x in Path(self.websirenes_keys.websirenes_keys_path).glob("*.parquet")]
        websirenes_keys = []
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

            websirenes_keys.append(key)

            # if os.getenv("IS_SEQUENTIAL", None) == "True":
            #     continue

            with Lock("found_stations"):
                shared_stations_set: set = Variable("found_stations").get()
                shared_stations_set.add(key)
                Variable("found_stations").set(shared_stations_set)

        return websirenes_keys

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

    def get_relative_humidity_in_square(self, square: Square, ds_time: xr.Dataset):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }
        assert (
            corner_data["top_left"]["r"].size == 3
        ), f"top_left['r'].size: {corner_data['top_left']['r'].size}"

        results = []
        for corner in corner_data:
            value = corner_data[corner]["r"].values
            if np.isnan(value).any():
                lat, lon = dict(square)[corner]
                value = self._find_nearest_non_null(ds_time, lat, lon, "r")
            results.append(value)
        assert len(results) == 4, f"len(results)={len(results)} != 4"
        assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        corner_sums = [np.sum(values) for values in results]
        best_corner = corners[np.argmax(corner_sums)]
        # return corner_data[best_corner]["r"].values.astype(np.float64)
        return corner_data[best_corner]["r"].values

    def get_temperature_in_square(self, square: Square, ds_time: xr.Dataset):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }
        assert (
            corner_data["top_left"]["t"].size == 3
        ), f"top_left['t'].size: {corner_data['top_left']['t'].size}"

        results = []
        for corner in corner_data:
            value = corner_data[corner]["t"].values
            if np.isnan(value).any():
                lat, lon = dict(square)[corner]
                value = self._find_nearest_non_null(ds_time, lat, lon, "t")
            results.append(value)
        assert len(results) == 4, f"len(results)={len(results)} != 4"
        assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        corner_sums = [np.sum(values) for values in results]
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["t"].values

    def get_u_component_in_square(self, square: Square, ds_time: xr.Dataset):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }
        assert (
            corner_data["top_left"]["u"].size == 3
        ), f"top_left['u'].size: {corner_data['top_left']['u'].size}"

        results = []
        for corner in corner_data:
            value = corner_data[corner]["u"].values
            if np.isnan(value).any():
                lat, lon = dict(square)[corner]
                value = self._find_nearest_non_null(ds_time, lat, lon, "u")
            results.append(value)
        assert len(results) == 4, f"len(results)={len(results)} != 4"
        assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        corner_sums = [np.sum(values) for values in results]
        best_corner = corners[np.argmax(corner_sums)]
        datazada = corner_data[best_corner]["u"].values
        return datazada

    def get_v_component_in_square(self, square: Square, ds_time: xr.Dataset):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }
        assert (
            corner_data["top_left"]["v"].size == 3
        ), f"top_left['v'].size: {corner_data['top_left']['v'].size}"

        results = []
        for corner in corner_data:
            value = corner_data[corner]["v"].values
            if np.isnan(value).any():
                lat, lon = dict(square)[corner]
                value = self._find_nearest_non_null(ds_time, lat, lon, "v")
            results.append(value)
        assert len(results) == 4, f"len(results)={len(results)} != 4"
        assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        corner_sums = [np.sum(values) for values in results]
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["v"].values

    def get_w_component_in_square(self, square: Square, ds_time: xr.Dataset):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }
        assert (
            corner_data["top_left"]["w"].size == 3
        ), f"top_left['w'].size: {corner_data['top_left']['w'].size}"

        results = []
        for corner in corner_data:
            value = corner_data[corner]["w"].values
            if np.isnan(value).any():
                lat, lon = dict(square)[corner]
                value = self._find_nearest_non_null(ds_time, lat, lon, "w")
            results.append(value)
        assert len(results) == 4, f"len(results)={len(results)} != 4"
        assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        corner_sums = [np.sum(values) for values in results]
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["w"].values

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

    def get_features_in_square(
        self,
        square: Square,
        ds_time: xr.Dataset,
    ) -> npt.NDArray[np.float64]:
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]

        variables = ["u10", "v10", "t2m", "sp", "d2m"]

        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }

        for var in variables:
            assert (
                corner_data["top_left"][var].size == 1
            ), f"top_left['{var}'].size: {corner_data['top_left'][var].size}"

        results = []
        for var in variables:
            values = [corner_data[corner][var].item() for corner in corners]
            max_value = max(values)

            if np.isnan(max_value):
                max_value = 0.0

            results.append(max_value)
        return np.array(results)

    def get_precipitation_in_square(
        self,
        square: Square,
        websirenes_keys: list[str],
        timestamp: pd.Timestamp,
        ds_time: xr.Dataset,
    ) -> float:
        """
        It would be worth searching for ways to decrease the error in the rainfall dataset. Directions may include applying preprocessing techniques to sparse data and adding data from other geographic regions. The former is relevant since rainfall data are unbalanced with many periods without rain. The latter would be helpful to avoid overfitting the models.
        Rafaela Castro paper
        """
        if len(websirenes_keys) == 0:
            # No stations in the square, use ERA5Land precipitation in square
            return self._get_era5_single_levels_precipitation_in_square(square, ds_time)

        precipitations_15_min_aggregated: list[float] = []
        for key in websirenes_keys:
            df_web = self.websirenes_keys.load_key(key)

            time_upper_bound = timestamp
            time_lower_bound = timestamp - timedelta(minutes=45)

            df_web_filtered = df_web[
                (df_web.index >= time_lower_bound) & (df_web.index <= time_upper_bound)
            ]

            m15 = df_web_filtered["m15"]

            if m15.isnull().all():
                # All values are NaN in station "key" from "time_lower_bound" to "time_upper_bound",
                # use ERA5Land max precipitation in square
                m15 = np.array(
                    self._get_era5_single_levels_precipitation_in_square(square, ds_time)
                )
            precipitations_15_min_aggregated.append(m15.sum().item())

        max_precipitation = max(precipitations_15_min_aggregated)
        # see "ge=0, but txt has -99.99 values" comment in WebSirenesParser.py
        if max_precipitation < 0:
            return self._get_era5_single_levels_precipitation_in_square(square, ds_time)
        return max_precipitation

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
