import numpy as np
import xarray as xr
from pydantic import BaseModel

from .Logger import logger

log = logger.get_logger(__name__)


class Square(BaseModel):
    top_left: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]


class ERA5Square:
    def _find_nearest_non_null(
        self, ds_time: xr.Dataset, lat: float, lon: float, data_var: str, max_radius=1.0, step=0.1
    ) -> float:
        log.error(f"NaN detected in {lat}-{lon}. This should not happen for ERA5")

        # This function is useful for ERA5Land where we don't have ocean data
        log.warning(f"Finding nearest non-null value called for lat={lat}, lon={lon}")
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

    def get_relative_humidity_in_square(
        self, square: Square, ds_time: xr.Dataset, corner_data: dict
    ):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        # coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        # corner_data = {
        #     corner: ds_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }

        pressure_levels_length = len([1000, 700, 200])
        # assert corner_data["top_left"]["r"].size == pressure_levels_length, (
        #     f"top_left['r'].size: {corner_data['top_left']['r'].size}"
        # )

        # results = []
        # for corner in corner_data:
        #     # value = corner_data[corner]["r"].values
        #     value = corner_data[corner]["r"].values
        #     # value = corner_data[corner]["r"].compute().values
        #     if np.isnan(value).any():
        #         lat, lon = dict(square)[corner]
        #         value = self._find_nearest_non_null(ds_time, lat, lon, "r")
        #     results.append(value)

        results = np.array([corner_data[corner]["r"] for corner in corners], dtype=np.float32)

        # assert len(results) == len(corners), f"len(results)={len(results)} != 4"
        # assert len(results[0]) == pressure_levels_length, f"len(results[0])={len(results[0])} != 3"
        # corner_sums = [np.sum(values) for values in results]
        corner_sums = np.sum(results, axis=1)
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["r"].values

    def get_temperature_in_square(
        self, square: Square, ds_time: xr.Dataset, corner_data: dict, verbose=False
    ):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        # coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        # corner_data = {
        #     corner: ds_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }

        pressure_levels_length = len([1000, 700, 200])
        # assert corner_data["top_left"]["t"].size == pressure_levels_length, (
        #     f"top_left['t'].size: {corner_data['top_left']['t'].size}"
        # )

        # results = []
        # for corner in corner_data:
        #     value = corner_data[corner]["t"].values
        #     # value = corner_data[corner]["t"].compute().values
        #     if np.isnan(value).any():
        #         lat, lon = dict(square)[corner]
        #         value = self._find_nearest_non_null(ds_time, lat, lon, "t")
        #     results.append(value)

        # results = np.empty((len(corners), 3), dtype=np.float32)
        # for i, corner in enumerate(corner_data):
        #     results[i] = corner_data[corner]["t"].values

        results = np.array([corner_data[corner]["t"] for corner in corners], dtype=np.float32)

        # assert len(results) == len(corners), f"len(results)={len(results)} != 4"
        # assert len(results[0]) == pressure_levels_length, f"len(results[0])={len(results[0])} != 3"

        # corner_sums = [np.sum(values) for values in results]
        corner_sums = np.sum(results, axis=1)
        best_corner = corners[np.argmax(corner_sums)]

        if verbose:
            log.debug(f"""
                corner_data:
                Top left lat lon: ({corner_data["top_left"].latitude.values}, {corner_data["top_left"].longitude.values})
                Top left temperature: {corner_data["top_left"]["t"].values}
                Top left pressure levels: {corner_data["top_left"]["pressure_level"].values}

                Bottom left lat lon: ({corner_data["bottom_left"].latitude.values}, {corner_data["bottom_left"].longitude.values})
                Bottom left temperature: {corner_data["bottom_left"]["t"].values}
                Bottom left pressure levels: {corner_data["bottom_left"]["pressure_level"].values}

                Bottom right lat lon: ({corner_data["bottom_right"].latitude.values}, {corner_data["bottom_right"].longitude.values})
                Bottom right temperature: {corner_data["bottom_right"]["t"].values}
                Bottom right pressure levels: {corner_data["bottom_right"]["pressure_level"].values}

                Top right lat lon: ({corner_data["top_right"].latitude.values}, {corner_data["top_right"].longitude.values})
                Top right temperature: {corner_data["top_right"]["t"].values}
                Top right pressure levels: {corner_data["top_right"]["pressure_level"].values}

                best_corner (most significant): {best_corner}
                return: {corner_data[best_corner]["t"].values}
            """)

        return corner_data[best_corner]["t"].values

    def get_u_component_in_square(self, square: Square, ds_time: xr.Dataset, corner_data: dict):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        # coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        # corner_data = {
        #     corner: ds_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }
        # assert corner_data["top_left"]["u"].size == 3, (
        #     f"top_left['u'].size: {corner_data['top_left']['u'].size}"
        # )

        # results = []
        # for corner in corner_data:
        #     value = corner_data[corner]["u"].values
        #     # value = corner_data[corner]["u"].compute().values
        #     if np.isnan(value).any():
        #         lat, lon = dict(square)[corner]
        #         value = self._find_nearest_non_null(ds_time, lat, lon, "u")
        #     results.append(value)

        # results = np.empty((len(corners), 3), dtype=np.float32)
        # for i, corner in enumerate(corner_data):
        #     results[i] = corner_data[corner]["u"].values
        results = np.array([corner_data[corner]["u"] for corner in corners], dtype=np.float32)

        # assert len(results) == 4, f"len(results)={len(results)} != 4"
        # assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        # corner_sums = [np.sum(values) for values in results]
        corner_sums = np.sum(results, axis=1)
        best_corner = corners[np.argmax(corner_sums)]
        datazada = corner_data[best_corner]["u"].values
        return datazada

    def get_v_component_in_square(self, square: Square, ds_time: xr.Dataset, corner_data: dict):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        # coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        # corner_data = {
        #     corner: ds_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }
        # assert corner_data["top_left"]["v"].size == 3, (
        #     f"top_left['v'].size: {corner_data['top_left']['v'].size}"
        # )

        # results = []
        # for corner in corner_data:
        #     value = corner_data[corner]["v"].values
        #     # value = corner_data[corner]["v"].compute().values
        #     if np.isnan(value).any():
        #         lat, lon = dict(square)[corner]
        #         value = self._find_nearest_non_null(ds_time, lat, lon, "v")
        #     results.append(value)

        # results = np.empty((len(corners), 3), dtype=np.float32)
        # for i, corner in enumerate(corner_data):
        #     results[i] = corner_data[corner]["v"].values
        results = np.array([corner_data[corner]["v"] for corner in corners], dtype=np.float32)

        # assert len(results) == 4, f"len(results)={len(results)} != 4"
        # assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        # corner_sums = [np.sum(values) for values in results]
        corner_sums = np.sum(results, axis=1)
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["v"].values

    def get_w_component_in_square(self, square: Square, ds_time: xr.Dataset, corner_data: dict):
        # all these functions violate the dry principle, but I decided to repeat them to leave the code "open to change"
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        # coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]
        # corner_data = {
        #     corner: ds_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }
        # assert corner_data["top_left"]["w"].size == 3, (
        #     f"top_left['w'].size: {corner_data['top_left']['w'].size}"
        # )

        # results = []
        # for corner in corner_data:
        #     value = corner_data[corner]["w"].values
        #     # value = corner_data[corner]["w"].compute().values
        #     if np.isnan(value).any():
        #         lat, lon = dict(square)[corner]
        #         value = self._find_nearest_non_null(ds_time, lat, lon, "w")
        #     results.append(value)

        # results = np.empty((len(corners), 3), dtype=np.float32)
        # for i, corner in enumerate(corner_data):
        #     results[i] = corner_data[corner]["w"].values
        results = np.array([corner_data[corner]["w"] for corner in corners], dtype=np.float32)

        # assert len(results) == 4, f"len(results)={len(results)} != 4"
        # assert len(results[0]) == 3, f"len(results[0])={len(results[0])} != 3"
        # corner_sums = [np.sum(values) for values in results]
        corner_sums = np.sum(results, axis=1)
        best_corner = corners[np.argmax(corner_sums)]
        return corner_data[best_corner]["w"].values

    def get_era5_single_levels_precipitation_in_square(
        self,
        square: Square,
        era5_at_time: xr.Dataset,
        corner_data: dict,
        data_var="tp",
        verbose=False,
    ) -> float:
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]

        # corner_data = {
        #     corner: era5_at_time.sel(latitude=lat, longitude=lon)
        #     for corner, (lat, lon) in zip(corners, coords)
        # }

        single_levels_length = 1
        # for corner in corners:
        # assert corner_data[corner][data_var].size == single_levels_length, (
        #     f"{corner}['{data_var}'].size: {corner_data[corner][{data_var}].size}"
        # )

        tp_values = [corner_data[corner][data_var].item() for corner in corners]
        max_tp = max(tp_values)

        if verbose:
            log.debug(f"""
                corner_data:
                Top left lat lon: ({corner_data["top_left"].latitude.values}, {corner_data["top_left"].longitude.values})
                Top left precipitation: {corner_data["top_left"][data_var].values}

                Bottom left lat lon: ({corner_data["bottom_left"].latitude.values}, {corner_data["bottom_left"].longitude.values})
                Bottom left precipitation: {corner_data["bottom_left"][data_var].values}

                Bottom right lat lon: ({corner_data["bottom_right"].latitude.values}, {corner_data["bottom_right"].longitude.values})
                Bottom right precipitation: {corner_data["bottom_right"][data_var].values}

                Top right lat lon: ({corner_data["top_right"].latitude.values}, {corner_data["top_right"].longitude.values})
                Top right precipitation: {corner_data["top_right"][data_var].values}

                max_tp: {max_tp}
            """)

        if np.isnan(max_tp):
            lat_mean, lon_mean = np.mean(coords, axis=0)
            max_tp = self._find_nearest_non_null(era5_at_time, lat_mean, lon_mean, data_var)
        m_to_mm = 1000
        return max_tp * m_to_mm

    def get_precipitation_in_square(
        self,
        square: Square,
        ds_time: xr.Dataset,
    ) -> float:
        return self.get_era5_single_levels_precipitation_in_square(square, ds_time, verbose=True)


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.ERA5Square
    import pandas as pd

    from .square import get_square

    timestamp = pd.Timestamp("2022-10-31T18:00:00")
    year = timestamp.year
    month = timestamp.month

    era5_square = ERA5Square()
    ds = xr.open_dataset(f"./data/reanalysis/ERA5-single-levels/monthly_data/RJ_{year}_{month}.nc")
    ds = ds.sel(valid_time=timestamp)
    lats = ds.latitude.values
    lons = ds.longitude.values
    log.debug("single levels dataset:")
    log.debug(ds)

    log.debug("lats:")
    log.debug(lats)

    log.debug("lons:")
    log.debug(lons)

    log.info(f"Spatial resolution: {sorted(lats)[1] - sorted(lats)[0]:.2f} degrees")
    log.debug(f"Grid: {lats.shape[0]}x{lons.shape[0]}")

    lat = sorted(lats)[4]
    lon = sorted(lons)[7]

    square = get_square(lat, lon, sorted(lats), sorted(lons))

    precipitation = era5_square.get_precipitation_in_square(square, ds)

    log.success(f"""
        Precipitation in square:
        top_left={square.top_left}
        bottom_left={square.bottom_left}
        bottom_right={square.bottom_right}
        top_right={square.top_right}
        {square.top_left} --- {square.top_right}
        | {" " * 20} {precipitation:.2f} mm  {" " * 20} |
        {square.bottom_left} --- {square.bottom_right}
    """)
    ds.close()

    ds = xr.open_dataset(
        f"./data/reanalysis/ERA5-pressure-levels/monthly_data/RJ_{year}_{month}.nc"
    )
    ds = ds.sel(valid_time=timestamp)
    log.debug("pressure levels dataset:")
    log.debug(ds)

    temperature = era5_square.get_temperature_in_square(square, ds, verbose=True)
    log.success(f"""
        Temperature in square:
        top_left={square.top_left}
        bottom_left={square.bottom_left}
        bottom_right={square.bottom_right}
        top_right={square.top_right}
        {square.top_left} --- {square.top_right}
        | {" " * 10} {temperature}  {" " * 10} |
        {square.bottom_left} --- {square.bottom_right}
    """)

    ds.close()
