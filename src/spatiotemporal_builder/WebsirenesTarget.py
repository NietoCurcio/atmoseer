import concurrent.futures
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .Logger import TqdmLogger, logger
from .WebSirenesParser import WebSirenesParser, websirenes_parser
from .WebSirenesSquare import WebSirenesSquare, websirenes_square

log = logger.get_logger(__name__)


class SpatioTemporalFeatures:
    def __init__(
        self,
        websirenes_parser: WebSirenesParser,
        websirenes_square: WebSirenesSquare,
    ):
        self.websirenes_features_path = Path(__file__).parent / "features"
        if not self.websirenes_features_path.exists():
            self.websirenes_features_path.mkdir()

        self.era5_single_levels_path = (
            Path(__file__).parent.parent.parent / "data/reanalysis/ERA5-single-levels"
        )
        self.era5_pressure_levels_path = (
            Path(__file__).parent.parent.parent / "data/reanalysis/ERA5-pressure-levels"
        )

        self.current_single_levels_ds = None
        self.current_pressure_levels_ds = None
        self.current_year_month = None
        self.websirenes_parser = websirenes_parser
        self.websirenes_square = websirenes_square

        lats, lons = self._get_grid_lats_lons()

        self.sorted_latitudes_ascending = np.sort(lats)
        self.sorted_longitudes_ascending = np.sort(lons)
        log.debug("SpatioTemporalFeatures initialized")
        log.debug(f"sorted_latitudes_ascending: {self.sorted_latitudes_ascending}")
        log.debug(f"sorted_longitudes_ascending: {self.sorted_longitudes_ascending}")
        log.info(
            f"Spatial resolution: {self.sorted_latitudes_ascending[1] - self.sorted_latitudes_ascending[0]:.2f} degrees"
        )

    def _write_features(self, features: npt.NDArray[np.float64], timestamp: pd.Timestamp):
        features_filename = (
            self.websirenes_features_path / f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
        )
        np.save(features_filename, features)

    def _get_grid_lats_lons(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        ds = self._get_era5_pressure_levels_dataset(2009, 6)
        lats = ds.coords["latitude"].values
        lons = ds.coords["longitude"].values
        return lats, lons

    def _get_era5_single_levels_dataset(self, year: int, month: int) -> xr.Dataset:
        if self.current_year_month == (year, month) and self.current_single_levels_ds is not None:
            return self.current_single_levels_ds

        era5_year_month_path = (
            self.era5_single_levels_path / "monthly_data" / f"RJ_{year}_{month}.nc"
        )

        if not os.path.exists(era5_year_month_path):
            raise FileNotFoundError(f"File {era5_year_month_path} not found")

        ds = xr.open_dataset(era5_year_month_path)
        # new version of ERA5 causes error in the code below
        # if "expver" in list(ds.coords.keys()):
        #     log.warning(">>>Oops! expver dimension found. Going to remove it.<<<")
        #     ds.sel(expver="0001", method="nearest")
        #     ds_combine = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        #     exit(0)
        #     ds_combine.load()
        #     ds = ds_combine
        ds = ds[["tp"]]
        self.current_single_levels_ds = ds
        self.current_year_month = (year, month)
        return self.current_single_levels_ds

    def _get_era5_pressure_levels_dataset(self, year: int, month: int) -> xr.Dataset:
        if self.current_year_month == (year, month) and self.current_pressure_levels_ds is not None:
            return self.current_pressure_levels_ds

        era5_year_month_path = (
            self.era5_pressure_levels_path / "monthly_data" / f"RJ_{year}_{month}.nc"
        )

        if not os.path.exists(era5_year_month_path):
            raise FileNotFoundError(f"File {era5_year_month_path} not found")

        ds = xr.open_dataset(era5_year_month_path)
        # new version of ERA5 causes error in the code below
        # if "expver" in list(ds.coords.keys()):
        #     log.warning(">>>Oops! expver dimension found. Going to remove it.<<<")
        #     ds_combine = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        #     ds_combine.load()
        #     ds = ds_combine
        ds = ds[["r", "t", "u", "v", "w"]]
        self.current_pressure_levels_ds = ds
        self.current_year_month = (year, month)
        return self.current_pressure_levels_ds

    def _process_grid(
        self,
        features: npt.NDArray[np.float64],
        ds_single_levels: xr.Dataset,
        ds_pressure_levels: xr.Dataset,
        timestamp: pd.Timestamp,
    ):
        top_down_lats = self.sorted_latitudes_ascending[::-1]
        left_right_lons = self.sorted_longitudes_ascending
        #
        for i, lat in tqdm(
            enumerate(top_down_lats),
            desc="Processing grid",
            leave=False,
            total=self.sorted_latitudes_ascending.size,
        ):
            for j, lon in enumerate(left_right_lons):
                square = self.websirenes_square.get_square(
                    lat, lon, self.sorted_latitudes_ascending, self.sorted_longitudes_ascending
                )

                if square is None:
                    continue

                websirene_keys = self.websirenes_square.get_keys_in_square(square)

                tp = self.websirenes_square.get_precipitation_in_square(
                    square, websirene_keys, timestamp, ds_single_levels
                )

                r1000, r700, r200 = self.websirenes_square.get_relative_humidity_in_square(
                    square, ds_pressure_levels
                )
                t1000, t700, t200 = self.websirenes_square.get_temperature_in_square(
                    square, ds_pressure_levels
                )
                u1000, u700, u200 = self.websirenes_square.get_u_component_in_square(
                    square, ds_pressure_levels
                )

                v1000, v700, v200 = self.websirenes_square.get_v_component_in_square(
                    square, ds_pressure_levels
                )
                w1000, w700, w200 = self.websirenes_square.get_w_component_in_square(
                    square, ds_pressure_levels
                )

                speed200 = np.sqrt(u200**2 + v200**2)
                speed700 = np.sqrt(u700**2 + v700**2)
                speed1000 = np.sqrt(u1000**2 + v1000**2)

                features[i, j] = [
                    tp,
                    r200,
                    r700,
                    r1000,
                    t200,
                    t700,
                    t1000,
                    u200,
                    u700,
                    u1000,
                    v200,
                    v700,
                    v1000,
                    speed200,
                    speed700,
                    speed1000,
                    w200,
                    w700,
                    w1000,
                ]

    def _process_timestamp(self, timestamp: pd.Timestamp):
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour

        ds_single_levels = self._get_era5_single_levels_dataset(year, month)
        ds_pressure_levels = self._get_era5_pressure_levels_dataset(year, month)

        time = f"{year}-{month}-{day}T{hour}:00:00.000000000"

        ds_single_levels_time = ds_single_levels.sel(valid_time=time, method="nearest")
        ds_pressure_levels_time = ds_pressure_levels.sel(valid_time=time, method="nearest")

        features_tuple = {
            "tp": "Total precipitation",
            "r200": "Relative humidity at 200 hPa",
            "r700": "Relative humidity at 700 hPa",
            "r1000": "Relative humidity at 1000 hPa",
            "t200": "Temperature at 200 hPa",
            "t700": "Temperature at 700 hPa",
            "t1000": "Temperature at 1000 hPa",
            "u200": "U component of wind",
            "u700": "U component of wind",
            "u1000": "U component of wind",
            "v200": "V component of wind",
            "v700": "V component of wind",
            "v1000": "V component of wind",
            "speed200": "Speed of wind at 200 hPa",
            "speed700": "Speed of wind at 700 hPa",
            "speed1000": "Speed of wind at 1000 hPa",
            "w200": "Vertical velocity at 200 hPa",
            "w700": "Vertical velocity at 700 hPa",
            "w1000": "Vertical velocity at 1000 hPa",
        }
        features = np.zeros(
            (
                self.sorted_latitudes_ascending.size,
                self.sorted_longitudes_ascending.size,
                len(features_tuple),
            ),
            dtype=np.float64,
        )

        self._process_grid(features, ds_single_levels_time, ds_pressure_levels_time, timestamp)
        self._write_features(features, timestamp)

    def build_timestamps_hourly(
        self,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        ignored_months: list[int],
        use_cache: bool = True,
    ):
        if use_cache and len(list(self.websirenes_features_path.glob("*.npy"))) > 0:
            log.warning(
                f"Using cached target. To clear cache delete the {self.websirenes_features_path} folder"
            )
            return

        minimum_date = self.websirenes_parser.minimum_date if start_date is None else start_date
        maximum_date = self.websirenes_parser.maximum_date if end_date is None else end_date

        assert minimum_date != pd.Timestamp.max, "minimum_date should be set during keys building"
        assert maximum_date != pd.Timestamp.min, "maximum_date should be set during keys building"

        timestamps = pd.date_range(start=minimum_date, end=maximum_date, freq="h")

        log.info(f"Building websirenes target from {timestamps[0]} to {timestamps[-1]}")
        start_time = time.time()
        THREE_MINUTES = 60 * 3
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []

            for timestamp in timestamps:
                if self.websirenes_features_path.joinpath(
                    f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
                ).exists():
                    continue

                if timestamp.month in ignored_months:
                    continue

                futures.append(executor.submit(self._process_timestamp, timestamp))

            pbar = tqdm(
                total=len(timestamps),
                desc="Processing timestamps",
                file=TqdmLogger(log),
                dynamic_ncols=True,
                mininterval=THREE_MINUTES,
            )
            for _ in concurrent.futures.as_completed(futures):
                pbar.update()
            pbar.close()
        # multiprocessing version took 22.98s 2011-04-12T21:00:00 2011-04-13T10:00:00
        # for i in tqdm(
        #     range(len(timestamps)),
        #     desc="Processing timestamps",
        #     file=TqdmLogger(log),
        #     dynamic_ncols=True,
        #     mininterval=THREE_MINUTES,
        # ):
        #     if self.websirenes_features_path.joinpath(
        #         f"{timestamps[i].strftime('%Y_%m_%d_%H')}_features.npy"
        #     ).exists():
        #         continue

        #     if timestamps[i].month in ignored_months:
        #         continue

        #     self._process_timestamp(timestamps[i])
        # sequential/synchronous version took 62.7s 2011-04-12T21:00:00 2011-04-13T10:00:00
        end_time = time.time()
        log.info(f"Target built in {end_time - start_time:.2f} seconds")

        if len(list(self.websirenes_features_path.glob("*.npy"))) != len(timestamps):
            log.error(
                f"Error building target. Expected {len(timestamps)} files but found {len(list(self.websirenes_features_path.glob('*.npy')))}"
            )
            exit(1)

        log.success(
            f"Websirenes features hourly built successfully in {self.websirenes_features_path}"
        )


websirenes_target = SpatioTemporalFeatures(websirenes_parser, websirenes_square)
