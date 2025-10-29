import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .AlertarioSquare import AlertarioSquare
from .INMETSquare import INMETSquare
from .Logger import TqdmLogger, logger
from .settings import settings
from .square import Square, get_square
from .WebSirenesSquare import WebSirenesSquare

log = logger.get_logger(__name__)


class StationsManager(BaseManager):
    pass


StationsManager.register("Set", set)
StationsManager.register("Dict", dict)


class SpatioTemporalFeatures:
    manager = StationsManager()
    manager.start()
    stations_cells = manager.Set()
    stations_inmet = manager.Set()
    stations_websirenes = manager.Set()
    stations_alertario = manager.Set()
    dataset_era5_year_month = manager.Dict()

    def __init__(
        self,
        websirenes_square: WebSirenesSquare,
        inmet_square: INMETSquare,
        alertario_square: AlertarioSquare,
    ):
        # folder_file = "features_era5_only_2011_2024"
        # folder_file = "features_websirenes_only_2011-2024"
        # folder_file = "features_inmet_only_2011-01_2024-10
        # folder_file = "features_alertario_only_2011-01_2024-10"

        # folder_file = "features_websirenes+inmet_2011_2024"
        # folder_file = "features_inmet+alertario_2011-2024"
        # folder_file = "features_websirenes+alertario_2011-2024"
        # folder_file = "features_websirenes+inmet+alertario_2011_2024"
        folder_file = "features"
        self.features_path = Path(__file__).parent / folder_file
        if not self.features_path.exists():
            self.features_path.mkdir()

        self.era5_single_levels_path = (
            Path(__file__).parent.parent.parent / "data/reanalysis/ERA5-single-levels"
        )
        self.era5_pressure_levels_path = (
            Path(__file__).parent.parent.parent / "data/reanalysis/ERA5-pressure-levels"
        )

        self.websirenes_square = websirenes_square
        self.inmet_square = inmet_square
        self.alertario_square = alertario_square

        lats, lons = self._get_grid_lats_lons()

        self.sorted_latitudes_ascending = np.sort(lats)
        self.sorted_longitudes_ascending = np.sort(lons)

        self.features_tuple = {
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

        log.debug("SpatioTemporalFeatures initialized")
        log.debug(
            f"Grid: {len(self.sorted_latitudes_ascending)}x{len(self.sorted_longitudes_ascending)}"
        )
        log.debug(f"sorted_latitudes_ascending: {self.sorted_latitudes_ascending}")
        log.debug(f"sorted_longitudes_ascending: {self.sorted_longitudes_ascending}")
        log.info(
            f"Spatial resolution: {self.sorted_latitudes_ascending[1] - self.sorted_latitudes_ascending[0]:.2f} degrees"
        )
        # TODO Log temporal resolution, easy to sutract [1] from [0] in timestamps of xr.Dataset

    def _write_features(self, features: npt.NDArray[np.float64], timestamp: pd.Timestamp):
        features_filename = self.features_path / f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
        np.save(features_filename, features)

    def _get_grid_lats_lons(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        ds = self._get_era5_single_levels_dataset(2018, 1)
        lats = ds.coords["latitude"].values
        lons = ds.coords["longitude"].values
        return lats, lons

    def _get_era5_single_levels_dataset(self, year: int, month: int) -> xr.Dataset:
        if (
            self.dataset_era5_year_month.get(("year", "month", "single_levels")) == (year, month)
            and self.dataset_era5_year_month.get("single_levels") is not None
        ):
            return self.dataset_era5_year_month.get("single_levels")

        era5_year_month_path = (
            self.era5_single_levels_path / "monthly_data" / f"RJ_{year}_{month}.nc"
        )

        if not os.path.exists(era5_year_month_path):
            raise FileNotFoundError(f"File {era5_year_month_path} not found")

        if self.dataset_era5_year_month.get("single_levels") is not None:
            self.dataset_era5_year_month.get("single_levels").close()

        ds = xr.open_dataset(era5_year_month_path)
        ds = ds[["tp"]]
        self.dataset_era5_year_month.update({("year", "month", "single_levels"): (year, month)})
        self.dataset_era5_year_month.update({"single_levels": ds})
        return ds

    def _get_era5_pressure_levels_dataset(self, year: int, month: int) -> xr.Dataset:
        if (
            self.dataset_era5_year_month.get(("year", "month", "pressure_levels")) == (year, month)
            and self.dataset_era5_year_month.get("pressure_levels") is not None
        ):
            return self.dataset_era5_year_month.get("pressure_levels")

        era5_year_month_path = (
            self.era5_pressure_levels_path / "monthly_data" / f"RJ_{year}_{month}.nc"
        )

        if not era5_year_month_path.exists():
            raise FileNotFoundError(f"File {era5_year_month_path} not found")

        if self.dataset_era5_year_month.get("pressure_levels") is not None:
            self.dataset_era5_year_month.get("pressure_levels").close()

        ds = xr.open_dataset(era5_year_month_path)
        ds = ds[["r", "t", "u", "v", "w"]]
        self.dataset_era5_year_month.update({("year", "month", "pressure_levels"): (year, month)})
        self.dataset_era5_year_month.update({"pressure_levels": ds})
        return ds

    def _get_precipitation_in_square(
        self,
        square: Square,
        timestamp: pd.Timestamp,
        ds: xr.Dataset,
        keys: list[tuple],
        lat_index: int,
        lon_index: int,
    ) -> float:
        if settings.only_ERA5:
            return self.websirenes_square.get_era5_single_levels_precipitation_in_square(square, ds)
        websirenes_keys = self.websirenes_square.get_keys_in_square(
            square, self.stations_websirenes
        )
        inmet_keys = self.inmet_square.get_keys_in_square(square, self.stations_inmet)
        alertario_keys = self.alertario_square.get_keys_in_square(square, self.stations_alertario)

        # with ThreadPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(
        #             self.websirenes_square.get_keys_in_square, square, self.stations_websirenes
        #         ),
        #         executor.submit(self.inmet_square.get_keys_in_square, square, self.stations_inmet),
        #         executor.submit(
        #             self.alertario_square.get_keys_in_square, square, self.stations_alertario
        #         ),
        #     ]
        #     websirenes_keys, inmet_keys, alertario_keys = [f.result() for f in futures]

        if inmet_keys or websirenes_keys or alertario_keys:
            keys.append((lat_index, lon_index))

        # tp_sirenes = self.websirenes_square.get_precipitation_in_square(
        #     square, websirenes_keys, timestamp, ds
        # )
        # tp_inmet = self.inmet_square.get_precipitation_in_square(square, inmet_keys, timestamp, ds)
        # tp_alertario = self.alertario_square.get_precipitation_in_square(
        #     square, alertario_keys, timestamp, ds
        # )

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.websirenes_square.get_precipitation_in_square,
                    square,
                    websirenes_keys,
                    timestamp,
                    ds,
                ),
                executor.submit(
                    self.inmet_square.get_precipitation_in_square, square, inmet_keys, timestamp, ds
                ),
                executor.submit(
                    self.alertario_square.get_precipitation_in_square,
                    square,
                    alertario_keys,
                    timestamp,
                    ds,
                ),
            ]
            tp_sirenes, tp_inmet, tp_alertario = [f.result() for f in futures]

        return max(tp_sirenes, tp_inmet, tp_alertario)

    def _process_grid(
        self,
        features: npt.NDArray[np.float64],
        ds_single_levels: xr.Dataset,
        ds_pressure_levels: xr.Dataset,
        timestamp: pd.Timestamp,
    ):
        # from line_profiler import LineProfiler

        # profiler = LineProfiler()
        # profiler.add_function(spatio_temporal_features.build_timestamps_hourly)
        # profiler.add_function(get_square)
        # profiler.add_function(self._get_precipitation_in_square)
        # profiler.add_function(self.websirenes_square.get_relative_humidity_in_square)
        # profiler.add_function(self.websirenes_square.get_temperature_in_square)
        # profiler.add_function(self.websirenes_square.get_u_component_in_square)
        # profiler.add_function(self.websirenes_square.get_v_component_in_square)
        # profiler.add_function(self.websirenes_square.get_w_component_in_square)
        # profiler.enable()

        top_down_lats = self.sorted_latitudes_ascending[::-1]
        left_right_lons = self.sorted_longitudes_ascending

        processed = 0
        keys = []
        # O(len(top_down_lats) * len(left_right_lons))
        # O(top_down_lats * left_right_lons * logn)

        print(top_down_lats)
        # let's save top_down_lats and left_right_lons into numpy files:
        np.save(self.features_path / "top_down_lats.npy", top_down_lats)
        np.save(self.features_path / "left_right_lons.npy", left_right_lons)
        log.success("top_down_lats and left_right_lons saved")

        for i, lat in enumerate(top_down_lats):
            for j, lon in enumerate(left_right_lons):
                # O(logn), uses bisect
                square = get_square(
                    lat, lon, self.sorted_latitudes_ascending, self.sorted_longitudes_ascending
                )

                if square is None:
                    continue

                tp = self._get_precipitation_in_square(
                    square, timestamp, ds_single_levels, keys, i, j
                )

                corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
                coords = [
                    square.top_left,
                    square.bottom_left,
                    square.bottom_right,
                    square.top_right,
                ]
                corner_data = {
                    corner: ds_pressure_levels.sel(latitude=lat, longitude=lon)
                    for corner, (lat, lon) in zip(corners, coords)
                }

                # O(4) ~ O(1)
                r1000, r700, r200 = self.websirenes_square.get_relative_humidity_in_square(
                    square, ds_pressure_levels, corner_data
                )

                # O(4) ~ O(1), all these below are the O(square)
                t1000, t700, t200 = self.websirenes_square.get_temperature_in_square(
                    square, ds_pressure_levels, corner_data
                )
                u1000, u700, u200 = self.websirenes_square.get_u_component_in_square(
                    square, ds_pressure_levels, corner_data
                )

                v1000, v700, v200 = self.websirenes_square.get_v_component_in_square(
                    square, ds_pressure_levels, corner_data
                )
                w1000, w700, w200 = self.websirenes_square.get_w_component_in_square(
                    square, ds_pressure_levels, corner_data
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
                processed += 1

        self.stations_cells.update(keys)

        total_squares = (len(top_down_lats) - 1) * (len(left_right_lons) - 1)
        assert processed == total_squares, "Not all squares processed"

        bottom_row_pressure_levels = ds_pressure_levels.sel(latitude=min(top_down_lats))
        bottom_row_single_levels = ds_single_levels.sel(latitude=min(top_down_lats))

        right_column_pressure_levels = ds_pressure_levels.sel(longitude=max(left_right_lons))
        right_column_single_levels = ds_single_levels.sel(longitude=max(left_right_lons))

        for j, lon in enumerate(left_right_lons):
            features[-1, j] = [
                bottom_row_single_levels["tp"].values[j] * 1000,
                bottom_row_pressure_levels["r"].sel(pressure_level=200).values[j],
                bottom_row_pressure_levels["r"].sel(pressure_level=700).values[j],
                bottom_row_pressure_levels["r"].sel(pressure_level=1000).values[j],
                bottom_row_pressure_levels["t"].sel(pressure_level=200).values[j],
                bottom_row_pressure_levels["t"].sel(pressure_level=700).values[j],
                bottom_row_pressure_levels["t"].sel(pressure_level=1000).values[j],
                bottom_row_pressure_levels["u"].sel(pressure_level=200).values[j],
                bottom_row_pressure_levels["u"].sel(pressure_level=700).values[j],
                bottom_row_pressure_levels["u"].sel(pressure_level=1000).values[j],
                bottom_row_pressure_levels["v"].sel(pressure_level=200).values[j],
                bottom_row_pressure_levels["v"].sel(pressure_level=700).values[j],
                bottom_row_pressure_levels["v"].sel(pressure_level=1000).values[j],
                np.sqrt(
                    bottom_row_pressure_levels["u"].sel(pressure_level=200).values[j] ** 2
                    + bottom_row_pressure_levels["v"].sel(pressure_level=200).values[j] ** 2
                ),
                np.sqrt(
                    bottom_row_pressure_levels["u"].sel(pressure_level=700).values[j] ** 2
                    + bottom_row_pressure_levels["v"].sel(pressure_level=700).values[j] ** 2
                ),
                np.sqrt(
                    bottom_row_pressure_levels["u"].sel(pressure_level=1000).values[j] ** 2
                    + bottom_row_pressure_levels["v"].sel(pressure_level=1000).values[j] ** 2
                ),
                bottom_row_pressure_levels["w"].sel(pressure_level=200).values[j],
                bottom_row_pressure_levels["w"].sel(pressure_level=700).values[j],
                bottom_row_pressure_levels["w"].sel(pressure_level=1000).values[j],
            ]
            processed += 1

        for i, lat in enumerate(top_down_lats):
            features[i, -1] = [
                right_column_single_levels["tp"].values[i] * 1000,
                right_column_pressure_levels["r"].sel(pressure_level=200).values[i],
                right_column_pressure_levels["r"].sel(pressure_level=700).values[i],
                right_column_pressure_levels["r"].sel(pressure_level=1000).values[i],
                right_column_pressure_levels["t"].sel(pressure_level=200).values[i],
                right_column_pressure_levels["t"].sel(pressure_level=700).values[i],
                right_column_pressure_levels["t"].sel(pressure_level=1000).values[i],
                right_column_pressure_levels["u"].sel(pressure_level=200).values[i],
                right_column_pressure_levels["u"].sel(pressure_level=700).values[i],
                right_column_pressure_levels["u"].sel(pressure_level=1000).values[i],
                right_column_pressure_levels["v"].sel(pressure_level=200).values[i],
                right_column_pressure_levels["v"].sel(pressure_level=700).values[i],
                right_column_pressure_levels["v"].sel(pressure_level=1000).values[i],
                np.sqrt(
                    right_column_pressure_levels["u"].sel(pressure_level=200).values[i] ** 2
                    + right_column_pressure_levels["v"].sel(pressure_level=200).values[i] ** 2
                ),
                np.sqrt(
                    right_column_pressure_levels["u"].sel(pressure_level=700).values[i] ** 2
                    + right_column_pressure_levels["v"].sel(pressure_level=700).values[i] ** 2
                ),
                np.sqrt(
                    right_column_pressure_levels["u"].sel(pressure_level=1000).values[i] ** 2
                    + right_column_pressure_levels["v"].sel(pressure_level=1000).values[i] ** 2
                ),
                right_column_pressure_levels["w"].sel(pressure_level=200).values[i],
                right_column_pressure_levels["w"].sel(pressure_level=700).values[i],
                right_column_pressure_levels["w"].sel(pressure_level=1000).values[i],
            ]
            processed += 1
        # the corner cell is processed twice, is the common point between the last row and the last column
        processed -= 1
        total_squares = len(top_down_lats) * len(left_right_lons)
        assert processed == total_squares, (
            "Not all cells processed failed to include last row and last column"
        )
        assert not np.any(np.isnan(features)), "Features should not have nan values"

        # profiler.disable()

        # with open("profile_results.txt", "w") as f:
        # profiler.print_stats(stream=f)

    def _process_timestamp(self, timestamp: pd.Timestamp):
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour

        time = f"{year}-{month}-{day}T{hour}:00:00.000000000"

        ds_single_levels_month = self._get_era5_single_levels_dataset(year, month)
        ds_pressure_levels_month = self._get_era5_pressure_levels_dataset(year, month)

        ds_single_levels_time = ds_single_levels_month.sel(valid_time=time, method="nearest")
        ds_pressure_levels_time = ds_pressure_levels_month.sel(valid_time=time, method="nearest")

        features = np.zeros(
            (
                self.sorted_latitudes_ascending.size,
                self.sorted_longitudes_ascending.size,
                len(self.features_tuple),
            ),
            dtype=np.float64,
        )

        self._process_grid(features, ds_single_levels_time, ds_pressure_levels_time, timestamp)
        self._write_features(features, timestamp)
        ds_single_levels_time.close()
        ds_pressure_levels_time.close()

    def build_timestamps_hourly(
        self,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        ignored_months: list[int],
        use_cache: bool = True,
    ):
        minimum_date = start_date
        maximum_date = end_date

        timestamps = pd.date_range(start=minimum_date, end=maximum_date, freq="h")

        log.info(f"Building websirenes target from {timestamps[0]} to {timestamps[-1]}")
        start_time = time.time()
        ONE_MINUTE = 60 * 1
        all_cached = True

        # with ProcessPoolExecutor() as executor:
        #     futures = []

        #     for timestamp in timestamps:
        #         if (
        #             use_cache
        #             and self.features_path.joinpath(
        #                 f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
        #             ).exists()
        #         ):
        #             continue

        #         if timestamp.month in ignored_months:
        #             continue

        #         all_cached = False
        #         futures.append(executor.submit(self._process_timestamp, timestamp))
        #     log.info(f"Tasks submitted - {len(futures)}")

        #     with tqdm(
        #         total=len(timestamps),
        #         desc="Processing timestamps",
        #         file=TqdmLogger(log),
        #         dynamic_ncols=True,
        #         mininterval=ONE_MINUTE,
        #     ) as pbar:
        #         for future in as_completed(futures):
        #             try:
        #                 future.result()
        #                 pbar.update()
        #             except Exception as e:
        #                 log.error(f"Error processing timestamp: {repr(e)}")
        #                 raise SystemExit(e)

        #     self.found_stations = self.stations_websirenes._getvalue()
        #     self.found_stations_inmet = self.stations_inmet._getvalue()
        #     self.found_stations_alertario = self.stations_alertario._getvalue()
        #     self.stations_cells = self.stations_cells._getvalue()
        #     self.manager.shutdown()

        # end_time = time.time()
        # log.info(f"Target built in {end_time - start_time:.2f} seconds - parallel")

        start_time = time.time()
        for i in tqdm(
            range(len(timestamps)),
            desc="Processing timestamps",
            file=TqdmLogger(log),
            dynamic_ncols=True,
            mininterval=ONE_MINUTE,
        ):
            if self.features_path.joinpath(
                f"{timestamps[i].strftime('%Y_%m_%d_%H')}_features.npy"
            ).exists():
                continue

            if timestamps[i].month in ignored_months:
                continue

            all_cached = False
            self._process_timestamp(timestamps[i])
        self.found_stations = self.stations_websirenes._getvalue()
        self.found_stations_inmet = self.stations_inmet._getvalue()
        self.found_stations_alertario = self.stations_alertario._getvalue()
        self.stations_cells = self.stations_cells._getvalue()
        self.manager.shutdown()
        end_time = time.time()
        log.info(f"Target built in {end_time - start_time:.2f} seconds - sequential")

        validated_total_timestamps = self.validate_timestamps(
            minimum_date, maximum_date, ignored_months
        )

        log.success(
            f"Websirenes features hourly built successfully in {self.features_path} - {validated_total_timestamps} files"
        )

        assert (
            settings.only_ERA5
            or all_cached
            or len(self.found_stations)
            == len(
                list(self.websirenes_square.websirenes_keys.websirenes_keys_path.glob("*.parquet"))
            )
        ), "Expected all websirenes stations to be found and processed"

        assert (
            settings.only_ERA5
            or all_cached
            or len(list(self.inmet_square.inmet_keys.inmet_keys_path.glob("*.parquet")))
        ), "Expected all inmet stations to be found and processed"

        assert (
            settings.only_ERA5
            or all_cached
            or len(list(self.alertario_square.alertario_keys.alertario_keys_path.glob("*.parquet")))
        ), "Expected all alertario stations to be found and processed"

        if not all_cached and not settings.only_ERA5:
            log.success(f"""
                All stations processed:
                Websirenes: {len(self.found_stations)} files
                INMET: {len(self.found_stations_inmet)} files
                Alertario: {len(self.found_stations_alertario)} files
                Total cells with station: {len(self.stations_cells)}
            """)

        if len(self.stations_cells) > 0:
            np.save(self.features_path / "stations_cells.npy", list(self.stations_cells))
            log.success(
                f"set {self.stations_cells} file created in {self.features_path / 'stations_cells.npy'}"
            )
        log.success(f"all_cached FELIPE: {all_cached}")

    def validate_timestamps(
        self, min_timestamp: pd.Timestamp, max_timestamp: pd.Timestamp, ignored_months: list[int]
    ) -> int:
        timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq="h")
        not_found = []
        total_timestamps = 0
        total_files = 0

        total_files_with_nan = 0
        files_with_nan = []

        for timestamp in timestamps:
            if timestamp.month in ignored_months:
                continue

            total_timestamps += 1
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            file = self.features_path / f"{year:04}_{month:02}_{day:02}_{hour:02}_features.npy"

            if not Path(file).exists():
                not_found.append(timestamp)
                continue

            features = np.load(file)

            if np.any(np.isnan(features)):
                total_files_with_nan += 1
                files_with_nan.append(file)

            assert features.shape[0] == len(self.sorted_latitudes_ascending), (
                f"shape[0] should be {len(self.sorted_latitudes_ascending)} but is {features.shape[0]}"
            )
            assert features.shape[1] == len(self.sorted_longitudes_ascending), (
                f"shape[1] should be {len(self.sorted_longitudes_ascending)} but is {features.shape[1]}"
            )
            assert features.shape[2] == len(self.features_tuple), (
                f"shape[2] should be {len(self.features_tuple)} but is {features.shape[2]}"
            )

            assert np.all(np.any(features != 0, axis=(1, 2))), (
                f"Should not have one row with all values as zero for {file}"
            )

            total_files += 1

        if total_files_with_nan > 0:
            log.error(f"Total files with nan values: {total_files_with_nan}")
            log.error(f"Files with nan values: {files_with_nan}")
            log.error(f"Percentage of files with nan values: {total_files_with_nan / total_files}")

            for file in files_with_nan:
                os.remove(file)
            log.error("Files with nan values removed")
            exit(1)

        if not_found:
            log.error(f"Missing timestamps: {not_found}")
            exit(1)

        assert total_files == total_timestamps, (
            "Mismatch between timestamps and files (ignoring specific months)"
        )

        log.success(
            f"""All timestamps found in target directory:
            From={min_timestamp}
            To={max_timestamp}
            Ignoring months={ignored_months}
            Total timestamps={total_timestamps}
            Shape: {features.shape}
            All rows have at least one non-zero value: {np.all(np.any(features != 0, axis=(1, 2)))}
            """
        )
        return total_timestamps


def plot_u_v_200_700_1000_levels(
    lon: npt.NDArray[np.float32],
    lat: npt.NDArray[np.float32],
    u_200: npt.NDArray[np.float32],
    v_200: npt.NDArray[np.float32],
    u_700: npt.NDArray[np.float32],
    v_700: npt.NDArray[np.float32],
    u_1000: npt.NDArray[np.float32],
    v_1000: npt.NDArray[np.float32],
):
    plt.close("all")

    fig_3d = plt.figure(figsize=(16, 10))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon, lat)
    frame = 19

    level_200 = np.full_like(u_200[frame], 200)
    level_700 = np.full_like(u_700[frame], 700)
    level_1000 = np.full_like(u_1000[frame], 1000)

    ax_3d.invert_zaxis()

    quiver_200 = ax_3d.quiver(
        lon_meshgrid,
        lat_meshgrid,
        level_200,
        u_200[frame],
        v_200[frame],
        np.zeros_like(u_200[frame]),
        length=0.3,
        normalize=True,
        color="red",
        label="200 hPa",
    )

    quiver_700 = ax_3d.quiver(
        lon_meshgrid,
        lat_meshgrid,
        level_700,
        u_700[frame],
        v_700[frame],
        np.zeros_like(u_700[frame]),
        length=0.3,
        normalize=True,
        color="blue",
        label="700 hPa",
    )

    quiver_1000 = ax_3d.quiver(
        lon_meshgrid,
        lat_meshgrid,
        level_1000,
        u_1000[frame],
        v_1000[frame],
        np.zeros_like(u_1000[frame]),
        length=0.3,
        normalize=True,
        color="green",
        label="1000 hPa",
    )

    ax_3d.set_xlabel("Longitude")
    ax_3d.set_ylabel("Latitude")
    ax_3d.set_zlabel("Pressure Level (hPa)")

    ax_3d.legend(loc="upper left")

    ax_3d.set_title("3D Visualization of Wind at 200, 700, and 1000 hPa")

    ax_3d.view_init(elev=10, azim=120)
    FRAMES_DIR = Path("./features-frames")
    plt.savefig(f"{FRAMES_DIR}/u_v_200_700_1000_levels.png", dpi=300, bbox_inches="tight")
    log.success(f"Saved {FRAMES_DIR}/u_v_200_700_1000_levels.png")
    plt.show()


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.WebsirenesTarget
    # https://g1.globo.com/rj/rio-de-janeiro/noticia/2022/10/31/rio-entra-em-estagio-de-mobilizacao-por-previsao-de-chuva.ghtml
    # https://www.poder360.com.br/brasil/rio-de-janeiro-tem-mes-de-janeiro-mais-chuvoso-em-27-anos/#:~:text=Em%202024%2C%20houve%20registro%20de,pluviom%C3%A9trica%20de%20348%2C9%20mil%C3%ADmetros
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .AlertarioCoords import get_alertario_coords
    from .AlertarioKeys import AlertarioKeys
    from .AlertarioParser import AlertarioParser
    from .INMETCoords import get_inmet_coords
    from .INMETKeys import INMETKeys
    from .INMETParser import INMETParser
    from .WebSirenesCoords import get_websirenes_coords
    from .WebSirenesKeys import WebSirenesKeys
    from .WebSirenesParser import WebSirenesParser

    def create_map():
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        return fig, ax

    def get_features(timestamps: list[pd.Timestamp]):
        features_list = []
        for timestamp in timestamps:
            features_path = (
                Path(__file__).parent
                / "features"
                / f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
            )
            if not features_path.exists():
                spatio_temporal_features._process_timestamp(timestamp)
            features_list.append(np.load(features_path))
        return np.stack(features_list, axis=0)

    FRAMES_DIR = Path("./features-frames-delete-it")
    FRAMES_DIR.mkdir(exist_ok=True)
    # TIMESTAMP = "2022-10-31T18:00:00"
    # TIMESTAMP = "2024-01-13T18:00:00"
    TIMESTAMP = "2024-01-13T18:00:00"
    INTERVAL_MS = 250

    spatio_temporal_features = SpatioTemporalFeatures(
        WebSirenesSquare(WebSirenesKeys(WebSirenesParser(), get_websirenes_coords())),
        INMETSquare(INMETKeys(INMETParser(), get_inmet_coords())),
        AlertarioSquare(AlertarioKeys(AlertarioParser(), get_alertario_coords())),
    )

    spatio_temporal_features.websirenes_square.websirenes_keys.initialize_keys()
    spatio_temporal_features.inmet_square.inmet_keys.initialize_keys()
    spatio_temporal_features.alertario_square.alertario_keys.initialize_keys()

    timestamp = pd.Timestamp(TIMESTAMP)
    features_path = (
        Path(__file__).parent / "features" / f"{timestamp.strftime('%Y_%m_%d_%H')}_features.npy"
    )

    if not features_path.exists():
        spatio_temporal_features._process_timestamp(timestamp)

    features = np.load(features_path)
    precipitation = features[:, :, 0]
    log.info(f"precipitation shape (fixed hour {timestamp}): {precipitation.shape}")

    # plt.figure(figsize=(12, 6))
    # sns.heatmap(precipitation, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
    # plt.title(f"Heatmap of tp values by Latitude and Longitude at {timestamp}")

    # plt.savefig(f"{FRAMES_DIR}/heatmap.png", dpi=300, bbox_inches="tight")
    log.success(f"Saved heatmap as {FRAMES_DIR}/heatmap.png")
    # plt.show()

    log.info("MAKING A MAP WITH THE HEATMAP")
    # ds = xr.open_dataset(
    #     f"./data/reanalysis/ERA5-single-levels/monthly_data/RJ_{timestamp.year}_{timestamp.month}.nc"
    # )
    ds2 = xr.open_dataset(
        f"./data/reanalysis/ERA5-pressure-levels/monthly_data/RJ_{timestamp.year}_{timestamp.month}.nc"
    )
    # ds = ds.sel(valid_time=timestamp)
    ds2 = ds2.sel(valid_time=timestamp)

    # precipitation = ds.tp.values * 1000
    u = ds2.sel(pressure_level=1000).u.values
    v = ds2.sel(pressure_level=1000).v.values

    fig, ax = create_map()
    print("felipe")
    print(spatio_temporal_features.sorted_longitudes_ascending.shape)
    print(spatio_temporal_features.sorted_latitudes_ascending[::-1].shape)
    print(precipitation.shape)
    heatmap = ax.pcolormesh(
        spatio_temporal_features.sorted_longitudes_ascending,
        spatio_temporal_features.sorted_latitudes_ascending[::-1],
        precipitation,
        cmap="coolwarm",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )
    quiver = ax.quiver(
        spatio_temporal_features.sorted_longitudes_ascending,
        spatio_temporal_features.sorted_latitudes_ascending[::-1],
        u,
        v,
        np.sqrt(u**2 + v**2),
        scale=40,
        cmap="cool",
        transform=ccrs.PlateCarree(),
    )
    for i in range(precipitation.shape[0]):
        for j in range(precipitation.shape[1]):
            lon = spatio_temporal_features.sorted_longitudes_ascending[j]
            lat = spatio_temporal_features.sorted_latitudes_ascending[::-1][i]
            value = precipitation[i, j]
            ax.text(lon, lat, f"{value:.2f}", ha="center", va="center", fontsize=10, color="black")
    cbar = plt.colorbar(
        heatmap,
        ax=ax,
        label="Total Precipitation (mm)",
        orientation="vertical",
        pad=0.01,
        aspect=50,
    )
    # plt.colorbar(
    #     quiver, ax=ax, label="Wind Speed (m/s)", orientation="horizontal", fraction=0.046, pad=0.04
    # )
    cbar_ax = fig.add_axes([0.35, 0.07, 0.40, 0.01])
    cbar2 = fig.colorbar(
        quiver, cax=cbar_ax, label="Wind Speed (m/s)", orientation="horizontal", fraction=0.01
    )
    # the quiver colorbar is taking too much space, it's too thick, making it thin:

    ax.set(xlabel="Longitude", ylabel="Latitude")
    # ax.set_title(f"Heatmap of tp values by Latitude and Longitude at {timestamp}")
    # ax.set_title(f"{timestamp} - Tp and wind heatmap (1000 hPa)", loc="left", x=0.0)
    ax.set_title("6:00 PM", fontsize=14)

    cbar2.ax.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=14)
    cbar2.ax.set_xlabel(cbar2.ax.get_xlabel(), fontsize=14)

    plt.savefig(f"{FRAMES_DIR}/heatmap_map_{TIMESTAMP}.png", dpi=300, bbox_inches="tight")
    log.success(f"Saved heatmap with map as {FRAMES_DIR}/heatmap_map.png")
    # plt.show()

    # ds.close()
    ds2.close()
    plt.close("all")
    plt.clf()
    exit(0)

    lats = spatio_temporal_features.sorted_latitudes_ascending[::-1]
    lons = spatio_temporal_features.sorted_longitudes_ascending
    log.info(f"""
        Lats: {lats}
        Lons: {lons}
    """)

    timestamps = pd.date_range(
        start=timestamp.replace(hour=0, minute=0, second=0),
        end=timestamp.replace(hour=23, minute=0, second=0),
        freq="h",
    )
    features = get_features(timestamps)

    fig, ax = create_map()

    u = features[:, :, :, 9]
    v = features[:, :, :, 12]
    # u = features[:, :, :, 8]
    # v = features[:, :, :, 11]
    spd = np.sqrt(u**2 + v**2)

    log.info(f"""
        Shapes:
        u: {u.shape}
        v: {v.shape}
        spd: {spd.shape}
        lon meshgrid: {lons.shape}
        lat meshgrid: {lats.shape}
    """)

    # assert u.shape[1:] == lon.shape, "Mismatch between u/v and lon/lat shapes"
    # assert v.shape[1:] == lat.shape, "Mismatch between v and lat shapes"

    quiver = ax.quiver(
        lons,
        lats,
        u[0],
        v[0],
        spd[0],
        scale=40,
        cmap="cool",
        transform=ccrs.PlateCarree(),
    )

    def _update_fn(frame):
        quiver.set_UVC(u[frame], v[frame], spd[frame])
        ax.set_title(f"Wind on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        return (quiver,)

    anim = animation.FuncAnimation(
        fig, _update_fn, frames=len(timestamps), interval=INTERVAL_MS, blit=True
    )
    plt.colorbar(quiver, ax=ax, label="Wind Speed (m/s)", orientation="vertical")
    ax.set(xlabel="Longitude", ylabel="Latitude")

    gif_file = "u_v_1000_hpa_wind_non_mesh.gif"
    anim.save(f"{FRAMES_DIR}/{gif_file}", writer="pillow", fps=1000 / INTERVAL_MS)
    # https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
    anim.event_source.stop()
    del anim
    log.success(f"Saved {FRAMES_DIR}/{gif_file}")

    for frame in tqdm(range(len(timestamps)), desc="Saving wind frames"):
        quiver.set_UVC(u[frame], v[frame], spd[frame])
        ax.set_title(f"Wind on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        plt.savefig(f"{FRAMES_DIR}/frame_u_v_{frame:02d}.png", dpi=300, bbox_inches="tight")
    log.success(f"Saved {len(timestamps)} frames as {FRAMES_DIR}/frame_u_v_*.png")

    t = features[:, :, :, 6]

    fig, ax = create_map()

    contour = ax.pcolormesh(
        lons,
        lats,
        t[0],
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
        alpha=0.5,
    )

    def _update_fn(frame):
        contour.set_array(t[frame].flatten())
        ax.set_title(f"Temperature on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        return (contour,)

    anim = animation.FuncAnimation(
        fig, _update_fn, frames=len(timestamps), interval=INTERVAL_MS, blit=True
    )

    plt.colorbar(contour, ax=ax, label="Temperature (K)", orientation="vertical")
    ax.set(xlabel="Longitude", ylabel="Latitude")

    gif_file = "temperature_1000_hpa.gif"
    anim.save(f"{FRAMES_DIR}/{gif_file}", writer="pillow", fps=1000 / INTERVAL_MS)
    anim.event_source.stop()
    del anim
    log.success(f"Saved {FRAMES_DIR}/{gif_file}")

    for frame in tqdm(range(len(timestamps)), desc="Saving temp frames"):
        contour.set_array(t[frame].flatten())
        ax.set_title(f"Temperature on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        plt.savefig(f"{FRAMES_DIR}/frame_temperature_{frame:02d}.png", dpi=300, bbox_inches="tight")
    log.success(f"Saved {len(timestamps)} frames as {FRAMES_DIR}/frame_temperature_*.png")

    tp = features[:, :, :, 0]
    log.info(f"tp shape: {tp.shape}")

    fig, ax = create_map()

    im = ax.imshow(
        tp[0],
        cmap="coolwarm",
        origin="upper",
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        vmin=tp.min(),
        vmax=tp.max(),
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(im, ax=ax, label="Total Precipitation (mm)", orientation="vertical")
    ax.set(xlabel="Longitude", ylabel="Latitude")

    def _update_fn(frame):
        im.set_data(tp[frame])
        ax.set_title(f"Total Precipitation on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update_fn, frames=len(timestamps), interval=INTERVAL_MS, blit=True
    )
    gif_file = "total_precipitation.gif"
    anim.save(f"{FRAMES_DIR}/{gif_file}", writer="pillow", fps=1000 / INTERVAL_MS)
    anim.event_source.stop()
    del anim
    log.success(f"Saved {FRAMES_DIR}/{gif_file}")

    for frame in tqdm(range(len(timestamps)), desc="Saving tp frames"):
        im.set_data(tp[frame])
        ax.set_title(f"Total Precipitation on {timestamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")
        plt.savefig(
            f"{FRAMES_DIR}/frame_total_precipitation_{frame:02d}.png", dpi=300, bbox_inches="tight"
        )
    log.success(f"Saved {len(timestamps)} frames as {FRAMES_DIR}/frame_total_precipitation_*.png")

    u_200 = features[:, :, :, 7]
    u_700 = features[:, :, :, 8]
    u_1000 = features[:, :, :, 9]
    v_200 = features[:, :, :, 10]
    v_700 = features[:, :, :, 11]
    v_1000 = features[:, :, :, 12]

    # plot_u_v_200_700_1000_levels(lons, lats, u_200, v_200, u_700, v_700, u_1000, v_1000)
