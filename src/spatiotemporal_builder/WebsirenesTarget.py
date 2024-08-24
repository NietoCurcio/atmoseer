import os
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .Logger import logger
from .WebSirenesParser import WebSirenesParser, websirenes_parser
from .WebSirenesSquare import WebSirenesSquare, websirenes_square

log = logger.get_logger(__name__)


def get_grid_lats_lons() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    era5land_path = Path(__file__).parent / "ERA5Land/monthly_data/RJ_2022_1.nc"
    ds = xr.open_dataset(era5land_path)
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values
    return lats, lons


class WebsirenesTarget:
    def __init__(
        self,
        websirenes_parser: WebSirenesParser,
        websirenes_square: WebSirenesSquare,
    ):
        self.websirenes_target_path = Path(__file__).parent / "target"
        if not self.websirenes_target_path.exists():
            self.websirenes_target_path.mkdir()

        self.current_ds = None
        self.current_year_month = None
        self.websirenes_parser = websirenes_parser
        self.websirenes_square = websirenes_square

        lats, lons = get_grid_lats_lons()

        self.sorted_latitudes_ascending = np.sort(lats)
        self.sorted_longitudes_ascending = np.sort(lons)

    def _write_target(self, target: np.ndarray, timestamp: pd.Timestamp):
        target_filename = self.websirenes_target_path / f"{timestamp.strftime('%Y_%m_%d_%H')}.npy"
        np.save(target_filename, target)

    def _get_era5land_dataset(self, year: int, month: int) -> xr.Dataset:
        if self.current_year_month == (year, month) and self.current_ds is not None:
            return self.current_ds

        df_era5land_path = Path(__file__).parent / f"ERA5Land/monthly_data/RJ_{year}_{month}.nc"
        if not os.path.exists(df_era5land_path):
            raise FileNotFoundError(f"File {df_era5land_path} not found")

        ds = xr.open_dataset(df_era5land_path)
        if "expver" in list(ds.coords.keys()):
            log.warning(">>>Oops! expver dimension found. Going to remove it.<<<")
            ds_combine = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            ds_combine.load()
            ds = ds_combine
        ds = ds[["u10", "v10", "d2m", "t2m", "sp", "tp"]]
        self.current_ds = ds
        self.current_year_month = (year, month)
        return self.current_ds

    def _process_grid(self, target: np.ndarray, ds_time: xr.Dataset, timestamp: pd.Timestamp):
        top_down_lats = self.sorted_latitudes_ascending[::-1]
        left_right_lons = self.sorted_longitudes_ascending
        for i, lat in enumerate(top_down_lats):
            for j, lon in enumerate(left_right_lons):
                square = self.websirenes_square.get_square(
                    lat, lon, self.sorted_latitudes_ascending, self.sorted_longitudes_ascending
                )

                if square is None:
                    continue

                websirene_keys = self.websirenes_square.get_keys_in_square(square)

                precipitation_in_square = self.websirenes_square.get_precipitation_in_square(
                    square, websirene_keys, timestamp, ds_time
                )

                target[i, j] = precipitation_in_square

    def build_timestamps_hourly(
        self,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        use_cache: bool = True,
    ):
        minimum_date = self.websirenes_parser.minimum_date if start_date is None else start_date
        maximum_date = self.websirenes_parser.maximum_date if end_date is None else end_date

        assert minimum_date != pd.Timestamp.max, "minimum_date should be set during keys building"
        assert maximum_date != pd.Timestamp.min, "maximum_date should be set during keys building"

        if use_cache and len(list(self.websirenes_target_path.glob("*.npy"))) > 0:
            log.info(
                f"Using cached target. To clear cache delete the {self.websirenes_target_path} folder"
            )
            return

        timestamps = pd.date_range(start=minimum_date, end=maximum_date, freq="h")

        log.info(f"Building websirenes target from {timestamps[0]} to {timestamps[-1]}")
        for timestamp in tqdm(timestamps):
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour

            ds = self._get_era5land_dataset(year, month)

            time = f"{year}-{month}-{day}T{hour}:00:00.000000000"
            ds_time = ds.sel(time=time, method="nearest")
            target = np.zeros(
                (self.sorted_latitudes_ascending.size, self.sorted_longitudes_ascending.size),
                dtype=np.float32,
            )
            self._process_grid(target, ds_time, timestamp)
            self._write_target(target, timestamp)

        log.success(f"Websirenes target built successfully in {self.websirenes_target_path}")


websirenes_target = WebsirenesTarget(websirenes_parser, websirenes_square)
