from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .AlertarioKeys import AlertarioKeys
from .ERA5Square import ERA5Square
from .Logger import logger
from .square import Square

log = logger.get_logger(__name__)


keys_cache = {}


class AlertarioSquare(ERA5Square):
    def __init__(self, alertario_keys: AlertarioKeys) -> None:
        self.alertario_keys = alertario_keys

    def get_keys_in_square(
        self, square: Square, stations_alertario: set, verbose: bool = False
    ) -> list[str]:
        if square in keys_cache:
            return keys_cache[square]

        keys = [x.stem for x in Path(self.alertario_keys.alertario_keys_path).glob("*.parquet")]
        alertario_keys = []
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

            alertario_keys.append(key)

        if len(alertario_keys) > 0:
            stations_alertario.update(alertario_keys)

        keys_cache[square] = alertario_keys

        return alertario_keys

    def get_precipitation_in_square(
        self,
        square: Square,
        alertario_keys: list[str],
        timestamp: pd.Timestamp,
        ds_time: xr.Dataset,
    ) -> float:
        corners = ["top_left", "bottom_left", "bottom_right", "top_right"]
        coords = [square.top_left, square.bottom_left, square.bottom_right, square.top_right]

        corner_data = {
            corner: ds_time.sel(latitude=lat, longitude=lon)
            for corner, (lat, lon) in zip(corners, coords)
        }

        if len(alertario_keys) == 0:
            return super().get_era5_single_levels_precipitation_in_square(
                square, ds_time, corner_data
            )

        h1_era5 = None
        time_upper_bound = timestamp
        time_lower_bound = timestamp - timedelta(minutes=45)

        precipitations_15_min_aggregated: list[float] = []
        for key in alertario_keys:
            df_alertario = self.alertario_keys.load_key(key)

            df_alertario_filtered = df_alertario[
                (df_alertario.datetime >= time_lower_bound)
                & (df_alertario.datetime <= time_upper_bound)
            ]

            m15 = df_alertario_filtered["precipitation"]
            h01 = df_alertario[df_alertario.datetime == time_upper_bound]["h01"]

            if m15.count() < 4:
                # Please see WebSirenesSquare:get_precipitation_in_square for more information
                if h1_era5 is None:
                    h1_era5 = super().get_era5_single_levels_precipitation_in_square(
                        square, ds_time, corner_data
                    )
                m15 = np.array([m15.sum(), h1_era5]).max()

            max_between_m15_and_h01 = np.array([m15.sum().item(), h01.max()]).max()
            # max_between_m15_and_h01 = np.array([m15.sum().item()]).max()
            precipitations_15_min_aggregated.append(max_between_m15_and_h01.item())

        return max(precipitations_15_min_aggregated)


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.AlertarioSquare
    # https://g1.globo.com/rj/rio-de-janeiro/noticia/2022/10/31/rio-entra-em-estagio-de-mobilizacao-por-previsao-de-chuva.ghtml
    from .AlertarioCoords import get_alertario_coords
    from .AlertarioParser import AlertarioParser
    from .square import get_square

    alertario_square = AlertarioSquare(AlertarioKeys(AlertarioParser(), get_alertario_coords()))
    timestamp = pd.Timestamp("2022-10-31T18:00:00")
    year = timestamp.year
    month = timestamp.month
    ds = xr.open_dataset(f"./data/reanalysis/ERA5-single-levels/monthly_data/RJ_{year}_{month}.nc")
    print("xr.Dataset:")
    print(ds)

    ds = ds.sel(valid_time=timestamp)
    lats = ds.latitude.values
    lons = ds.longitude.values
    print(f"Grid: {lats.shape[0]}x{lons.shape[0]}")
    lat = lats[4]
    lon = lons[7]

    square = get_square(lat, lon, sorted(lats), sorted(lons))
    print(f"""
        square:
        top_left={square.top_left}
        bottom_left={square.bottom_left}
        bottom_right={square.bottom_right}
        top_right={square.top_right}
        {square.top_left} --- {square.top_right}
        | {" " * 48} |
        {square.bottom_left} --- {square.bottom_right}
    """)

    keys = alertario_square.get_keys_in_square(square, set())
    print(f"keys: {keys}")

    precipitation = alertario_square.get_precipitation_in_square(square, keys, timestamp, ds)
    print(f"precipitation: {precipitation}")

    print(f"""
        Precipitation in square:
        {square.top_left} --- {square.top_right}
        | {" " * 20} {precipitation:.2f} mm  {" " * 20} |
        {square.bottom_left} --- {square.bottom_right}
    """)
