from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .ERA5Square import ERA5Square
from .INMETKeys import INMETKeys
from .Logger import logger
from .square import Square

log = logger.get_logger(__name__)


class INMETSquare(ERA5Square):
    def __init__(self, inmet_keys: INMETKeys) -> None:
        self.inmet_keys = inmet_keys

    def get_keys_in_square(
        self, square: Square, stations_inmet: set, verbose: bool = False
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

    def get_precipitation_in_square(
        self,
        square: Square,
        inmet_keys: list[str],
        timestamp: pd.Timestamp,
        ds_time: xr.Dataset,
    ) -> float:
        if len(inmet_keys) == 0:
            return super().get_era5_single_levels_precipitation_in_square(square, ds_time)
        precipitations: list[float] = []
        for key in inmet_keys:
            df_web = self.inmet_keys.load_key(key)
            df_web_filtered = df_web[df_web.index == timestamp]
            h1 = df_web_filtered["precipitation"]
            if h1.isnull().all():
                h1 = np.array(
                    super().get_era5_single_levels_precipitation_in_square(square, ds_time)
                )
            precipitations.append(h1.item())
        return max(precipitations)


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.INMETSquare
    # https://g1.globo.com/rj/rio-de-janeiro/noticia/2022/10/31/rio-entra-em-estagio-de-mobilizacao-por-previsao-de-chuva.ghtml
    from .INMETCoords import get_inmet_coords
    from .INMETParser import INMETParser
    from .square import get_square

    inmet_square = INMETSquare(INMETKeys(INMETParser(), get_inmet_coords()))
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

    keys = inmet_square.get_keys_in_square(square, set())
    print(f"keys: {keys}")

    precipitation = inmet_square.get_precipitation_in_square(square, keys, timestamp, ds)
    print(f"precipitation: {precipitation}")

    print(f"""
        Precipitation in square:
        {square.top_left} --- {square.top_right}
        | {" " * 20} {precipitation:.2f} mm  {" " * 20} |
        {square.bottom_left} --- {square.bottom_right}
    """)
