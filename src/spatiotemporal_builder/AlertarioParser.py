from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pandera as pa
import xarray as xr
from sklearn.impute import KNNImputer
from tqdm import tqdm

from .Logger import logger

log = logger.get_logger(__name__)


class AlertarioSchema(pa.DataFrameModel):
    station: str
    datetime: pd.Timestamp
    precipitation: float = pa.Field(nullable=True, ge=0)
    latitude: str
    longitude: str


class AlertarioParser:
    inmet_path = Path(__file__).parent.parent.parent / "data/ws/inmet"
    rain_gauge_path = Path(__file__).parent.parent.parent / "data/ws/rain_gauge"

    minimum_date = pd.Timestamp.max
    maximum_date = pd.Timestamp.min

    def _get_region_of_interest(self) -> dict:
        ds = xr.open_dataset("./data/reanalysis/ERA5-single-levels/monthly_data/RJ_2018_1.nc")
        lats = ds.latitude.values
        lons = ds.longitude.values
        fartest_north = lats.max()
        fartest_south = lats.min()
        fartest_east = lons.max()
        fartest_west = lons.min()
        region_of_interest = {
            "north": fartest_north,
            "south": fartest_south,
            "east": fartest_east,
            "west": fartest_west,
        }
        return region_of_interest

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        chunk_size = 50_000
        chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
        imputed_chunks = []
        log.info(f"Imputing missing values for precipitation in chunks of {chunk_size} rows")
        for chunk in tqdm(chunks):
            chunk_numeric = chunk[["precipitation"]]
            imputer = KNNImputer(n_neighbors=2)
            imputed_chunk = imputer.fit_transform(chunk_numeric)
            chunk.loc[:, "precipitation"] = imputed_chunk
            imputed_chunks.append(chunk)
        df = pd.concat(imputed_chunks)
        del imputed_chunks
        return df

    def _get_rain_gauge_stations_in_region(self) -> pd.DataFrame:
        region_of_interest = self._get_region_of_interest()

        df_region = pd.read_parquet("./src/spatiotemporal_builder/alertario")
        df_region = df_region[
            (df_region["latitude"] <= region_of_interest["north"])
            & (df_region["latitude"] >= region_of_interest["south"])
            & (df_region["longitude"] <= region_of_interest["east"])
            & (df_region["longitude"] >= region_of_interest["west"])
        ]
        df_region[AlertarioSchema.datetime] = pd.to_datetime(
            df_region[AlertarioSchema.datetime]
        ).dt.tz_localize(None)
        # selcting only the columns that are needed
        df_region = df_region[
            [
                AlertarioSchema.station,
                AlertarioSchema.datetime,
                AlertarioSchema.precipitation,
                AlertarioSchema.latitude,
                AlertarioSchema.longitude,
            ]
        ]

        total_values = df_region.shape[0]
        missing_values = df_region[AlertarioSchema.precipitation].isna().sum()
        percentage_missing = (missing_values / total_values) * 100

        if percentage_missing > 10:
            log.error(
                f"Percentage of missing values for precipitation in the region of interest: {percentage_missing:.2f}% ({missing_values}/{total_values}) greater than 10% - exiting"
            )
            exit(1)

        if percentage_missing > 0:
            log.warning(
                f"Percentage of missing values for precipitation in the region of interest: {percentage_missing:.2f}% ({missing_values}/{total_values})"
            )
            df_region = self._impute_missing_values(df_region)

        missing_values = df_region[AlertarioSchema.precipitation].isna().sum()
        assert missing_values == 0, "Missing values after imputation should be zero"

        df_region["latitude"] = df_region["latitude"].apply(lambda x: str(x))
        df_region["longitude"] = df_region["longitude"].apply(lambda x: str(x))
        df_region["station"] = df_region["station"].str.strip()

        AlertarioSchema.validate(df_region)
        log.info(f"Dataframe region describe:\n{df_region.describe()}")

        return df_region

    def list_rain_gauge_stations(self) -> Generator[pd.DataFrame, None, None]:
        df_region = self._get_rain_gauge_stations_in_region()
        return df_region
        # self.unique_names = df_region[AlertarioSchema.station].unique()
        # for name in self.unique_names:
        #     df_station = df_region[df_region[AlertarioSchema.station] == name]
        #     yield df_station


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.AlertarioParser
    alertario_parser = AlertarioParser()
    for df in alertario_parser.list_rain_gauge_stations():
        print(df)
        break
