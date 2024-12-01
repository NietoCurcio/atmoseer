import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .AlertarioParser import AlertarioParser
from .Logger import logger

log = logger.get_logger(__name__)


class AlertarioKeys:
    def __init__(self, alertario_parser: AlertarioParser) -> None:
        self.alertario_keys_path = Path(__file__).parent / "alertario_keys"
        if not self.alertario_keys_path.exists():
            self.alertario_keys_path.mkdir()
        self.alertario_parser = alertario_parser

    def _write_key(self, df: pd.DataFrame):
        row = df.iloc[0]
        assert isinstance(row["latitude"], str), f"{type(row['latitude'])}"
        assert isinstance(row["longitude"], str), f"{type(row['longitude'])}"
        key = f"{row['latitude']}_{row['longitude']}"
        df.to_parquet(self.alertario_keys_path / f"{key}.parquet")

    def load_key(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(f"{self.alertario_keys_path}/{key}.parquet")

    def build_keys(self, use_cache: bool = True) -> None:
        total_files = len(list(self.alertario_keys_path.glob("*.parquet")))
        if use_cache and total_files > 0:
            log.warning(
                f"Using cached keys, {total_files} files found. To clear cache delete the {self.alertario_keys_path} folder"
            )
            return

        stations = self.alertario_parser.list_rain_gauge_stations()
        unique_names = stations["station"].unique()

        log.info(f"Processing {len(unique_names)} files to build keys")
        minimum_date = pd.Timestamp.max
        maximum_date = pd.Timestamp.min
        for name in tqdm(unique_names):
            station = stations[stations["station"] == name]
            if station["datetime"].min() < minimum_date:
                minimum_date = station["datetime"].min()
            if station["datetime"].max() > maximum_date:
                maximum_date = station["datetime"].max()
            self._write_key(station)

        log.info(f"""
            Minimum date: {minimum_date}
            Maximum date: {maximum_date}
        """)
        log.success(f"Alertario keys built successfully in {self.alertario_keys_path}")

        minimum_maximum_dates_path = (
            self.alertario_keys_path / "minimum_maximum_dates_alertario.json"
        )
        with open(minimum_maximum_dates_path, "w") as f:
            json.dump(
                {
                    "minimum_date": minimum_date.isoformat(),
                    "maximum_date": maximum_date.isoformat(),
                },
                f,
                indent=4,
            )

        log.success(f"Minimum and maximum dates written to {minimum_maximum_dates_path}")


if __name__ == "__main__":
    # python -m src.spatiotemporal_builder.AlertarioKeys
    alertario_keys = AlertarioKeys(AlertarioParser())
    alertario_keys.build_keys()
