import argparse
from typing import Optional

import pandas as pd

from .Logger import logger
from .WebSirenesKeys import websirenes_keys
from .WebsirenesTarget import websirenes_target

log = logger.get_logger(__name__)


def validate_dates(
    start_date: Optional[str], end_date: Optional[str]
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if not start_date or not end_date:
        return None, None

    s_date = pd.Timestamp(start_date)
    e_date = pd.Timestamp(end_date)

    MIN_START_DATE = pd.Timestamp("2011-04-12 20:30:00")
    MAX_END_DATE = pd.Timestamp("2022-06-02 21:30:00")

    if not (MIN_START_DATE <= s_date <= e_date <= MAX_END_DATE):
        raise ValueError(
            f"start_date and end_date must be between {MIN_START_DATE} and {MAX_END_DATE}"
        )

    log.info(f"Building spatiotemporal data from {s_date} to {e_date}")
    return s_date, e_date


def check_data_requirements():
    ""


def build_target(start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]):
    try:
        websirenes_keys.build_keys(start_date, end_date)
        websirenes_target.build_timestamps_hourly(start_date, end_date)
    except Exception as e:
        log.error(f"Error while building target: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build spatiotemporal data")
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Start date in the format YYYY-MM-DD-H",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date in the format YYYY-MM-DD-H",
    )

    check_data_requirements()

    args = parser.parse_args()

    start_date, end_date = validate_dates(args.start_date, args.end_date)

    build_target(start_date, end_date)
