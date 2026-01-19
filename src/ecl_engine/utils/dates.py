import pandas as pd


def month_end_index(start, periods: int):
    start_me = pd.to_datetime(start) + pd.offsets.MonthEnd(0)
    # use "ME" (MonthEnd) instead of deprecated "M"
    return pd.date_range(start=start_me, periods=periods, freq="ME")
