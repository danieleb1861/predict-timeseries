"""Data preparation and causal feature engineering helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


def make_datetime_from_date_time(date_col, time_col):
    """
    Combine separate date and time columns into a single datetime Series.

    Inputs
    ----------
    date_col : array-like
        Date strings (e.g., "2023-08-01", "01/08/2023").
    time_col : array-like
        Time strings (e.g., "14:30:00", "14:30:00.123", "14:30").

    Returns
    -------
    pandas.Series
        Datetime series (datetime64[ns]) obtained by combining date and time.

    Raises
    ------
    ValueError
        If none of the supported date/time format combinations can parse all rows.
    """

    date_value = pd.Series(date_col).astype(str).str.strip()
    time_value = pd.Series(time_col).astype(str).str.strip()

    dfmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"]
    tfmts = ["%H:%M:%S.%f", "%H:%M:%S", "%H:%M"]

    # try all combinations
    for dfmt in dfmts:
        try:
            dpart = pd.to_datetime(date_value, format=dfmt, errors="coerce")
            if dpart.isna().any():
                raise ValueError("bad date")
        except Exception:
            continue

        ok = False
        for tfmt in tfmts:
            try:
                tpart = pd.to_datetime(time_value, format=tfmt, errors="coerce")
                if tpart.isna().any():
                    raise ValueError("bad time")
                # combine date + time
                t = dpart + (tpart.dt.hour * pd.Timedelta(hours=1) +
                             tpart.dt.minute * pd.Timedelta(minutes=1) +
                             tpart.dt.second * pd.Timedelta(seconds=1) +
                             (tpart.dt.microsecond * pd.Timedelta(microseconds=1)))
                ok = True
                return t
            except Exception:
                continue

        if ok:
            return t

    # fallback: try concatenated string
    try:
        both = date_value + " " + time_value
        t = pd.to_datetime(both, errors="coerce")
        if not t.isna().any():
            return t
    except Exception:
        pass

    # If we reach here, parsing failed
    exD = date_value[~date_value.isna()].iloc[0]
    exT = time_value[~time_value.isna()].iloc[0]
    raise ValueError(f'Could not parse date/time columns. Example: "{exD}" / "{exT}".')


def preprocess_dataset(path, leg_id, colMap, data_types, rating_kW, varsPrev, varsLin, Ts="1s", max_fill_seconds=None):
    '''
    Preprocess a single file: load, parse datetime, resample to fixed cadence.
    '''
    T = pd.read_parquet(path).astype(data_types, copy=False)

    # datetime
    dtime = make_datetime_from_date_time(T[colMap["date"]], T[colMap["time"]])

    # build standardized data_frame
    data_frame = pd.DataFrame({
        "dtime": dtime,
        # PU conversion of active/reactive power
        "act_pow_pu":  (T[colMap["act_pow"]]   / rating_kW).astype(np.float32),
        "react_pow_pu":(T[colMap["react_pow"]] / rating_kW).astype(np.float32),
        "man_port":    T[colMap["man_port"]],
        "man_stbd":    T[colMap["man_stbd"]],
        "spd_port":    T[colMap["spd_port"]].astype(np.float32),
        "spd_stbd":    T[colMap["spd_stbd"]].astype(np.float32),
        "air_temp":    T[colMap["air_temp"]].astype(np.float32),
        "wind_intensity": T[colMap["wind_int"]].astype(np.float32),
        "sea_temp":    T[colMap["sea_temp"]].astype(np.float32),
        "sea_state":   T[colMap["sea_state"]].astype(np.float32),
    })

    print(f"--- Dataset {path} analysis starting... ---")
    print()

    # sort + drop duplicates
    print(f"Dataset {path} time duplicates before preprocessing:", data_frame.duplicated(subset="dtime").sum())
    print()
    data_frame = data_frame.sort_values("dtime").drop_duplicates(subset="dtime")
    # resample within file
    data_frame = data_frame.set_index("dtime")

    # # Don't fill massive holes in data
    # dt = data_frame.index.to_series().diff().dt.total_seconds().dropna()
    # print(dt.describe())
    # print(dt.value_counts().head(20))
    # print("top gaps:", dt.nlargest(10).to_list())
    # print("median:", dt.median(), "mode:", dt.mode().iloc[0], "max:", dt.max())
    # Ts = f"{int(round(dt.mode().iloc[0]))}s"

    if max_fill_seconds is None:
        data_frame_prev = data_frame[varsPrev].resample(Ts).ffill()
        data_frame_lin  = data_frame[varsLin].resample(Ts).interpolate(method="time")
    else:
        lim = int(max_fill_seconds)  # con Ts=1s sono secondi
        data_frame_prev = data_frame[varsPrev].resample(Ts).ffill(limit=lim)
        data_frame_lin  = data_frame[varsLin].resample(Ts).interpolate(method="time", limit=lim, limit_area="inside")

    data_frame_res = data_frame_lin.join(data_frame_prev, how="outer")

    # FORWARDfill remaining NaN (implement limits to not bridge long gaps)
    if max_fill_seconds is None:
        print("Forward fill to be applied: ", data_frame_res.isna().sum().sum())
        data_frame_res = data_frame_res.ffill()
    else:
        print("Forward fill to be applied: ", data_frame_res.isna().sum().sum())
        data_frame_res = data_frame_res.ffill(limit=lim)
    print()

    # BACKWARDfill remaining NaN (implement limits to not bridge long gaps)
    if data_frame_res.isna().sum().sum() > 0:
        if max_fill_seconds is None:
            print("Backward fill to be applied: ", data_frame_res.isna().sum().sum())
            data_frame_res = data_frame_res.bfill()
        else:
            print("Backward fill to be applied: ", data_frame_res.isna().sum().sum())
            data_frame_res = data_frame_res.bfill(limit=lim)
    print()

    # DROP OUTSCALE values (optional)
    data_frame_res = data_frame_res[data_frame_res["act_pow_pu"].between(-8, 8)]   # (PU limits)
    data_frame_res = data_frame_res[data_frame_res["react_pow_pu"].between(-8, 8)] # (PU limits)

    data_frame_res = data_frame_res.reset_index()  # reset index to have the correct index after resampling and dropping values of the data_frame
    # print(data_frame_res.isna())
    print(f"Dataset {path} NaN residual content:")
    print(data_frame_res.isna().sum().sort_values(ascending=False))
    print()
    print("Rows with NaN:", data_frame_res.isna().any(axis=1).sum())
    print()

    # index/metadata for NeuralNets
    data_frame_res["time_leg"] = np.arange(len(data_frame_res), dtype=np.int32)
    data_frame_res["leg_id"]   = np.int16(leg_id)

    # optional but very useful for plotting: seconds from start of this leg
    t0 = data_frame_res["dtime"].iloc[0]
    data_frame_res["t_seconds"] = (data_frame_res["dtime"] - t0).dt.total_seconds().astype(np.float32)
    return data_frame_res


def cached_preprocess_dataset(path, leg_id, colMap, data_types, data_dir, rating_kW, varsPrev, varsLin, Ts="1s", max_fill_seconds=None):
    """
    Load a cached preprocessed leg when source data and preprocessing settings match.

    The cache is stored under data_dir/cache and never modifies raw input data.
    """
    source_path = Path(path)
    cache_dir = Path(data_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_stat = source_path.stat()
    settings = {
        "source": str(source_path.resolve()),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_size": source_stat.st_size,
        "leg_id": int(leg_id),
        "rating_kW": float(rating_kW),
        "Ts": Ts,
        "max_fill_seconds": max_fill_seconds,
        "colMap": colMap,
        "data_types": data_types,
        "varsPrev": varsPrev,
        "varsLin": varsLin,
    }
    settings_text = json.dumps(settings, sort_keys=True, default=str)
    cache_key = hash_pandas_object(pd.Index([settings_text])).values[0]
    cache_path = cache_dir / f"{source_path.stem}_leg{leg_id}_{cache_key:016x}.parquet"

    if cache_path.exists():
        print(f"[cache] preprocessed leg hit: {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"[cache] preprocessed leg miss: {source_path.name}")
    data_frame_res = preprocess_dataset(path, leg_id, colMap, data_types, rating_kW, varsPrev, varsLin, Ts=Ts, max_fill_seconds=max_fill_seconds)
    data_frame_res.to_parquet(cache_path, index=False)
    return data_frame_res


def add_engineered(data_frame):
    """
    Add causal engineered features to the input dataframe for load forecasting.

    Dynamic features are computed per `leg_id` to avoid boundary artefacts.
    Rolling features are backward-looking (`min_periods=1`) and do not use
    future samples.
    """
    data_frame = data_frame.copy()

    # Static algebraic features
    data_frame["lever_diff"] = data_frame["spd_port"] - data_frame["spd_stbd"]
    S = np.sqrt(data_frame["act_pow_pu"]**2 + data_frame["react_pow_pu"]**2)
    data_frame["pfactor"] = np.where(S > 0, data_frame["act_pow_pu"] / S, 0.0).astype(np.float32)

    # IMPORTANT: compute dynamic features per leg_id to avoid boundary artefacts.
    grouped_df = data_frame.groupby("leg_id", sort=False)[["act_pow_pu", "react_pow_pu"]]
    differences = grouped_df.diff().fillna(0.0).astype(np.float32)

    # Share rolling group/index work between active and reactive power.
    rolling_stats = {}
    group_index = None
    for w in [5, 10, 20]:
        rolling = grouped_df.rolling(window=w, min_periods=1)
        rolling_stats[w] = (
            rolling.mean().reset_index(level=0, drop=True).astype(np.float32),
            rolling.std().reset_index(level=0, drop=True).fillna(0.0).astype(np.float32),
        )
        # Every window has the same row order; retain only one index array.
        if group_index is None:
            group_index = rolling_stats[w][0].index
        for values in rolling_stats[w]:
            values.index = group_index

    # Keep the historical column order: active features before reactive features.
    for column, suffix, derivative in (("act_pow_pu", "", "dP"), ("react_pow_pu", "_Q", "dQ")):
        data_frame[derivative] = differences[column]
        for w, (means, stds) in rolling_stats.items():
            data_frame[f"moving_avg_{w}{suffix}"] = means[column]
            data_frame[f"moving_std_{w}{suffix}"] = stds[column]

    return data_frame
