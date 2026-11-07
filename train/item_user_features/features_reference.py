"""Pandas ports of Chris Deotte's item/user feature notebooks, used by test_features.py.

Each function transcribes one notebook in train/item_user_features/ (cuDF -> pandas,
cupy -> numpy) as literally as possible. Input `df` has `ts` in epoch seconds, as in the
reference (see `to_reference_frame`). Each returns the frame the notebook writes, with
its index (aid or session) reset to a column. Deliberate changes:
- item_12 / item_13 tile the observed number of days instead of a literal 21 / 7.
- Sorts are stable (`kind='stable'`), so the order of tied timestamps is the input order.
"""

import numpy as np
import pandas as pd
import polars as pl


def to_reference_frame(events: pl.DataFrame) -> pd.DataFrame:
    """Polars events (ts Datetime(ms)) -> pandas with ts in epoch seconds, as the notebooks read."""
    return events.with_columns(pl.col("ts").dt.epoch("s").cast(pl.Int64)).to_pandas()


def _f32(df, cols):
    for c in cols:
        df[c] = df[c].astype("float32")


def _i32(df, cols):
    for c in cols:
        df[c] = df[c].astype("int32")


def _out(df: pd.DataFrame, key: str) -> pd.DataFrame:
    df.index.name = key
    return df.reset_index()


def _hour_day(df):
    df["hour"] = df.ts % (60 * 60 * 24)
    df["day"] = df.ts % (60 * 60 * 24 * 7)


# --- item features ---------------------------------------------------------------------

def _item_stats(df, sfx=""):
    """The block shared by item-features-4/10/20/21/22/40/41."""
    item_features = df.groupby("aid").agg(
        {"type": "mean", "aid": "count", "session": "nunique", "hour": "mean", "day": "mean"}
    )
    item_features.columns = [f"{c}{sfx}" for c in ["buy_ratio", "count_item", "count_user", "hour_mean", "day_mean"]]
    item_features2 = df.groupby("aid").agg({"type": "std", "hour": "std", "day": "std"}).fillna(-1)
    item_features2.columns = [f"{c}{sfx}" for c in ["buy_ratio_std", "hour_std", "day_std"]]
    _f32(item_features2, item_features2.columns)
    item_features[f"repeat{sfx}"] = item_features[f"count_item{sfx}"] / item_features[f"count_user{sfx}"]
    _f32(item_features, [f"{c}{sfx}" for c in ["buy_ratio", "hour_mean", "day_mean", "repeat"]])
    _i32(item_features, [f"count_item{sfx}", f"count_user{sfx}"])
    return item_features, item_features2


def _item_transitions(df):
    """The prev/next/repeat block of item-features-10/20/21/22."""
    df = df.sort_values(["session", "ts"], kind="stable")
    df["prev"] = df.groupby("session").aid.diff(1)
    df["next"] = df.groupby("session").aid.diff(-1)
    df["order"] = (df.type == 2).astype("int8")
    df["cart"] = (df.type == 1).astype("int8")
    df["n"] = df.groupby(["session", "aid", "type"]).aid.transform("count")
    tmp = df.loc[df["type"] == 2].drop_duplicates(["session", "aid"]).groupby("aid").n.mean()
    tmp2 = df.loc[df["type"] == 1].drop_duplicates(["session", "aid"]).groupby("aid").n.mean()
    item_features3 = df.groupby("aid").agg({"prev": "nunique", "next": "nunique", "order": "sum", "cart": "sum"})
    item_features3 = item_features3.merge(tmp, left_index=True, right_index=True, how="left")
    item_features3.columns = ["prev3", "next3", "orders3", "carts3", "buy_count3"]
    item_features3 = item_features3.merge(tmp2, left_index=True, right_index=True, how="left")
    item_features3.columns = ["prev3", "next3", "orders3", "carts3", "order_repeat3", "cart_repeat3"]
    item_features3 = item_features3.fillna(-1)
    _i32(item_features3, ["prev3", "next3", "orders3", "carts3"])
    _f32(item_features3, ["order_repeat3", "cart_repeat3"])
    return df, item_features3


def item_4(train):
    df = train.copy()
    _hour_day(df)
    a, b = _item_stats(df)
    return _out(pd.concat([a, b], axis=1), "aid")


def _item_10_like(df):
    df, item_features3 = _item_transitions(df)
    a, b = _item_stats(df, "3")
    return pd.concat([a, b, item_features3], axis=1)


def item_10(test):
    df = test.copy()
    _hour_day(df)
    return _out(_item_10_like(df), "aid")


def _day_std(df, days):
    """item_day_std, user_day_std: per-aid std over the aid x day grid, missing days = 0."""
    aids = df.aid.unique()
    tmp = df.groupby(["aid", "day"]).agg({"aid": "count", "session": "count"})
    tmp.columns = ["item", "user"]
    dff = pd.DataFrame({"aid": np.repeat(aids, len(days)).astype("int32"),
                        "day": np.tile(days, len(aids)).astype("int8")}).set_index(["aid", "day"])
    dff = dff.merge(tmp, left_index=True, right_index=True, how="left").fillna(0).reset_index()
    item_features3 = dff.groupby("aid").agg({"item": "std", "user": "std"}).fillna(-1)
    item_features3.columns = ["item_day_std", "user_day_std"]
    _f32(item_features3, item_features3.columns)
    return item_features3


def item_12(train):
    df = train.copy()
    mn = df.ts.min()
    df["day"] = (df.ts - mn) // (60 * 60 * 24)
    item_features3 = _day_std(df, df.day.unique())
    df["mn"] = df.groupby("session").ts.transform("min")
    df = df.loc[df.mn > mn + 60 * 60 * 24 * 3]
    df["t"] = df.ts - df.mn
    df = df.loc[df.t < 60 * 60 * 24 * 7]
    item_features = df.groupby("aid").agg({"type": "mean", "aid": "count", "session": "nunique"})
    item_features.columns = ["buy_ratio", "count_item", "count_user"]
    item_features2 = df.groupby("aid").agg({"type": "std"}).fillna(-1)
    item_features2.columns = ["buy_ratio_std"]
    _f32(item_features2, ["buy_ratio_std"])
    item_features["repeat"] = item_features.count_item / item_features.count_user
    _f32(item_features, ["buy_ratio", "repeat"])
    _i32(item_features, ["count_item", "count_user"])
    # cudf.concat(axis=1) outer-joins on aid: aids outside the filtered window get nulls
    item_features = pd.concat([item_features, item_features2, item_features3], axis=1)
    item_features.columns = [f"{c}_5" for c in item_features.columns]
    return _out(item_features, "aid")


def item_13(test):
    df = test.copy()
    mn = df.ts.min()
    df["day"] = (df.ts - mn) // (60 * 60 * 24)
    aids = df.aid.unique()
    days = np.sort(df.day.unique())
    item_features3 = _day_std(df, days)
    item_features = df.groupby(["aid", "day"]).agg({"type": "mean", "aid": "count", "session": "nunique"})
    item_features.columns = ["buy_ratio", "count_item", "count_user"]
    _f32(item_features, ["buy_ratio"])
    _i32(item_features, ["count_item", "count_user"])
    dff = pd.DataFrame({"aid": np.repeat(aids, len(days)).astype("int32"),
                        "day": np.tile(days, len(aids)).astype("int8")}).set_index(["aid", "day"])
    dff = dff.merge(item_features, left_index=True, right_index=True, how="left").fillna(-1).reset_index()
    dff = dff.sort_values(["aid", "day"])
    dff2 = pd.DataFrame(index=np.sort(aids))
    for c in ["buy_ratio", "count_item", "count_user"]:
        v = dff[c].values.reshape((-1, len(days)))
        for k in range(len(days)):
            dff2[f"{c}_{k}"] = v[:, k]
    _i32(dff2, [f"{c}_{k}" for c in ["count_item", "count_user"] for k in range(len(days))])
    _f32(dff2, [f"buy_ratio_{k}" for k in range(len(days))])
    item_features = pd.concat([dff2, item_features3], axis=1)
    item_features.columns = [f"{c}_6" for c in item_features.columns]
    return _out(item_features, "aid")


def _rename_last(df, sfx):
    """COLS = [f'{x[:-1]}{sfx}' ...] of item-features-20/21/22."""
    df.columns = [f"{x[:-1]}{sfx}" for x in df.columns]
    return df


def item_20(train):
    df = train.copy()
    mx = df.ts.max()
    df = df.loc[df.ts >= mx - 60 * 60 * 24 * 7].copy()
    _hour_day(df)
    return _out(_rename_last(_item_10_like(df), "7"), "aid")


def item_21(train):
    df = train.copy()
    df["hour"] = df.ts % (60 * 60 * 24)
    df = df.loc[(df.hour >= 60 * 60 * 14) | (df.hour < 60 * 60 * 2)].copy()
    df["day"] = df.ts % (60 * 60 * 24 * 7)
    return _out(_rename_last(_item_10_like(df), "8"), "aid")


def item_22(train):
    df = train.copy()
    df["hour"] = df.ts % (60 * 60 * 24)
    df = df.loc[(df.hour < 60 * 60 * 14) & (df.hour >= 60 * 60 * 2)].copy()
    df["day"] = df.ts % (60 * 60 * 24 * 7)
    return _out(_rename_last(_item_10_like(df), "9"), "aid")


def _item_buys(events, sfx):
    df = events.loc[events["type"] != 0].copy()
    _hour_day(df)
    a, b = _item_stats(df)
    item_features = pd.concat([a, b], axis=1)
    item_features.columns = [f"{x}{sfx}" for x in item_features.columns]
    return _out(item_features, "aid")


def item_40(train):
    return _item_buys(train, "11")


def item_41(test):
    return _item_buys(test, "12")


# --- user features ---------------------------------------------------------------------

def _user_stats(df, sfx):
    """The block shared by user-features-4/10/20/21 (columns ...2, renamed '2' -> sfx)."""
    df["aid2"] = df.aid
    user_features = df.groupby("session").agg(
        {"type": "mean", "aid": "count", "aid2": "nunique", "hour": "mean", "day": "mean"}
    )
    user_features.columns = ["buy_ratio2", "count_item2", "unique_item2", "hour_mean2", "day_mean2"]
    user_features2 = df.groupby("session").agg({"type": "std", "hour": "std", "day": "std"}).fillna(-1)
    user_features2.columns = ["buy_ratio_std2", "hour_std2", "day_std2"]
    _f32(user_features2, user_features2.columns)
    user_features["repeat2"] = user_features.count_item2 / user_features.unique_item2
    _f32(user_features, ["buy_ratio2", "hour_mean2", "day_mean2", "repeat2"])
    _i32(user_features, ["count_item2", "unique_item2"])
    user_features = pd.concat([user_features, user_features2], axis=1)
    user_features.columns = [x.replace("2", sfx) for x in user_features.columns]
    return _out(user_features, "session")


def user_4(test):
    df = test.copy()
    _hour_day(df)
    return _user_stats(df, "2")


def user_7(test):
    df = test.copy()
    mn = df.ts.min()
    user_features = df.groupby("session").agg({"ts": "max"})
    user_features.columns = ["ts_max7"]
    user_features2 = df.groupby("session").agg({"ts": "min"})
    user_features2.columns = ["ts_min7"]
    user_features = pd.concat([user_features, user_features2], axis=1)
    user_features["ts_diff7"] = user_features.ts_max7 - user_features.ts_min7
    for c in ["ts_max7", "ts_min7"]:
        user_features[c] -= mn
    return _out(user_features.astype("int32"), "session")


def user_10(test):
    df = test.loc[test["type"] != 0].copy()
    _hour_day(df)
    return _user_stats(df, "10")


def _last_day(test):
    df = test.sort_values(["session", "ts"], kind="stable").reset_index(drop=True)
    df["mx"] = df.groupby("session").ts.transform("max")
    df["mx"] = df.mx - df.ts
    return df.loc[df.mx < 60 * 60 * 24].drop("mx", axis=1)


def user_20(test):
    df = _last_day(test)
    _hour_day(df)
    return _user_stats(df, "14")


def user_21(test):
    df = _last_day(test)
    df = df.loc[df["type"] != 0].copy()
    _hour_day(df)
    return _user_stats(df, "15")


# notebook -> (function, which events it reads)
FEATURES = {
    "item4": (item_4, "train"),
    "item10": (item_10, "test"),
    "item12": (item_12, "train"),
    "item13": (item_13, "test"),
    "item20": (item_20, "train"),
    "item21": (item_21, "train"),
    "item22": (item_22, "train"),
    "item40": (item_40, "train"),
    "item41": (item_41, "test"),
    "user4": (user_4, "test"),
    "user7": (user_7, "test"),
    "user10": (user_10, "test"),
    "user20": (user_20, "test"),
    "user21": (user_21, "test"),
}
