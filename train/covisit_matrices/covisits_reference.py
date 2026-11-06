"""Pandas ports of Chris Deotte's covisitation notebooks, used by test_covisits.py.

Each `ref_<VER>` transcribes the pair-building code of
train/covisit_matrices/{cpu,gpu}-<VER>.ipynb as literally as
possible, with cudf replaced by pandas. Pandas merges keep left rows in order, then right
rows, so `drop_duplicates` (keep first) picks the copy the notebook's sort implies; on
cudf that choice was arbitrary.

Input: pandas DataFrame with session, aid, ts (epoch seconds, int) and type, sorted by
session then ts, like one of the reference's data files (sessions never straddle
files). Output: the full matrix (no top-k) as a Polars DataFrame aid, aid_y, value.
"""

import numpy as np
import pandas as pd
import polars as pl


def to_reference_frame(events: pl.DataFrame) -> pd.DataFrame:
    return events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas()


def _out(df: pd.DataFrame) -> pl.DataFrame:
    out = df.groupby(["aid_x", "aid_y"], as_index=False)["wgt"].sum()
    return pl.from_pandas(out.rename(columns={"aid_x": "aid", "wgt": "value"}))


def _sum_parts(parts: list[pd.DataFrame]) -> pl.DataFrame:
    return _out(pd.concat(parts))


def ref_93(df):
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df.reset_index(drop=True)
    df["n"] = df.groupby("session").cumcount()
    df = df.loc[df.n < 30].drop("n", axis=1)
    df = df.merge(df, on="session")
    df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]
    df = df[["session", "aid_x", "aid_y", "ts_x", "ts_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
    df["wgt"] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
    df["wgt"] = df["wgt"] * ((1 / 2) ** ((df.ts_x - df.ts_y).abs() / 60 / 60))
    return _out(df)


def ref_99(df):
    chunk = df.loc[df.ts > 1662328791 - 60 * 60 * 24 * 28]
    chunk = chunk.loc[chunk["type"].isin([1, 2])]
    # gen_pairs
    df = pd.merge(chunk, chunk, on="session")
    pairs = df.query("aid_x != aid_y")[["session", "aid_x", "aid_y", "ts_x", "ts_y"]].drop_duplicates(
        ["session", "aid_x", "aid_y"]
    )
    pairs["wgt"] = (1 / 2) ** (np.abs(pairs.ts_y - pairs.ts_x) / 60 / 60)
    return _out(pairs)


def ref_115(df):
    type_weight = {0: 1, 1: 1, 2: 1}
    df = df.loc[df.ts > 1662328791 - 60 * 60 * 24 * 21]
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df.merge(df, on="session")
    df = df[["session", "aid_x", "aid_y", "type_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
    w = (1 / 2) ** ((df.ts_y - df.ts_x).abs() / 60 / 60)
    df["wgt"] = df.type_y.map(type_weight) * w
    return _out(df)


def ref_116(df):
    type_weight = {0: 0, 1: 1, 2: 1}
    df = df.loc[df.ts > 1662328791 - 60 * 60 * 24 * 28]
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df.merge(df, on="session")
    df = df.loc[(df.ts_y - df.ts_x) > 0]
    df = df[["session", "aid_x", "aid_y", "type_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
    w = (1 / 2) ** ((df.ts_y - df.ts_x) / 60 / 60)
    df["wgt"] = df.type_y.map(type_weight) * w
    return _out(df)


def ref_155(df):
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df.merge(df.loc[df["type"].isin([1, 2])], on="session")
    df = df.loc[((df.ts_y - df.ts_x) > 0)]
    df = df[["session", "aid_x", "aid_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
    df["wgt"] = (1 / 2) ** ((df.ts_y - df.ts_x) / 60 / 60 / 6)
    return _out(df)


def ref_157(df):
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df.merge(df, on="session")
    df = df[["session", "aid_x", "aid_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
    df["wgt"] = (1 / 2) ** ((df.ts_y - df.ts_x).abs() / 60 / 60 / 6)
    return _out(df)


def _ref_165_166(train, test, short: bool):
    parts = []
    for df, max_len in [(train, 6), (test, 3)]:  # reference: files 0-99 train, 100-119 test
        df = df.copy()
        df["x"] = df.groupby("session").aid.transform("count")
        df = df.loc[df.x <= max_len] if short else df.loc[df.x > max_len]
        df = df.drop("x", axis=1)
        df = df.sort_values(["session", "ts"], ascending=[True, False])
        df = df.merge(df.loc[df["type"].isin([1, 2])], on="session")
        df = df[["session", "aid_x", "aid_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
        df["wgt"] = (1 / 2) ** ((df.ts_y - df.ts_x).abs() / 60 / 60)
        parts.append(df)
    return _sum_parts(parts)


def ref_165(train, test):
    return _ref_165_166(train, test, short=True)


def ref_166(train, test):
    return _ref_165_166(train, test, short=False)


def _ref_167_168(df, morning: bool):
    df = df.copy()
    df.ts += 60 * 60 * 2
    df["x"] = df.ts % (60 * 60 * 24)
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    x = df.loc[df.x < 60 * 60 * 14] if morning else df.loc[df.x >= 60 * 60 * 14]
    df = x.merge(df.loc[df["type"].isin([1, 2])], on="session")
    df = df[["session", "aid_x", "aid_y", "ts_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
    df["wgt"] = (1 / 2) ** ((df.ts_y - df.ts_x).abs() / 60 / 60)
    return _out(df)


def ref_167(df):
    return _ref_167_168(df, morning=True)


def ref_168(df):
    return _ref_167_168(df, morning=False)


def _ref_neighbours(df, keep, carts_orders=False, time_weight=False):
    if carts_orders:
        df = df.loc[df["type"].isin([1, 2])]
    df = df.sort_values(["session", "ts"], ascending=[True, True])
    df["k"] = np.arange(len(df))
    df = df.merge(df.drop_duplicates(["session", "aid"]), on=["session"])
    df = df.loc[keep(df.k_y - df.k_x) & (df.aid_x != df.aid_y)]
    if time_weight:
        df = df[["session", "aid_x", "aid_y", "ts_x", "ts_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
        df["wgt"] = (1 / 2) ** ((df.ts_x - df.ts_y).abs() / 60 / 60)
    else:
        df = df[["session", "aid_x", "aid_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
        df["wgt"] = 1
    return _out(df)


def ref_217(df):
    return _ref_neighbours(df, lambda d: d.abs() == 1)


def ref_220(df):
    return _ref_neighbours(df, lambda d: (d.abs() >= 1) & (d.abs() <= 2))


def ref_226(df):
    return _ref_neighbours(df, lambda d: (d >= 1) & (d <= 3), carts_orders=True)


def ref_232(df):
    return _ref_neighbours(df, lambda d: (d.abs() >= 1) & (d.abs() <= 3), time_weight=True)


def ref_235(df):
    return _ref_neighbours(df, lambda d: (d.abs() >= 1) & (d.abs() <= 6), time_weight=True)


def _split_files(df, parts):
    """Cut sessions into files as data/make_train_valid.ipynb does."""
    users = df.session.unique()
    chunk = int(np.ceil(len(users) / parts))
    return [df.loc[df.session.isin(users[k * chunk : (k + 1) * chunk])] for k in range(parts)]


def ref_239(train, test, train_parts=100, test_parts=20):
    parts = []
    for df in _split_files(train, train_parts) + _split_files(test, test_parts):
        df = df.merge(df.loc[df["type"].isin([1, 2])], on=["session"])
        df = df.loc[(df.ts_y - df.ts_x > 0) & (df.ts_y - df.ts_x < 60 * 60 * 2)]
        df["d"] = df.ts_y - df.ts_x
        df = df.sort_values(["d"], ascending=[True])
        df = df.drop_duplicates(["aid_x", "aid_y"]).drop("d", axis=1)
        df["wgt"] = (1 / 2) ** ((df.ts_y - df.ts_x) / 60 / 60)
        parts.append(df)
    return _sum_parts(parts)


def _ref_700_701(df):
    df = df.copy()
    df["mn"] = df.groupby("session").ts.transform("min")
    df = df.loc[df.mn > 1662328791 - 60 * 60 * 24 * 21]
    df = df.drop("mn", axis=1)
    df = df.sort_values(["session", "ts"], ascending=[True, True])
    df = df.reset_index(drop=True)
    df["n"] = df.groupby("session").cumcount()
    return df


def ref_700(df):
    df = _ref_700_701(df)
    df = df.loc[df.n == 0].merge(df.loc[df["type"] > 0], on=["session"])
    df = df.loc[df.aid_x != df.aid_y]
    df = df.sort_values(["session", "ts_y"], ascending=[True, True])
    df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y", "type_y"])
    df["wgt"] = 1
    return _out(df)


def ref_701(df):
    df = _ref_700_701(df)
    df = df.loc[df.n == 0].merge(df, on=["session"])
    df = df.loc[df.aid_x != df.aid_y]
    df = df.sort_values(["session", "ts_y"], ascending=[True, True], kind="stable")
    df = df[["session", "aid_x", "aid_y", "ts_x", "ts_y", "type_y"]].drop_duplicates(
        ["session", "aid_x", "aid_y", "type_y"]
    )
    df["wgt"] = (1 / 2) ** ((df.ts_x - df.ts_y).abs() / 60 / 60)
    return _out(df)
