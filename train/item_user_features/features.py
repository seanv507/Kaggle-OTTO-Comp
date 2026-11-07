"""Item and user features from Chris Deotte's part of the 3rd place OTTO solution.

Converts the item-features-* / user-features-* notebooks of train/item_user_features/
(run by script.ipynb; removed after conversion, read them at upstream commit 2bb8962).
Each `FEATURES` entry is one notebook: a function of the train or test events returning
a LazyFrame keyed by `aid` or `session`, with the notebook's columns in its order. `features_reference.py` holds the pandas transcriptions the tests
compare against.

Conventions:
- Events are sorted by session, then ts (as data/*.parquet are); `prev`/`next` use that
  row order.
- Times are epoch seconds (`ts.dt.epoch("s")`), as in the notebooks: `hour` is
  ts % 1 day and `day` is ts % 1 week (epoch weeks start on a Thursday).
- Standard deviations use ddof=1; a single-row group's std is -1, as the notebooks'
  `fillna(-1)`.
- Kept quirks of the notebooks: `prev3`/`next3` count the distinct *differences* of
  consecutive aid ids within a session (`aid.diff()`), not distinct neighbours;
  `item_day_std` and `user_day_std` are the same (both count rows per aid and day).
"""

import polars as pl

from event_types import CARTS, CLICKS, ORDERS

DAY = 60 * 60 * 24
WEEK = DAY * 7
T = pl.col("t")  # epoch seconds


def _events(events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return events.lazy().with_columns(t=pl.col("ts").dt.epoch("s"))


def _hour_day(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(hour=T % DAY, day=T % WEEK)


def _std(col: str) -> pl.Expr:
    return pl.col(col).std().fill_null(-1).fill_nan(-1).cast(pl.Float32)


def _rename(lf: pl.LazyFrame, key: str, fn) -> pl.LazyFrame:
    return lf.rename(lambda c: c if c == key else fn(c))


# --- item features ---------------------------------------------------------------------

def _item_stats(lf: pl.LazyFrame, sfx: str = "", hour_day: bool = True) -> pl.LazyFrame:
    """buy_ratio, count_item, count_user, [hour_mean, day_mean,] repeat, buy_ratio_std,
    [hour_std, day_std] per aid, each name + sfx."""
    means = [pl.col("hour").mean().cast(pl.Float32).alias("hour_mean"),
             pl.col("day").mean().cast(pl.Float32).alias("day_mean")] if hour_day else []
    stds = [_std("hour").alias("hour_std"), _std("day").alias("day_std")] if hour_day else []
    return (
        lf.group_by("aid")
        .agg(
            pl.col("type").mean().cast(pl.Float32).alias("buy_ratio"),
            pl.len().cast(pl.Int32).alias("count_item"),
            pl.col("session").n_unique().cast(pl.Int32).alias("count_user"),
            *means,
            _std("type").alias("buy_ratio_std"),
            *stds,
        )
        .with_columns(repeat=(pl.col("count_item") / pl.col("count_user")).cast(pl.Float32))
        .select("aid", "buy_ratio", "count_item", "count_user", *[e.meta.output_name() for e in means],
                "repeat", "buy_ratio_std", *[e.meta.output_name() for e in stds])
        .pipe(_rename, "aid", lambda c: f"{c}{sfx}")
    )


def _repeat(lf: pl.LazyFrame, typ: int, name: str) -> pl.LazyFrame:
    """Mean over the sessions that `typ` an aid of how many times they do."""
    return (
        lf.filter(pl.col("type") == typ)
        .group_by("session", "aid").len()
        .group_by("aid").agg(pl.col("len").mean().alias(name))
    )


def _item_transitions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """prev3, next3, orders3, carts3, order_repeat3, cart_repeat3 per aid."""
    same_prev = pl.col("session") == pl.col("session").shift(1)
    same_next = pl.col("session") == pl.col("session").shift(-1)
    trans = (
        lf.with_columns(
            prev=pl.when(same_prev).then(pl.col("aid") - pl.col("aid").shift(1)),
            next=pl.when(same_next).then(pl.col("aid") - pl.col("aid").shift(-1)),
        )
        .group_by("aid")
        .agg(
            pl.col("prev").drop_nulls().n_unique().cast(pl.Int32).alias("prev3"),
            pl.col("next").drop_nulls().n_unique().cast(pl.Int32).alias("next3"),
            (pl.col("type") == ORDERS).sum().cast(pl.Int32).alias("orders3"),
            (pl.col("type") == CARTS).sum().cast(pl.Int32).alias("carts3"),
        )
    )
    return (
        trans.join(_repeat(lf, ORDERS, "order_repeat3"), on="aid", how="left")
        .join(_repeat(lf, CARTS, "cart_repeat3"), on="aid", how="left")
        .with_columns(pl.col("order_repeat3", "cart_repeat3").fill_null(-1).cast(pl.Float32))
    )


def _item_10_like(lf: pl.LazyFrame) -> pl.LazyFrame:
    return _item_stats(lf, "3").join(_item_transitions(lf), on="aid")


def item_4(train) -> pl.LazyFrame:
    return _item_stats(_hour_day(_events(train)))


def item_10(test) -> pl.LazyFrame:
    return _item_10_like(_hour_day(_events(test)))


def _day_std(lf: pl.LazyFrame) -> pl.LazyFrame:
    """item_day_std, user_day_std: std (ddof=1) of an aid's rows per day over every day of
    `lf` (days without rows count 0), from the sums instead of the notebooks' aid x day grid."""
    n_days = pl.col("day").n_unique()
    per_day = lf.group_by("aid", "day").agg(c=pl.len().cast(pl.Float64))
    days = lf.select(n=n_days)
    std = (
        per_day.group_by("aid").agg(s=pl.col("c").sum(), ss=(pl.col("c") ** 2).sum())
        .join(days, how="cross")
        .select(
            "aid",
            item_day_std=pl.when(pl.col("n") > 1)
            .then(((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).clip(0).sqrt())
            .otherwise(-1)
            .cast(pl.Float32),
        )
    )
    return std.with_columns(user_day_std=pl.col("item_day_std"))


def item_12(train) -> pl.LazyFrame:
    lf = _events(train)
    mn = T.min()
    lf = lf.with_columns(day=(T - mn) // DAY, smin=T.min().over("session"), mn=mn)
    window = lf.filter(pl.col("smin") > pl.col("mn") + 3 * DAY, T - pl.col("smin") < WEEK)
    stats = _item_stats(window, hour_day=False)
    # the notebook's concat(axis=1) outer-joins on aid: aids outside `window` get nulls
    return (
        _day_std(lf).join(stats, on="aid", how="left")
        .select("aid", "buy_ratio", "count_item", "count_user", "repeat", "buy_ratio_std",
                "item_day_std", "user_day_std")
        .pipe(_rename, "aid", lambda c: f"{c}_5")
    )


def item_13(test) -> pl.LazyFrame:
    lf = _events(test)
    lf = lf.with_columns(day=(T - T.min()) // DAY)
    days = lf.select(pl.col("day").unique().sort()).collect()["day"].to_list()
    per_day = lf.group_by("aid", "day").agg(
        buy_ratio=pl.col("type").mean().cast(pl.Float32),
        count_item=pl.len().cast(pl.Int32),
        count_user=pl.col("session").n_unique().cast(pl.Int32),
    )
    pivot = per_day.group_by("aid").agg(
        pl.when(pl.col("day") == d).then(pl.col(c)).max().fill_null(-1).alias(f"{c}_{k}")
        for c in ["buy_ratio", "count_item", "count_user"]
        for k, d in enumerate(days)
    )
    return pivot.join(_day_std(lf), on="aid").pipe(_rename, "aid", lambda c: f"{c}_6")


def _last_char(sfx: str):
    return lambda c: f"{c[:-1]}{sfx}"


def item_20(train) -> pl.LazyFrame:
    lf = _events(train)
    lf = lf.filter(T >= T.max() - WEEK)
    return _item_10_like(_hour_day(lf)).pipe(_rename, "aid", _last_char("7"))


def item_21(train) -> pl.LazyFrame:
    lf = _hour_day(_events(train))
    lf = lf.filter((pl.col("hour") >= 14 * 3600) | (pl.col("hour") < 2 * 3600))
    return _item_10_like(lf).pipe(_rename, "aid", _last_char("8"))


def item_22(train) -> pl.LazyFrame:
    lf = _hour_day(_events(train))
    lf = lf.filter((pl.col("hour") < 14 * 3600) & (pl.col("hour") >= 2 * 3600))
    return _item_10_like(lf).pipe(_rename, "aid", _last_char("9"))


def item_40(train) -> pl.LazyFrame:
    lf = _hour_day(_events(train).filter(pl.col("type") != CLICKS))
    return _item_stats(lf, "11")


def item_41(test) -> pl.LazyFrame:
    lf = _hour_day(_events(test).filter(pl.col("type") != CLICKS))
    return _item_stats(lf, "12")


# --- user features ---------------------------------------------------------------------

def _user_stats(lf: pl.LazyFrame, sfx: str) -> pl.LazyFrame:
    out = (
        lf.group_by("session")
        .agg(
            pl.col("type").mean().cast(pl.Float32).alias("buy_ratio2"),
            pl.len().cast(pl.Int32).alias("count_item2"),
            pl.col("aid").n_unique().cast(pl.Int32).alias("unique_item2"),
            pl.col("hour").mean().cast(pl.Float32).alias("hour_mean2"),
            pl.col("day").mean().cast(pl.Float32).alias("day_mean2"),
            _std("type").alias("buy_ratio_std2"),
            _std("hour").alias("hour_std2"),
            _std("day").alias("day_std2"),
        )
        .with_columns(repeat2=(pl.col("count_item2") / pl.col("unique_item2")).cast(pl.Float32))
        .select("session", "buy_ratio2", "count_item2", "unique_item2", "hour_mean2", "day_mean2",
                "repeat2", "buy_ratio_std2", "hour_std2", "day_std2")
    )
    return out.pipe(_rename, "session", lambda c: c.replace("2", sfx))


def _last_day(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(T.max().over("session") - T < DAY)


def user_4(test) -> pl.LazyFrame:
    return _user_stats(_hour_day(_events(test)), "2")


def user_7(test) -> pl.LazyFrame:
    lf = _events(test)
    mn = lf.select(mn=T.min())
    return (
        lf.group_by("session").agg(mx=T.max(), mi=T.min())
        .join(mn, how="cross")
        .select(
            "session",
            ts_max7=(pl.col("mx") - pl.col("mn")).cast(pl.Int32),
            ts_min7=(pl.col("mi") - pl.col("mn")).cast(pl.Int32),
            ts_diff7=(pl.col("mx") - pl.col("mi")).cast(pl.Int32),
        )
    )


def user_10(test) -> pl.LazyFrame:
    return _user_stats(_hour_day(_events(test).filter(pl.col("type") != CLICKS)), "10")


def user_20(test) -> pl.LazyFrame:
    return _user_stats(_hour_day(_last_day(_events(test))), "14")


def user_21(test) -> pl.LazyFrame:
    lf = _last_day(_events(test)).filter(pl.col("type") != CLICKS)
    return _user_stats(_hour_day(lf), "15")


# output name (the notebook's .pqt) -> (function, which events it reads)
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
