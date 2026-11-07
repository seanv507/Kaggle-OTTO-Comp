"""Merge candidates, scores, features and targets into the rankers' training rows:
Chris Deotte's Make-152 (clicks), Make-172 (orders) and Make-174 (carts) notebooks,
steps A-G. `merge_reference.py` holds the pandas transcriptions the tests compare against.

One row per candidate of model 709 for the target (100 per session, padding included):
`user, item, pos, wgt, <target>`, then the score of every other candidate run that
proposed the item (`score_<k>`, -3 if it did not), item and user features, and
embedding similarities. For 172 / 174 only sessions with at least one positive are kept.
Step H (the clicks ranker's out-of-fold predictions, `click_oof`) needs the trained
rankers and belongs to train/ranker_models.

Conventions (see merge_reference.py for the notebook quirks kept):
- Inputs are save_candidates.py / save_features.py / save_embeddings.py outputs; events
  are sorted by session, ts, and row order breaks ts ties as the notebooks' stable sorts.
- Joins and feature lookups are lazy (engine-agnostic); the embedding similarities are
  numpy gathers on the collected rows, as in the notebooks.
- `make` handles any subset of sessions; save_merged.py runs it in session chunks.
"""

import warnings

import numpy as np
import polars as pl
import polars.selectors as cs

from event_types import CARTS, CLICKS, ORDERS

TARGET = {"152": "clicks", "172": "orders", "174": "carts"}
TYPE = {"clicks": CLICKS, "carts": CARTS, "orders": ORDERS}


def score_sources(use: str) -> list[tuple[str, str]]:
    """(score column, candidate run) in the order Make-152-B / 172-B / 174-B merge them."""
    other_709 = "709_orders" if use == "clicks" else "709_clicks"
    return (
        [(f"score_{k}", f"564_{k}") for k in [1, 3, 4, 5, 6, 7]]
        + [("score_10", "564_F_orders"), ("score_11", "612_orders")]
        + [("score_12", "564_F_clicks"), ("score_13", "612_clicks")]
        + [("score_30", other_709)]
        + [(f"score_{14 + j}", f"564_{k}") for j, k in enumerate(range(20, 28))]
        + [("score_22", "614_orders"), ("score_23", "614_clicks")]
        + [(f"score_{24 + j}", f"564_{k}") for j, k in enumerate(range(31, 34))]
    )


def base_run(use: str) -> str:
    """The 709 run whose candidates are the rows (carts use the orders list)."""
    return f"709_{'orders' if use == 'carts' else use}"


def runs_needed(model: str) -> list[str]:
    use = TARGET[model]
    return [base_run(use)] + [run for _, run in score_sources(use)]


FEATURES_ITEM_B = ["item12", "item13", "item4", "item10"]
FEATURES_ITEM_E = ["item20", "item21", "item22"]


# --- lazy pieces ------------------------------------------------------------------------

def _candidates(lf: pl.LazyFrame, name: str = "score") -> pl.LazyFrame:
    return lf.select(
        user=pl.col("session").cast(pl.Int32),
        pos=pl.col("pos").cast(pl.Int32),
        item=pl.col("aid").cast(pl.Int32),
        **{name: pl.col("score").cast(pl.Float32)},
    )


def _feature(lf: pl.LazyFrame, on: str, rename=None) -> pl.LazyFrame:
    """A feature file keyed for joining on `on` (item/last: aid, user: session), int32
    columns as float32 (the notebooks' cast before each merge)."""
    key = "session" if on == "user" else "aid"
    lf = lf.with_columns(cs.integer().exclude(key).cast(pl.Float32)).rename({key: on})
    if rename:
        lf = lf.rename(lambda c: c if c == on else rename(c))
    return lf


def _left(users: pl.LazyFrame, other: pl.LazyFrame, on) -> pl.LazyFrame:
    return users.join(other, on=on, how="left", maintain_order="left")


def step_a(cands: dict[str, pl.LazyFrame], labels: pl.LazyFrame, use: str, keep_positive_users: bool) -> pl.LazyFrame:
    users = _candidates(cands[base_run(use)], "wgt").select("user", "item", "pos", "wgt")
    truth = (
        labels.filter(pl.col("type") == TYPE[use])
        .select(user=pl.col("session").cast(pl.Int32), item=pl.col("ground_truth"))
        .explode("item", empty_as_null=False)
        .drop_nulls()
        .with_columns(pl.col("item").cast(pl.Int32), pl.lit(1, pl.Int8).alias(use))
    )
    users = _left(users, truth, ["user", "item"]).with_columns(pl.col(use).fill_null(0))
    if keep_positive_users:
        users = users.filter(pl.col(use).sum().over("user") > 0)
    return users


def step_b(users: pl.LazyFrame, cands: dict[str, pl.LazyFrame], feats: dict[str, pl.LazyFrame], use: str) -> pl.LazyFrame:
    for col, run in score_sources(use):
        scores = _candidates(cands[run], col).filter(pl.col(col) != -2).select("user", "item", col)
        users = _left(users, scores, ["user", "item"]).with_columns(pl.col(col).fill_null(-3))
    for name in FEATURES_ITEM_B:
        users = _left(users, _feature(feats[name], "item"), "item")
    return _left(users, _feature(feats["user4"], "user"), "user")


def step_c(users: pl.LazyFrame, feats: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    user7 = feats["user7"].rename({"session": "user"})
    return _left(users, user7, "user").sort("user", "pos", maintain_order=True)


def step_e(users: pl.LazyFrame, feats: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    for name in FEATURES_ITEM_E:
        users = _left(users, _feature(feats[name], "item"), "item")
    return _left(users, _feature(feats["user10"], "user"), "user")


def _nth(events: pl.LazyFrame, k: int, name: str, newest_first: bool = False) -> pl.LazyFrame:
    """The k-th event's aid per session (0 = first; newest first if asked), ties in ts in
    row order (the notebooks' stable sorts)."""
    if newest_first:
        events = events.sort("session", "ts", descending=[False, True], maintain_order=True)
    return (
        events.with_row_index("_r")
        .filter(pl.col("_r") - pl.col("_r").min().over("session") == k)
        .select(user=pl.col("session"), **{name: pl.col("aid")})
    )


# --- embedding similarities (numpy) ------------------------------------------------------

def normalize(embed: np.ndarray) -> np.ndarray:
    return embed / np.sqrt(np.sum(embed * embed, axis=1, keepdims=True))


def _cos(embed: np.ndarray, a: pl.Series, b: pl.Series, missing: float = -1.0) -> np.ndarray:
    """embed[a] . embed[b] (embed normalized); `missing` where b is null."""
    has = b.is_not_null().to_numpy()
    out = np.full(len(a), missing, dtype=np.float32)
    ai = a.to_numpy()[has]
    bi = b.to_numpy()[has].astype(np.int64)
    out[has] = np.sum(embed[ai] * embed[bi], axis=1)
    return out


def _lookup(users: pl.DataFrame, events: pl.LazyFrame, parts: list[pl.LazyFrame], engine) -> pl.DataFrame:
    """users' (user) column with the per-session aids of `parts` joined on, in row order."""
    out = users.lazy().select("user")
    for p in parts:
        out = _left(out, p, "user")
    return out.collect(engine=engine)


def step_d(users: pl.DataFrame, events: pl.LazyFrame, embed: np.ndarray, engine) -> pl.DataFrame:
    buys = events.filter(pl.col("type") != CLICKS)
    last = events.group_by("session", maintain_order=True).agg(last=pl.col("aid").last()).rename({"session": "user"})
    aids = _lookup(users, events, [last] + [_nth(buys, k, f"buy_{k}") for k in range(5)], engine)
    z = {"z1": _cos(embed, users["item"], aids["last"], missing=np.nan)}
    z |= {f"z{k + 2}": _cos(embed, users["item"], aids[f"buy_{k}"]) for k in range(5)}
    return users.with_columns(pl.Series(k, v, pl.Float32) for k, v in z.items())


def step_f(users: pl.DataFrame, events: pl.LazyFrame, embed: np.ndarray, feats: dict[str, pl.LazyFrame], engine) -> pl.DataFrame:
    per_session = events.group_by("session", maintain_order=True).agg(
        last=pl.col("aid").last(), first=pl.col("aid").first()
    )
    # last2: the last aid more than a day before the session's end (else the last aid)
    older = (
        events.with_columns(age=(pl.col("ts").max().over("session") - pl.col("ts")).dt.total_seconds())
        .filter(pl.col("age") > 24 * 60 * 60)
        .group_by("session", maintain_order=True)
        .agg(last2=pl.col("aid").last())
    )
    per_session = (
        _left(per_session, older, "session")
        .rename({"session": "user"})
        .with_columns(pl.col("last2").fill_null(pl.col("last")))
    )
    aids = _lookup(users, events, [per_session], engine)
    e_last = aids["last"]
    z10 = _cos(embed, aids["first"], e_last)
    z11 = _cos(embed, aids["last2"], e_last)
    rows = pl.LazyFrame({"item": users["item"], "last": e_last, "user": users["user"]})
    rows = _left(rows, _feature(feats["item40"], "item"), "item")
    rows = _left(rows, _feature(feats["item41"], "item"), "item")
    rows = _left(rows, _feature(feats["item41"], "last", rename=lambda c: c.replace("12", "13")), "last")
    rows = _left(rows, _feature(feats["user20"], "user"), "user")
    rows = _left(rows, _feature(feats["user21"], "user"), "user")
    rows = rows.drop("item", "last", "user").collect(engine=engine)
    return users.with_columns(pl.Series("z10", z10), pl.Series("z11", z11)).hstack(rows)


def step_g(users: pl.DataFrame, events: pl.LazyFrame, emb: np.ndarray, engine) -> pl.DataFrame:
    buys = events.filter(pl.col("type") != CLICKS)
    parts = (
        [_nth(events, k, f"last_{k}", newest_first=True) for k in range(5)]
        + [_nth(events, k, f"first_{k}") for k in range(5)]
        + [_nth(buys, k, f"buy_{k}", newest_first=True) for k in range(5)]
    )
    aids = _lookup(users, events, parts, engine)
    cols = [c for c in aids.columns if c != "user"]
    sims = np.column_stack([_cos(emb, users["item"], aids[c], missing=np.nan).astype(np.float64) for c in cols])
    sims[sims == -1] = np.nan  # the notebook's -1 sentinel also catches a similarity of exactly -1
    with warnings.catch_warnings():  # all-nan / single-value rows -> nan, filled with -1 below
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(sims, axis=1)
        std = np.nanstd(sims, axis=1, ddof=1)
    out = {c: sims[:, i] for i, c in enumerate(cols)}
    for k in range(4):  # the notebook takes all 15 columns for each k (sic)
        out[f"e_mean_{k}"] = mean
        out[f"e_std_{k}"] = std
    return users.with_columns(
        pl.Series(c, np.nan_to_num(v, nan=-1.0), pl.Float32) for c, v in out.items()
    )


def make(
    model: str,
    cands: dict[str, pl.LazyFrame],
    labels: pl.LazyFrame,
    events: pl.LazyFrame | pl.DataFrame,
    feats: dict[str, pl.LazyFrame],
    embed: np.ndarray,
    giba_embed: np.ndarray | None = None,
    engine="in-memory",
) -> pl.DataFrame:
    """train_v152 / train_v172 / train_v174 (before step H) for the sessions in `events`.

    `cands`: candidate run -> save_candidates.py file; `feats`: feature name ->
    save_features.py file; `embed`: save_embeddings.py's array; `giba_embed`: Giba's
    embbedings_fm10.npy for step G (the notebook falls back to `embed` without it)."""
    use = TARGET[model]
    events = events.lazy()
    sessions = events.select(pl.col("session").unique())
    cands = {k: v.join(sessions, on="session", how="semi") for k, v in cands.items()}
    emb = normalize(embed)
    users = step_a(cands, labels, use, keep_positive_users=model != "152")
    users = step_c(step_b(users, cands, feats, use), feats).collect(engine=engine)
    users = step_d(users, events, emb, engine)
    users = step_e(users.lazy(), feats).collect(engine=engine)
    if model != "152":
        users = step_f(users, events, emb, feats, engine)
        users = step_g(users, events, emb if giba_embed is None else normalize(giba_embed), engine)
    return users
