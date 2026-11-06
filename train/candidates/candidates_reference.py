# ruff: noqa: B905, B007  (transcribed loops: zip truncation is part of the semantics)
"""Pandas ports of Chris Deotte's candidate notebooks, used by test_candidates.py.

Code is transcribed from train/candidates/*.ipynb (also at upstream commit 2bb8962) as
literally as possible. Deliberate changes:
- `sort_values('ts', ascending=False)` uses kind='stable', so ties in ts always keep
  the earliest row (the notebooks' default quicksort only does that on short sessions).
- Input `ts` is epoch seconds, as in the reference; use `to_reference_frame`.
"""

import itertools
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl


def to_reference_frame(events: pl.DataFrame) -> pd.DataFrame:
    return events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas()


def add_d(test_df: pd.DataFrame) -> pd.DataFrame:
    """make-valid-with-d.ipynb"""
    test_df = test_df.sort_values(["session", "ts"])
    test_df["d"] = test_df.groupby("session").ts.diff()
    test_df.d = (test_df.d > 60 * 60 * 2).astype("int16").fillna(0)
    test_df.d = test_df.groupby("session").d.cumsum()
    return test_df


def add_a_b(valid: pd.DataFrame) -> pd.DataFrame:
    """make-valid-lists.ipynb (without the day column and the pickled lists)."""
    valid = valid.sort_values(["session", "ts"])
    valid["x"] = valid.groupby(["session", "d"]).d.cumcount()
    valid["a"] = 0
    valid.loc[valid.x == 0, "a"] = 1
    valid = valid.sort_values(["session", "ts"], ascending=[True, False])
    valid["x"] = valid.groupby(["session", "d"]).d.cumcount()
    valid["b"] = 0
    valid.loc[valid.x == 0, "b"] = 1
    valid = valid.sort_values(["session", "ts"])
    valid = valid.drop("x", axis=1)
    return valid


def source_sets_orders(df: pd.DataFrame) -> dict[str, list[int]]:
    """The lists built at the top of model-564 `suggest_orders` (one session)."""
    aids = df.aid.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))

    mx = df.d.max()
    aids2 = df.loc[df.d == mx].aid.tolist()
    unique_aids4 = list(dict.fromkeys(aids2[::-1]))

    mx = df.ts.max()
    aids2 = df.loc[df.ts >= mx - 60 * 60 / 2].aid.tolist()
    unique_aids5 = list(dict.fromkeys(aids2[::-1]))  # recent 1 hour

    df2 = df.drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids2 = list(dict.fromkeys(aids2[::-1]))  # first of each session

    df2 = df.sort_values("ts", ascending=False, kind="stable").drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids3 = list(dict.fromkeys(aids2))  # last of each session

    df = df.loc[df["type"].isin([1, 2])]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    return {
        "all": unique_aids,
        "first_of_d": unique_aids2,
        "last_of_d": unique_aids3,
        "last_d": unique_aids4,
        "last_30min": unique_aids5,
        "buys": unique_buys,
    }


def source_sets_aids(df: pd.DataFrame) -> dict[str, list[int]]:
    """The lists built at the top of model-564 `suggest_aids` (one session)."""
    aids = df.aid.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))

    df2 = df.sort_values("ts", ascending=False, kind="stable").drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids3 = list(dict.fromkeys(aids2[::-1]))  # last of each session

    mx = df.d.max()
    aids2 = df.loc[df.d == mx].aid.tolist()
    unique_aids4 = list(dict.fromkeys(aids2[::-1]))

    df = df.loc[df["type"].isin([1, 2])]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    return {
        "all": unique_aids,
        "last_of_d_oldest": unique_aids3,
        "last_d": unique_aids4,
        "buys": unique_buys,
    }


def source_sets_620(row: list) -> dict[str, list[int]]:
    """The lists built at the top of model-620 `suggest_aids`, from one
    make-valid-lists row [session, aids, types, ts, d, a, b, day]."""
    aids = row[1]
    types = row[2]
    ds = row[4]
    ds2 = row[6]
    unique_aids = list(dict.fromkeys(aids[::-1]))
    unique_aids3 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds2[i] == 1][::-1]))
    mx = max(ds)
    unique_aids4 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds[i] == mx][::-1]))
    unique_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))
    return {"all": unique_aids, "last_of_d": unique_aids3, "last_d": unique_aids4, "buys": unique_buys}


def valid_lists(valid: pd.DataFrame) -> list[list]:
    """make-valid-lists.ipynb rows: [session, aid, type, ts, d, a, b] lists per session."""
    return [
        [h.session.iloc[0], h.aid.to_list(), h.type.to_list(), h.ts.to_list(), h.d.to_list(),
         h.a.to_list(), h.b.to_list()]
        for _, h in valid.sort_values(["session", "ts"]).groupby("session")
    ]


# ---------------------------------------------------------------------------
# model-564-*: suggest_aids / suggest_orders
# ---------------------------------------------------------------------------
# Transcribed from model-564-1.ipynb (the variants differ only in INFER, the branch
# condition and whether they return aids or their scores: see `branch_564` and
# `_out`). Each covisitation matrix is a dict aid -> list of aid_y, best first, keyed
# by the notebook's variable name: top_20 -> M["top20"], top_20_test2 -> M["top20test2"].
# Returns (aids, scores) where scores are what the -wgt notebooks write: the Counter
# value, -1 for history aids in the covisit branch and -2 for popular-item padding.
#
# model-612 / model-614 add rules for sessions with a single unique aid (marked
# "model-612" below), under the INFER names top20new2 (v701) and top20new (v700) here:
# 612 = all rules with branch "covisit"; 614 = INFER ["top20new", "top20new2"] only,
# because 614 wraps every other covisit rule in `if 0:`.

type_weight_multipliers = {0: 1, 1: 6, 2: 3}

ALL_564 = ["top20", "top20b", "top20c", "top20d", "top20e", "top20f", "top20orders", "top20buy2buy",
           "top20buy2buy2", "top20test", "top20test2", "top20buy"]


def branch_564(branch: str, unique_aids: list) -> bool:
    """True for the history branch. `auto` is the notebooks' `len(unique_aids)>=20`,
    `history` their `if 1:` and `covisit` their `if 0:`."""
    return {"auto": len(unique_aids) >= 20, "history": True, "covisit": False}[branch]


def _ranked(aids_temp: Counter, n: int, exclude=()) -> tuple[list, list]:
    top = [(k, v) for k, v in aids_temp.most_common(n) if k not in exclude]
    return [k for k, v in top], [v for k, v in top]


def _out(aids, scores, top, item_ct):
    """The notebooks' `(result + top_x[:ITEM_CT-len(result)])[:ITEM_CT]`, for the aids and
    (with top_x = [-2]*100, as in -wgt) for the scores."""
    aids = (aids + top[: item_ct - len(aids)])[:item_ct]
    scores = (scores + [-2] * 100)[:item_ct]
    return aids, scores


def suggest_aids_564(df, M, INFER=ALL_564, branch="auto", ITEM_CT=50, top_clicks=()):
    top_clicks = list(top_clicks)
    top_20, top_20b, top_20c = M.get("top20"), M.get("top20b"), M.get("top20c")
    top_20e, top_20f, top_20_test2 = M.get("top20e"), M.get("top20f"), M.get("top20test2")
    top_20_new2 = M.get("top20new2")

    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))

    df2 = df.sort_values("ts", ascending=False, kind="stable").drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids3 = list(dict.fromkeys(aids2[::-1]))  # last of each session

    mx = df.d.max()
    aids2 = df.loc[df.d == mx].aid.tolist()
    unique_aids4 = list(dict.fromkeys(aids2[::-1]))

    df = df.loc[df["type"].isin([1, 2])]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))  # noqa: F841 (unused, as in the notebook)

    ln = len(unique_aids)

    if branch_564(branch, unique_aids):
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        if "top20c" in INFER:
            aids3 = list(itertools.chain(*[top_20c[aid][:20] for aid in unique_aids[:2] if aid in top_20c]))
            for i, aid in enumerate(aids3):
                aids_temp[aid] += 0.6
        if "top20b" in INFER:
            aids3 = list(itertools.chain(*[top_20b[aid][:15] for aid in unique_aids3 if aid in top_20b]))
            for i, aid in enumerate(aids3):
                aids_temp[aid] += 0.3
        if "top20test2" in INFER:
            aids3 = list(itertools.chain(*[top_20_test2[aid][:20] for aid in unique_aids[:2] if aid in top_20_test2]))
            for i, aid in enumerate(aids3):
                aids_temp[aid] += 0.6

        result, scores = _ranked(aids_temp, ITEM_CT)
        return _out(result, scores, top_clicks, ITEM_CT)

    aids_temp = Counter()

    # model-612
    weights3 = [2, 2] + [1] * 28
    if "top20new2" in INFER and len(unique_aids) == 1:
        aids5 = list(itertools.chain(*[top_20_new2[aid][:30] for aid in unique_aids[-1:] if aid in top_20_new2]))
        w5 = weights3 * int(len(aids5) // 30)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    if "top20" in INFER:
        aids2 = list(itertools.chain(*[top_20[aid][:20] for aid in unique_aids if aid in top_20]))
        for i, aid in enumerate(aids2):
            m = 0.1 + 0.9 * (ln - (i // 20)) / ln
            aids_temp[aid] += m
            if i % 20 == 0:
                aids_temp[aid] += m
    if "top20b" in INFER:
        aids3 = list(itertools.chain(*[top_20b[aid][:20] for aid in unique_aids[:2] if aid in top_20b]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 1
            if i % 20 == 0:
                aids_temp[aid] += 1
    if "top20test2" in INFER:
        aids3 = list(itertools.chain(*[top_20_test2[aid][:20] for aid in unique_aids[:2] if aid in top_20_test2]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 1
            if i % 20 == 0:
                aids_temp[aid] += 1
    if "top20f" in INFER:
        aids4 = list(itertools.chain(*[top_20f[aid][:10] for aid in unique_aids4 if aid in top_20f]))
        for i, aid in enumerate(aids4):
            w = i // 10
            aids_temp[aid] += 1 - w * 0.1
            if i % 10 == 0:
                aids_temp[aid] += 1 - w * 0.1
    if "top20e" in INFER:
        aids5 = list(itertools.chain(*[top_20e[aid][:20] for aid in unique_aids3 if aid in top_20e]))
        for i, aid in enumerate(aids5):
            aids_temp[aid] += 1
            if i % 20 == 0:
                aids_temp[aid] += 1
        top_aids2 = [k for k, v in aids_temp.most_common(1) if k not in unique_aids]
    if "top20c" in INFER:
        aids3 = list(itertools.chain(*[top_20c[aid][:10] for aid in top_aids2 if aid in top_20c]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 1
            if i % 10 == 0:
                aids_temp[aid] += 1
    top_aids2, scores = _ranked(aids_temp, ITEM_CT, exclude=unique_aids)

    result = unique_aids + top_aids2[: ITEM_CT - len(unique_aids)]
    scores = [-1] * len(unique_aids) + scores[: ITEM_CT - len(unique_aids)]
    return _out(result, scores, top_clicks, ITEM_CT)


def suggest_orders_564(df, M, INFER=ALL_564, branch="auto", ITEM_CT=50, top_orders=()):
    top_orders = list(top_orders)
    top_20b, top_20c, top_20d = M.get("top20b"), M.get("top20c"), M.get("top20d")
    top_20e, top_20f, top_20_orders = M.get("top20e"), M.get("top20f"), M.get("top20orders")
    top_20_buy2buy, top_20_buy2buy2 = M.get("top20buy2buy"), M.get("top20buy2buy2")
    top_20_test, top_20_buy = M.get("top20test"), M.get("top20buy")
    top_20_new, top_20_new2 = M.get("top20new"), M.get("top20new2")

    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))

    mx = df.d.max()
    aids2 = df.loc[df.d == mx].aid.tolist()
    unique_aids4 = list(dict.fromkeys(aids2[::-1]))

    mx = df.ts.max()
    aids2 = df.loc[df.ts >= mx - 60 * 60 / 2].aid.tolist()
    unique_aids5 = list(dict.fromkeys(aids2[::-1]))  # recent 1 hour

    df2 = df.drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids2 = list(dict.fromkeys(aids2[::-1]))  # first of each session

    df2 = df.sort_values("ts", ascending=False, kind="stable").drop_duplicates("d")
    aids2 = df2.aid.tolist()
    unique_aids3 = list(dict.fromkeys(aids2))  # last of each session

    df = df.loc[df["type"].isin([1, 2])]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    if branch_564(branch, unique_aids):
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        for aid in unique_aids2:
            aids_temp[aid] += 0.5
        for aid in unique_aids3:
            aids_temp[aid] += 0.5
        if "top20buy2buy" in INFER:
            aids3 = list(itertools.chain(*[top_20_buy2buy[aid][:40] for aid in unique_buys if aid in top_20_buy2buy]))
            for i, aid in enumerate(aids3):
                aids_temp[aid] += 0.05
                if i % 40 == 0:
                    aids_temp[aid] += 0.05
        if "top20buy2buy2" in INFER:
            aids3 = list(itertools.chain(*[top_20_buy2buy2[aid][:40] for aid in unique_buys if aid in top_20_buy2buy2]))
            for i, aid in enumerate(aids3):
                aids_temp[aid] += 0.1
                if i % 40 == 0:
                    aids_temp[aid] += 0.1
        if "top20test" in INFER:
            aids4 = list(itertools.chain(*[top_20_test[aid][:40] for aid in unique_aids if aid in top_20_test]))
            for i, aid in enumerate(aids4):
                aids_temp[aid] += 0.05
                if i % 40 == 0:
                    aids_temp[aid] += 0.05
        if "top20c" in INFER:
            aids5 = list(itertools.chain(*[top_20c[aid][:20] for aid in unique_aids[:1] if aid in top_20c]))
            for i, aid in enumerate(aids5):
                aids_temp[aid] += 0.05
                if i % 20 == 0:
                    aids_temp[aid] += 0.05
        if "top20d" in INFER:
            aids6 = list(itertools.chain(*[top_20d[aid][:20] for aid in unique_buys[:1] if aid in top_20d]))
            for i, aid in enumerate(aids6):
                aids_temp[aid] += 0.05
                if i % 20 == 0:
                    aids_temp[aid] += 0.05
        if "top20b" in INFER:
            aids7 = list(itertools.chain(*[top_20b[aid][:5] for aid in unique_aids3 if aid in top_20b]))
            for i, aid in enumerate(aids7):
                aids_temp[aid] += 0.25
                if i % 5 == 0:
                    aids_temp[aid] += 0.25
            aids7 = list(itertools.chain(*[top_20b[aid][:5] for aid in unique_aids2 if aid in top_20b]))
            for i, aid in enumerate(aids7):
                aids_temp[aid] += 0.125
                if i % 5 == 0:
                    aids_temp[aid] += 0.125

        result, scores = _ranked(aids_temp, ITEM_CT)
        return _out(result, scores, top_orders, ITEM_CT)

    weights = [2, 2] + [1] * 8
    weights2 = [2, 2] + [1] * 53
    weights3 = [2, 2] + [1] * 18

    ln = len(unique_aids)

    aids_temp = Counter()
    if "top20orders" in INFER:
        aids2 = list(itertools.chain(*[top_20_orders[aid][:10] for aid in unique_aids if aid in top_20_orders]))
        w2 = weights * int(len(aids2) // 10)
        for i, (aid, w) in enumerate(zip(aids2, w2)):
            m = 0.25 + 0.75 * (ln - (i // 10)) / ln
            aids_temp[aid] += w * m
    if "top20buy2buy" in INFER:
        aids3 = list(itertools.chain(*[top_20_buy2buy[aid][:10] for aid in unique_buys if aid in top_20_buy2buy]))
        w3 = weights * int(len(aids3) // 10)
        for i, (aid, w) in enumerate(zip(aids3, w3)):
            aids_temp[aid] += w / 2
    if "top20test" in INFER:
        aids4 = list(itertools.chain(*[top_20_test[aid][:10] for aid in unique_aids if aid in top_20_test]))
        w4 = weights * int(len(aids4) // 10)
        for i, (aid, w) in enumerate(zip(aids4, w4)):
            m = 0.25 + 0.75 * (ln - (i // 10)) / ln
            aids_temp[aid] += w * m
    if "top20buy2buy2" in INFER:
        aids5 = list(itertools.chain(*[top_20_buy2buy2[aid][:10] for aid in unique_buys if aid in top_20_buy2buy2]))
        w5 = weights * int(len(aids5) // 10)
        for i, (aid, w) in enumerate(zip(aids5, w5)):
            aids_temp[aid] += w / 2
    if "top20c" in INFER:
        aids5 = list(itertools.chain(*[top_20c[aid][:55] for aid in unique_aids[:1] if aid in top_20c]))
        w5 = weights2 * int(len(aids5) // 55)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    # model-612
    if len(unique_aids) == 1:
        if "top20new2" in INFER:
            aids5 = list(itertools.chain(*[top_20_new2[aid][:20] for aid in unique_aids[-1:] if aid in top_20_new2]))
            w5 = weights3 * int(len(aids5) // 20)
            for aid, w in zip(aids5, w5):
                aids_temp[aid] += w
        if "top20new" in INFER:
            aids5 = list(itertools.chain(*[top_20_new[aid][:20] for aid in unique_aids[-1:] if aid in top_20_new]))
            w5 = weights3 * int(len(aids5) // 20)
            for aid, w in zip(aids5, w5):
                aids_temp[aid] += w

    if "top20d" in INFER:
        aids5 = list(itertools.chain(*[top_20d[aid][:20] for aid in unique_buys[:1] if aid in top_20d]))
        w5 = weights3 * int(len(aids5) // 20)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    # NEW
    ln2 = len(unique_aids5)
    if "top20buy" in INFER:
        aids5 = list(itertools.chain(*[top_20_buy[aid][:20] for aid in unique_aids5 if aid in top_20_buy]))
        w5 = weights3 * int(len(aids5) // 20)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += 2 * w / ln2
    if "top20f" in INFER:
        aids4 = list(itertools.chain(*[top_20f[aid][:5] for aid in unique_aids4 if aid in top_20f]))
        for i, aid in enumerate(aids4):
            w = i // 5
            aids_temp[aid] += 1 / 2 - w * 0.05
            if i % 5 == 0:
                aids_temp[aid] += 1 / 2 - w * 0.05
    if "top20e" in INFER:
        aids5 = list(itertools.chain(*[top_20e[aid][:55] for aid in unique_aids3 if aid in top_20e]))
        w5 = weights2 * int(len(aids5) // 55)
        for i, (aid, w) in enumerate(zip(aids5, w5)):
            w2 = i // 55
            aids_temp[aid] += w - w2 * 0.1
    if "top20e" in INFER:
        aids5 = list(itertools.chain(*[top_20e[aid][:10] for aid in unique_aids2 if aid in top_20e]))
        w5 = weights * int(len(aids5) // 10)
        for i, (aid, w) in enumerate(zip(aids5, w5)):
            w2 = i // 10
            aids_temp[aid] += w / 2.0 - w2 * 0.05

    sorted_aids, scores = _ranked(aids_temp, ITEM_CT, exclude=unique_aids)

    result = unique_aids + sorted_aids[: ITEM_CT - len(unique_aids)]
    scores = [-1] * len(unique_aids) + scores[: ITEM_CT - len(unique_aids)]
    return _out(result, scores, top_orders, ITEM_CT)


# ---------------------------------------------------------------------------
# model-620: suggest_aids on the make-valid-lists rows (clicks only)
# ---------------------------------------------------------------------------


def suggest_aids_620(df, M, ITEM_CT2=50, top_clicks=()):
    """model-620 `suggest_aids`. `df` is a make-valid-lists row
    [session, aids, types, ts, d, a, b] (see `valid_lists`). Returns (session, aids)."""
    top_clicks = list(top_clicks)
    top_20, top_20b, top_20c = M["top20"], M["top20b"], M["top20c"]
    top_20e, top_20f, top_20_test2 = M["top20e"], M["top20f"], M["top20test2"]
    top_20_new2 = M["top20new2"]

    session = df[0]
    aids = df[1]
    types = df[2]
    ds = df[4]
    ds2 = df[6]

    unique_aids = list(dict.fromkeys(aids[::-1]))
    unique_aids3 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds2[i] == 1][::-1]))
    mx = np.max(ds)
    unique_aids4 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds[i] == mx][::-1]))
    unique_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))  # noqa: F841

    ln = len(unique_aids)

    if len(unique_aids) >= 15:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        aids3 = list(itertools.chain(*[top_20c[aid][:20] for aid in unique_aids[:2] if aid in top_20c]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.6
        aids3 = list(itertools.chain(*[top_20b[aid][:15] for aid in unique_aids3 if aid in top_20b]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.3
        aids3 = list(itertools.chain(*[top_20_test2[aid][:20] for aid in unique_aids[:2] if aid in top_20_test2]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.6

        result = [k for k, v in aids_temp.most_common(ITEM_CT2) if k not in unique_aids]
        return session, (result + top_clicks[: ITEM_CT2 - len(result)])[:ITEM_CT2]

    aids_temp = Counter()

    weights3 = [2, 2] + [1] * 28
    if len(unique_aids) == 1:
        aids5 = list(itertools.chain(*[top_20_new2[aid][:30] for aid in unique_aids[-1:] if aid in top_20_new2]))
        w5 = weights3 * int(len(aids5) // 30)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    aids2 = list(itertools.chain(*[top_20[aid][:20] for aid in unique_aids if aid in top_20]))
    for i, aid in enumerate(aids2):
        m = 0.1 + 0.9 * (ln - (i // (20))) / ln
        aids_temp[aid] += m
        if i % (20) == 0:
            aids_temp[aid] += m

    aids3 = list(itertools.chain(*[top_20b[aid][:20] for aid in unique_aids[:2] if aid in top_20b]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (20) == 0:
            aids_temp[aid] += 1

    aids3 = list(itertools.chain(*[top_20_test2[aid][:20] for aid in unique_aids[:2] if aid in top_20_test2]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (20) == 0:
            aids_temp[aid] += 1

    aids4 = list(itertools.chain(*[top_20f[aid][:10] for aid in unique_aids4 if aid in top_20f]))
    for i, aid in enumerate(aids4):
        w = i // (10)
        aids_temp[aid] += 1 - w * 0.1
        if i % (10) == 0:
            aids_temp[aid] += 1 - w * 0.1

    aids5 = list(itertools.chain(*[top_20e[aid][:20] for aid in unique_aids3 if aid in top_20e]))
    for i, aid in enumerate(aids5):
        aids_temp[aid] += 1
        if i % (20) == 0:
            aids_temp[aid] += 1
    top_aids2 = [k for k, v in aids_temp.most_common(1) if k not in unique_aids]

    aids3 = list(itertools.chain(*[top_20c[aid][:10] for aid in top_aids2 if aid in top_20c]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (10) == 0:
            aids_temp[aid] += 1
    top_aids2 = [k for k, v in aids_temp.most_common(ITEM_CT2) if k not in unique_aids]

    result = top_aids2
    return session, (result + top_clicks[: ITEM_CT2 - len(result)])[:ITEM_CT2]


# ---------------------------------------------------------------------------
# model-709: suggest_aids / suggest_orders on make-valid-lists rows, 100 candidates
# ---------------------------------------------------------------------------
# Extra matrices: day = v155_d0 (top_40_day), day2 = v157_d0 (top_40_day2),
# less = v167, more = v168, less2 = v165, more2 = v166. `click_df` is model-620's output
# (session -> 50 aids, padding included); 709 uses its first ITEM_CT3 = 10.


def _sets_709(df):
    aids, types, tss, ds, ds1, ds2 = df[1], df[2], df[3], df[4], df[5], df[6]
    unique_aids = list(dict.fromkeys(aids[::-1]))
    mx = np.max(ds)
    unique_aids4 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds[i] == mx][::-1]))
    mx = np.max(tss)
    unique_aids5 = list(dict.fromkeys([f for i, f in enumerate(aids) if tss[i] >= mx - 60 * 60 / 2][::-1]))
    unique_aids6 = list(dict.fromkeys([f for i, f in enumerate(aids) if tss[i] >= mx - 60 * 60 * 24][::-1]))
    unique_aids2 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds1[i] == 1][::-1]))
    unique_aids3 = list(dict.fromkeys([f for i, f in enumerate(aids) if ds2[i] == 1][::-1]))
    unique_buys = list(dict.fromkeys([f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1]))
    return unique_aids, unique_aids2, unique_aids3, unique_aids4, unique_aids5, unique_aids6, unique_buys


def suggest_aids_709(df, M, click_df, ITEM_CT=100, ITEM_CT3=10, top_clicks=()):
    """model-709 `suggest_aids`. Returns (aids, scores) like the 564 ports (scores as
    model-709-wgt writes them)."""
    top_clicks = list(top_clicks)
    top_20, top_20b, top_20c = M["top20"], M["top20b"], M["top20c"]
    top_20e, top_20f, top_20_test2 = M["top20e"], M["top20f"], M["top20test2"]
    top_20_new2 = M["top20new2"]

    session = df[0]
    aids = df[1]
    types = df[2]

    top_day = M["day2"]
    click_aids = click_df[session][:ITEM_CT3]

    unique_aids, _, unique_aids3, unique_aids4, _, unique_aids6, _ = _sets_709(df)

    ln = len(unique_aids)  # noqa: F841 (used only by a rule the notebook commented out)

    if len(unique_aids) >= 15:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        aids3 = list(itertools.chain(*[top_20c[aid][: 20 * 2] for aid in unique_aids[:2] if aid in top_20c]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.6
        aids3 = list(itertools.chain(*[top_20b[aid][: 15 * 2] for aid in unique_aids3 if aid in top_20b]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.3
        aids3 = list(itertools.chain(*[top_20_test2[aid][: 20 * 2] for aid in unique_aids[:2] if aid in top_20_test2]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.6

        result, scores = _ranked(aids_temp, ITEM_CT)
        return _out(result, scores, top_clicks, ITEM_CT)

    aids_temp = Counter()

    # NEW
    MM = 4
    aids2 = list(itertools.chain(*[top_day[aid][: 10 * MM] for aid in unique_aids6 if aid in top_day]))
    for i, aid in enumerate(aids2):
        aids_temp[aid] += 1

    # NEW NEW
    aids2 = list(itertools.chain(*[top_20[aid][:20] for aid in click_aids if aid in top_20]))
    for i, aid in enumerate(aids2):
        aids_temp[aid] += 0.5

    weights3 = [2, 2] + [1] * 28
    if len(unique_aids) == 1:
        aids5 = list(itertools.chain(*[top_20_new2[aid][:30] for aid in unique_aids[-1:] if aid in top_20_new2]))
        w5 = weights3 * int(len(aids5) // 30)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    # FROM GIBA
    for i, a in enumerate(unique_aids):
        w0 = np.max([1 - (0.35 * i), 0.001])  # Weight aid order starting from the last one.
        if a in top_20:
            for j, aj in enumerate(top_20[a]):
                w1 = np.max([1 - (0.005 * j), 0.01])  # Weight the candidate aid from the dict
                aids_temp[aj] += w0 * w1

    aids3 = list(itertools.chain(*[top_20b[aid][: 20 * 2] for aid in unique_aids[:2] if aid in top_20b]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (20 * 2) == 0:
            aids_temp[aid] += 1

    aids3 = list(itertools.chain(*[top_20_test2[aid][: 20 * 2] for aid in unique_aids[:2] if aid in top_20_test2]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (20 * 2) == 0:
            aids_temp[aid] += 1

    aids4 = list(itertools.chain(*[top_20f[aid][: 10 * 2] for aid in unique_aids4 if aid in top_20f]))
    for i, aid in enumerate(aids4):
        w = i // (10 * 2)
        aids_temp[aid] += 1 - w * 0.1
        if i % (10 * 2) == 0:
            aids_temp[aid] += 1 - w * 0.1

    aids5 = list(itertools.chain(*[top_20e[aid][: 20 * 2] for aid in unique_aids3 if aid in top_20e]))
    for i, aid in enumerate(aids5):
        aids_temp[aid] += 1
        if i % (20 * 2) == 0:
            aids_temp[aid] += 1
    top_aids2 = [k for k, v in aids_temp.most_common(1) if k not in unique_aids]

    aids3 = list(itertools.chain(*[top_20c[aid][: 10 * 2] for aid in top_aids2 if aid in top_20c]))
    for i, aid in enumerate(aids3):
        aids_temp[aid] += 1
        if i % (10 * 2) == 0:
            aids_temp[aid] += 1
    top_aids2, scores = _ranked(aids_temp, ITEM_CT, exclude=unique_aids)

    result = unique_aids + top_aids2[: ITEM_CT - len(unique_aids)]
    scores = [-1] * len(unique_aids) + scores[: ITEM_CT - len(unique_aids)]
    return _out(result, scores, top_clicks, ITEM_CT)


def suggest_orders_709(df, M, click_df, ITEM_CT=100, ITEM_CT3=10, top_orders=()):
    """model-709 `suggest_orders` (also the carts list). Returns (aids, scores)."""
    top_orders = list(top_orders)
    top_20b, top_20c, top_20d = M["top20b"], M["top20c"], M["top20d"]
    top_20e, top_20f, top_20_orders = M["top20e"], M["top20f"], M["top20orders"]
    top_20_buy2buy, top_20_buy2buy2 = M["top20buy2buy"], M["top20buy2buy2"]
    top_20_test, top_20_buy = M["top20test"], M["top20buy"]
    top_20_new, top_20_new2 = M["top20new"], M["top20new2"]
    top_40_less, top_40_more, top_40_less2, top_40_more2 = M["less"], M["more"], M["less2"], M["more2"]

    session = df[0]
    aids = df[1]
    types = df[2]

    top_day = M["day"]
    click_aids = click_df[session][:ITEM_CT3]

    unique_aids, unique_aids2, unique_aids3, unique_aids4, unique_aids5, unique_aids6, unique_buys = _sets_709(df)

    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        for aid in unique_aids2:
            aids_temp[aid] += 0.5
        for aid in unique_aids3:
            aids_temp[aid] += 0.5

        aids3 = list(itertools.chain(*[top_20_buy2buy[aid][:40] for aid in unique_buys if aid in top_20_buy2buy]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.05
            if i % 40 == 0:
                aids_temp[aid] += 0.05
        aids3 = list(itertools.chain(*[top_20_buy2buy2[aid][:40] for aid in unique_buys if aid in top_20_buy2buy2]))
        for i, aid in enumerate(aids3):
            aids_temp[aid] += 0.1
            if i % 40 == 0:
                aids_temp[aid] += 0.1

        aids4 = list(itertools.chain(*[top_20_test[aid][:40] for aid in unique_aids if aid in top_20_test]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.05
            if i % 40 == 0:
                aids_temp[aid] += 0.05
        aids5 = list(itertools.chain(*[top_20c[aid][:20] for aid in unique_aids[:1] if aid in top_20c]))
        for i, aid in enumerate(aids5):
            aids_temp[aid] += 0.05
            if i % 20 == 0:
                aids_temp[aid] += 0.05
        aids6 = list(itertools.chain(*[top_20d[aid][:20] for aid in unique_buys[:1] if aid in top_20d]))
        for i, aid in enumerate(aids6):
            aids_temp[aid] += 0.05
            if i % 20 == 0:
                aids_temp[aid] += 0.05

        aids7 = list(itertools.chain(*[top_20b[aid][:5] for aid in unique_aids3 if aid in top_20b]))
        for i, aid in enumerate(aids7):
            aids_temp[aid] += 0.25
            if i % 5 == 0:
                aids_temp[aid] += 0.25
        aids7 = list(itertools.chain(*[top_20b[aid][:5] for aid in unique_aids2 if aid in top_20b]))
        for i, aid in enumerate(aids7):
            aids_temp[aid] += 0.125
            if i % 5 == 0:
                aids_temp[aid] += 0.125

        aids4 = list(itertools.chain(*[top_day[aid][:40] for aid in unique_aids6 if aid in top_day]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.05
            if i % 40 == 0:
                aids_temp[aid] += 0.05
        aids4 = list(itertools.chain(*[top_20_test[aid][:20] for aid in click_aids if aid in top_20_test]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.05
            if i % 20 == 0:
                aids_temp[aid] += 0.05
        aids4 = list(itertools.chain(*[top_20_buy[aid][:20] for aid in click_aids if aid in top_20_buy]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.05
            if i % 20 == 0:
                aids_temp[aid] += 0.05
        for aid in click_aids:
            aids_temp[aid] += 0.05

        # NEW STUFF
        aids4 = list(itertools.chain(*[top_40_more[aid][:5] for aid in unique_aids if aid in top_40_more]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.03
            if i % 5 == 0:
                aids_temp[aid] += 0.03
        aids4 = list(itertools.chain(*[top_40_more2[aid][:5] for aid in unique_aids if aid in top_40_more2]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.03
            if i % 5 == 0:
                aids_temp[aid] += 0.03
        aids4 = list(itertools.chain(*[top_40_less[aid][:5] for aid in unique_aids if aid in top_40_less]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.03
            if i % 5 == 0:
                aids_temp[aid] += 0.03
        aids4 = list(itertools.chain(*[top_40_less2[aid][:5] for aid in unique_aids if aid in top_40_less2]))
        for i, aid in enumerate(aids4):
            aids_temp[aid] += 0.03
            if i % 5 == 0:
                aids_temp[aid] += 0.03

        result, scores = _ranked(aids_temp, ITEM_CT)
        return _out(result, scores, top_orders, ITEM_CT)

    weights = [2, 2] + [1] * 8
    weights2 = [2, 2] + [1] * 53
    weights3 = [2, 2] + [1] * 18
    weights4 = [2, 2] + [1] * 38
    weights5 = [2, 2] + [1] * 28

    ln = len(unique_aids)

    MM = 3
    aids_temp = Counter()
    aids2 = list(itertools.chain(*[top_20_orders[aid][: 10 * MM] for aid in unique_aids if aid in top_20_orders]))
    w2 = weights5 * int(len(aids2) // (10 * MM))
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid][: 10 * MM] for aid in unique_buys if aid in top_20_buy2buy]))
    w3 = weights5 * int(len(aids3) // (10 * MM))
    aids4 = list(itertools.chain(*[top_20_test[aid][: 10 * MM] for aid in unique_aids if aid in top_20_test]))
    w4 = weights5 * int(len(aids4) // (10 * MM))
    aids5 = list(itertools.chain(*[top_20_buy2buy2[aid][: 10 * MM] for aid in unique_buys if aid in top_20_buy2buy2]))
    w5 = weights5 * int(len(aids5) // (10 * MM))
    for i, (aid, w) in enumerate(zip(aids2, w2)):
        m = 0.25 + 0.75 * (ln - (i // (10 * MM))) / ln
        aids_temp[aid] += w * m
    for i, (aid, w) in enumerate(zip(aids3, w3)):
        aids_temp[aid] += w / 2
    for i, (aid, w) in enumerate(zip(aids4, w4)):
        m = 0.25 + 0.75 * (ln - (i // (10 * MM))) / ln
        aids_temp[aid] += w * m
    for i, (aid, w) in enumerate(zip(aids5, w5)):
        aids_temp[aid] += w / 2

    for top_40_use in [top_40_more, top_40_less, top_40_more2, top_40_less2]:
        MM = 1
        aids2 = list(itertools.chain(*[top_40_use[aid][: 10 * MM] for aid in unique_aids if aid in top_40_use]))
        w2 = weights * int(len(aids2) // (10 * MM))
        for i, (aid, w) in enumerate(zip(aids2, w2)):
            m = 0.25 + 0.75 * (ln - (i // (10 * MM))) / ln
            aids_temp[aid] += w * m

    MM = 4
    aids2 = list(itertools.chain(*[top_day[aid][: 10 * MM] for aid in unique_aids6 if aid in top_day]))
    w2 = weights4 * int(len(aids2) // (10 * MM))
    for i, (aid, w) in enumerate(zip(aids2, w2)):
        aids_temp[aid] += 1

    ln0 = len(click_aids)
    aids4 = list(itertools.chain(*[top_20_test[aid][:20] for aid in click_aids if aid in top_20_test]))
    w4 = weights3 * int(len(aids4) // (20))
    for i, (aid, w) in enumerate(zip(aids4, w4)):
        m = 0.25 + 0.75 * (ln0 - (i // (20))) / ln0
        aids_temp[aid] += w * m
    aids4 = list(itertools.chain(*[top_20_buy[aid][:20] for aid in click_aids if aid in top_20_buy]))
    w4 = weights3 * int(len(aids4) // (20))
    for i, (aid, w) in enumerate(zip(aids4, w4)):
        m = 0.25 + 0.75 * (ln0 - (i // (20))) / ln0
        aids_temp[aid] += w * m
    for aid in click_aids:
        aids_temp[aid] += 1

    aids5 = list(itertools.chain(*[top_20c[aid][:55] for aid in unique_aids[:1] if aid in top_20c]))
    w5 = weights2 * int(len(aids5) // 55)
    for aid, w in zip(aids5, w5):
        aids_temp[aid] += w

    if len(unique_aids) == 1:
        aids5 = list(itertools.chain(*[top_20_new2[aid][:20] for aid in unique_aids[-1:] if aid in top_20_new2]))
        w5 = weights3 * int(len(aids5) // 20)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w
        aids5 = list(itertools.chain(*[top_20_new[aid][:20] for aid in unique_aids[-1:] if aid in top_20_new]))
        w5 = weights3 * int(len(aids5) // 20)
        for aid, w in zip(aids5, w5):
            aids_temp[aid] += w

    aids5 = list(itertools.chain(*[top_20d[aid][:20] for aid in unique_buys[:1] if aid in top_20d]))
    w5 = weights3 * int(len(aids5) // 20)
    for aid, w in zip(aids5, w5):
        aids_temp[aid] += w

    ln2 = len(unique_aids5)
    aids5 = list(itertools.chain(*[top_20_buy[aid][:20] for aid in unique_aids5 if aid in top_20_buy]))
    w5 = weights3 * int(len(aids5) // 20)
    for aid, w in zip(aids5, w5):
        aids_temp[aid] += 2 * w / ln2

    aids4 = list(itertools.chain(*[top_20f[aid][:5] for aid in unique_aids4 if aid in top_20f]))
    for i, aid in enumerate(aids4):
        w = i // 5
        aids_temp[aid] += 1 / 2 - w * 0.05
        if i % 5 == 0:
            aids_temp[aid] += 1 / 2 - w * 0.05
    aids5 = list(itertools.chain(*[top_20e[aid][:55] for aid in unique_aids3 if aid in top_20e]))
    w5 = weights2 * int(len(aids5) // 55)
    for i, (aid, w) in enumerate(zip(aids5, w5)):
        w2 = i // 55
        aids_temp[aid] += w - w2 * 0.1
    aids5 = list(itertools.chain(*[top_20e[aid][:10] for aid in unique_aids2 if aid in top_20e]))
    w5 = weights * int(len(aids5) // 10)
    for i, (aid, w) in enumerate(zip(aids5, w5)):
        w2 = i // 10
        aids_temp[aid] += w / 2.0 - w2 * 0.05

    sorted_aids, scores = _ranked(aids_temp, ITEM_CT, exclude=unique_aids)

    result = unique_aids + sorted_aids[: ITEM_CT - len(unique_aids)]
    scores = [-1] * len(unique_aids) + scores[: ITEM_CT - len(unique_aids)]
    return _out(result, scores, top_orders, ITEM_CT)
