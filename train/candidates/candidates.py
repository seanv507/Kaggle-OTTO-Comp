"""Candidate generation from Chris Deotte's part of the 3rd place OTTO solution.

Converts the rule-based `suggest_aids` / `suggest_orders` notebooks in train/candidates/
(model-564-*, model-612/614/620/709, run by script.ipynb). The notebooks loop over
sessions in Python, summing covisitation neighbours of a session's aids into a
`Counter`; here each rule is a join of a session's "source" aids with a covisitation
matrix.

Conventions:
- Events must be sorted by session, then ts (as data/*.parquet are). Row order stands in
  for the reference's sort order; `pos` is an event's row number within its session.
- Sub-sessions (`d`): a new one starts after a gap of more than 2 hours, as in
  make-valid-with-d.ipynb.
- Ties in ts: the reference finds the last event of each sub-session with
  `sort_values('ts', ascending=False).drop_duplicates('d')` (model-564) or a descending
  sort + cumcount (make-valid-lists, flag `b`). For tied timestamps both keep the
  earliest of the tied rows (pandas' stable sort keeps original order). Pandas' default
  quicksort is only stable for short arrays, so on sessions longer than 16 events the
  564 notebooks' choice among ties was arbitrary; here it is always the earliest row.
"""

from typing import Literal

import polars as pl

from event_types import CARTS, ORDERS

Events = pl.LazyFrame | pl.DataFrame

SUBSESSION_GAP = pl.duration(hours=2)


def add_subsessions(events: Events) -> pl.LazyFrame:
    """Add `pos`, `d`, `a` and `b` to session-sorted events.

    - `pos`: row number within the session (0-based), the reference's list position.
    - `d`: sub-session number; increments when ts - previous ts > 2 hours.
    - `a`: 1 on the first event of each sub-session.
    - `b`: 1 on the last event of each sub-session: the earliest row with the
      sub-session's latest ts (see the module docstring on ties).
    """
    # global row index minus the session's first: the GPU engine has no int_range in .over()
    row = pl.col("_row") - pl.col("_row").min().over("session")
    gap = pl.col("ts").diff().over("session") > SUBSESSION_GAP
    return (
        events.lazy()
        .with_row_index("_row")
        .with_columns(
            pos=row.cast(pl.UInt32),
            d=gap.fill_null(False).cast(pl.UInt16).cum_sum().over("session").cast(pl.UInt16),
        )
        .drop("_row")
        .with_columns(_at_last_ts=pl.col("ts") == pl.col("ts").max().over("session", "d"))
        .with_columns(
            a=(pl.col("pos") == pl.col("pos").min().over("session", "d")).cast(pl.UInt8),
            # (not .filter() inside .over(): the GPU engine rejects it)
            b=(
                pl.col("pos")
                == pl.when("_at_last_ts").then("pos").min().over("session", "d")
            ).cast(pl.UInt8),
        )
        .drop("_at_last_ts")
    )


# --- session source sets ---
# Each is the reference's `list(dict.fromkeys(...))` of a session's aids: a filter on the
# events, then either "recent" (the reversed list: most recent occurrence first) or
# "oldest" (the list in order: first occurrence first).
Order = Literal["recent", "oldest"]

SOURCE_SETS: dict[str, tuple[pl.Expr, Order]] = {
    # unique_aids
    "all": (pl.lit(True), "recent"),
    # unique_aids2: first event of each sub-session, latest sub-session first
    "first_of_d": (pl.col("a") == 1, "recent"),
    # unique_aids3 in suggest_orders and model-620: last event of each sub-session,
    # latest sub-session first
    "last_of_d": (pl.col("b") == 1, "recent"),
    # unique_aids3 in the 564/612/614 suggest_aids: the same aids, earliest sub-session first
    "last_of_d_oldest": (pl.col("b") == 1, "oldest"),
    # unique_aids4: events of the last sub-session
    "last_d": (pl.col("d") == pl.col("d").max().over("session"), "recent"),
    # unique_aids5: events within 30 minutes of the session's last event
    "last_30min": (
        pl.col("ts").max().over("session") - pl.col("ts") <= pl.duration(minutes=30),
        "recent",
    ),
    # unique_buys: carts and orders
    "buys": (pl.col("type").is_in([CARTS, ORDERS]), "recent"),
}


def source_set(events: pl.LazyFrame, name: str) -> pl.LazyFrame:
    """One source set as `session, aid, src_idx`: src_idx is the aid's index in the
    reference's list (0 = first). `events` must have gone through `add_subsessions`."""
    keep, order = SOURCE_SETS[name]
    recent = order == "recent"
    key = pl.col("pos").max() if recent else pl.col("pos").min()
    return (
        events.filter(keep)
        .group_by("session", "aid")
        .agg(key.alias("_key"))
        .with_columns(
            src_idx=(pl.col("_key").rank("ordinal", descending=recent).over("session") - 1).cast(pl.UInt32)
        )
        .drop("_key")
        .sort("session", "src_idx")
    )


# ---------------------------------------------------------------------------
# Covisitation rules
# ---------------------------------------------------------------------------
# A rule is one of the notebooks' loops:
#     aids3 = list(itertools.chain(*[M[aid][:k] for aid in SOURCE[:n] if aid in M]))
#     for i, aid in enumerate(aids3):       # or zip(aids3, [2,2]+[1]*(k-2) repeated)
#         aids_temp[aid] += weight(i)
#         if i % k == 0: aids_temp[aid] += weight(i)   # "first neighbour" bonus
# `i` is the position in the concatenated neighbour lists, so a neighbour list shorter
# than k shifts i // k and i % k for the following source aids; this is reproduced.
#
# Scores must match the reference bit for bit, because ties are broken by insertion
# order: each += is kept as its own contribution, ordered by `order` (rule, i, bonus),
# and summed sequentially (cum_sum) in that order, as the Counter does.

RULE_STRIDE = 2**33  # order = rule number * RULE_STRIDE + 2 * i + bonus

# columns available to weight expressions
IX = pl.col("i")  # position in the rule's concatenated neighbour lists
LN = pl.col("ln")  # len(unique_aids)
LN2 = pl.col("ln2")  # len(unique_aids5)


def pattern(k: int) -> pl.Expr:
    """`[2,2] + [1]*(k-2)` repeated: the notebooks' weights/weights2/weights3 lists."""
    return pl.when(IX % k < 2).then(2).otherwise(1)


class Rule:
    """One covisitation loop. `include` names it for the notebooks' INFER lists."""

    def __init__(
        self,
        include: str,
        source: str,
        k: int,
        weight: pl.Expr | float,
        *,
        n_sources: int | None = None,
        bonus: bool = False,
        truncate: bool = False,
        matrix: str | None = None,
    ):
        self.include = include
        self.matrix = matrix or include
        self.source = source  # a SOURCE_SETS name, or TOP1 (see below)
        self.k = k
        self.weight = weight if isinstance(weight, pl.Expr) else pl.lit(float(weight))
        self.n_sources = n_sources  # SOURCE[:n]
        self.bonus = bonus  # if i % k == 0: add the weight again
        self.truncate = truncate  # zip with a pattern list: keep i < k * (n // k)


# source for model-564 suggest_aids' second-order rule:
# `top_aids2 = [k for k,v in aids_temp.most_common(1) if k not in unique_aids]`
TOP1 = "top1_new"


def neighbours(matrix: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """`aid, aid_y, rank` from a covisitation matrix sorted by aid, value descending,
    aid_y (the `save_covisits.py` output): rank 0 is the reference's M[aid][0]."""
    return (
        matrix.lazy()
        .select("aid", "aid_y")
        .with_row_index("_row")
        .with_columns(rank=(pl.col("_row") - pl.col("_row").min().over("aid")).cast(pl.UInt32))
        .drop("_row")
    )


def rule_contributions(
    sources: pl.LazyFrame, nbrs: pl.LazyFrame, rule: Rule, rule_no: int, session_info: pl.LazyFrame
) -> pl.LazyFrame:
    """Contributions `session, aid, score, order` of one rule.

    `sources`: session, aid, src_idx. `session_info`: session, ln, ln2."""
    if rule.n_sources is not None:
        sources = sources.filter(pl.col("src_idx") < rule.n_sources)
    key = pl.col("src_idx").cast(pl.Int64) * 2**20 + pl.col("rank")
    rows = (
        sources.join(nbrs.filter(pl.col("rank") < rule.k), on="aid")
        .with_columns(
            i=(key.rank("ordinal").over("session") - 1).cast(pl.Int64),
            n=pl.len().over("session"),
        )
        .join(session_info, on="session", how="left")
    )
    if rule.truncate:
        rows = rows.filter(pl.col("i") < rule.k * (pl.col("n") // rule.k))
    rows = rows.select(
        "session",
        pl.col("aid_y").alias("aid"),
        score=rule.weight.cast(pl.Float64),
        order=rule_no * RULE_STRIDE + 2 * pl.col("i"),
        _bonus=(pl.col("i") % rule.k == 0) if rule.bonus else pl.lit(False),
    )
    bonus = rows.filter("_bonus").with_columns(order=pl.col("order") + 1)
    return pl.concat([rows, bonus]).drop("_bonus")


def total_scores(contribs: pl.LazyFrame) -> pl.LazyFrame:
    """`session, aid, score, first`: sum in `order` (sequentially, as += does) and the
    first `order`, the Counter's insertion order."""
    return contribs.group_by("session", "aid").agg(
        score=pl.col("score").sort_by("order").cum_sum().last(),
        first=pl.col("order").min(),
    )


def rank_scores(totals: pl.LazyFrame) -> pl.LazyFrame:
    """Sort by session, then Counter.most_common order (score descending, then insertion
    order), and add `rnk` (0 = best) within the session."""
    return (
        totals.sort(["session", "score", "first"], descending=[False, True, False])
        .with_row_index("_row")
        .with_columns(rnk=(pl.col("_row") - pl.col("_row").min().over("session")).cast(pl.UInt32))
        .drop("_row")
    )


# ---------------------------------------------------------------------------
# History scores: the >= 20 aids branch
# ---------------------------------------------------------------------------
TYPE_WEIGHT = {0: 1, 1: 6, 2: 3}  # type_weight_multipliers


def logspace_weights(start: float, stop: float = 1.0) -> pl.Expr:
    """`np.logspace(start, stop, len(aids), base=2, endpoint=True) - 1` at `pos`,
    computed in the order numpy's linspace does (step = delta / (n - 1);
    y = pos * step + start; last y = stop), so the floats match."""
    n = pl.len().over("session")
    delta = stop - start
    y = pl.col("pos").cast(pl.Float64) * (delta / (n - 1).cast(pl.Float64)) + start
    y = (
        pl.when(n == 1).then(pl.lit(start))  # numpy: num=1 gives [start]
        .when(pl.col("pos") == n - 1).then(pl.lit(stop))
        .otherwise(y)
    )
    return pl.lit(2.0).pow(y) - 1


def history_contributions(events: pl.LazyFrame, start: float, rule_no: int = 0) -> pl.LazyFrame:
    """`aids_temp[aid] += w * type_weight_multipliers[t]` over the session's events."""
    # (an explicit default: the GPU engine rejects replace_strict without one; types are 0-2)
    mult = pl.col("type").replace_strict(TYPE_WEIGHT, default=None, return_dtype=pl.Float64)
    return events.select(
        "session",
        "aid",
        score=logspace_weights(start) * mult,
        order=rule_no * RULE_STRIDE + 2 * pl.col("pos").cast(pl.Int64),
    )


def constant_contributions(sources: pl.LazyFrame, value: float, rule_no: int) -> pl.LazyFrame:
    """`for aid in SOURCE: aids_temp[aid] += value`."""
    return sources.select(
        "session",
        "aid",
        score=pl.lit(value, pl.Float64),
        order=rule_no * RULE_STRIDE + 2 * pl.col("src_idx").cast(pl.Int64),
    )


# ---------------------------------------------------------------------------
# model-564: suggest_aids (clicks) and suggest_orders
# ---------------------------------------------------------------------------
# Rule lists in the notebooks' order. `include` values are the INFER names, which are
# also the keys of the `matrices` dict: top20 = v93, top20b = v217, top20c = v220,
# top20d = v226, top20e = v232, top20f = v235, top20orders = v95, top20buy2buy = v90,
# top20buy2buy2 = v99, top20test = v116, top20test2 = v115, top20buy = v239.

M_ORDERS = 0.25 + 0.75 * (LN - IX // 10) / LN  # m in suggest_orders

ORDERS_HISTORY_564 = [
    Rule("top20buy2buy", "buys", 40, 0.05, bonus=True),
    Rule("top20buy2buy2", "buys", 40, 0.1, bonus=True),
    Rule("top20test", "all", 40, 0.05, bonus=True),
    Rule("top20c", "all", 20, 0.05, n_sources=1, bonus=True),
    Rule("top20d", "buys", 20, 0.05, n_sources=1, bonus=True),
    Rule("top20b", "last_of_d", 5, 0.25, bonus=True),
    Rule("top20b", "first_of_d", 5, 0.125, bonus=True),
]

ORDERS_COVISIT_564 = [
    Rule("top20orders", "all", 10, pattern(10) * M_ORDERS, truncate=True),
    Rule("top20buy2buy", "buys", 10, pattern(10) / 2, truncate=True),
    Rule("top20test", "all", 10, pattern(10) * M_ORDERS, truncate=True),
    Rule("top20buy2buy2", "buys", 10, pattern(10) / 2, truncate=True),
    Rule("top20c", "all", 55, pattern(55), n_sources=1, truncate=True),
    Rule("top20d", "buys", 20, pattern(20), n_sources=1, truncate=True),
    Rule("top20buy", "last_30min", 20, 2 * pattern(20) / LN2, truncate=True),
    Rule("top20f", "last_d", 5, 0.5 - (IX // 5) * 0.05, bonus=True),
    Rule("top20e", "last_of_d", 55, pattern(55) - (IX // 55) * 0.1, truncate=True),
    Rule("top20e", "first_of_d", 10, pattern(10) / 2.0 - (IX // 10) * 0.05, truncate=True),
]

AIDS_HISTORY_564 = [
    Rule("top20c", "all", 20, 0.6, n_sources=2),
    Rule("top20b", "last_of_d_oldest", 15, 0.3),
    Rule("top20test2", "all", 20, 0.6, n_sources=2),
]

AIDS_COVISIT_564 = [
    Rule("top20", "all", 20, 0.1 + 0.9 * (LN - IX // 20) / LN, bonus=True),
    Rule("top20b", "all", 20, 1, n_sources=2, bonus=True),
    Rule("top20test2", "all", 20, 1, n_sources=2, bonus=True),
    Rule("top20f", "last_d", 10, 1 - (IX // 10) * 0.1, bonus=True),
    Rule("top20e", "last_of_d_oldest", 20, 1, bonus=True),
    Rule("top20c", TOP1, 10, 1, bonus=True),  # neighbours of the best new aid so far
]

ALL_564 = sorted({r.include for r in ORDERS_HISTORY_564 + ORDERS_COVISIT_564 + AIDS_HISTORY_564 + AIDS_COVISIT_564})

Branch = Literal["auto", "history", "covisit"]
Target = Literal["clicks", "orders"]


def _session_info(events: pl.LazyFrame, all_aids: pl.LazyFrame, last_30min: pl.LazyFrame) -> pl.LazyFrame:
    ln = all_aids.group_by("session").agg(ln=pl.len())
    ln2 = last_30min.group_by("session").agg(ln2=pl.len())
    return ln.join(ln2, on="session", how="left")


def _top1_new(contribs: list[pl.LazyFrame], all_aids: pl.LazyFrame) -> pl.LazyFrame:
    """The reference's `[k for k,v in aids_temp.most_common(1) if k not in unique_aids]`
    as a source set (empty if the best aid is a history aid)."""
    best = rank_scores(total_scores(pl.concat(contribs))).filter(pl.col("rnk") == 0)
    return best.join(all_aids, on=["session", "aid"], how="anti").select(
        "session", "aid", src_idx=pl.lit(0, pl.UInt32)
    )


def _contributions(
    events: pl.LazyFrame,
    matrices: dict[str, pl.LazyFrame],
    history: list[pl.LazyFrame],
    rules: list[Rule],
    include: set[str],
    first_rule_no: int,
) -> list[pl.LazyFrame]:
    """`history` contributions followed by those of the included `rules`."""
    sets: dict[str, pl.LazyFrame] = {}

    def source(name: str) -> pl.LazyFrame:
        if name not in sets:
            sets[name] = source_set(events, name)
        return sets[name]

    info = _session_info(events, source("all"), source("last_30min"))
    contribs = list(history)
    for no, rule in enumerate(rules, start=first_rule_no):
        if rule.include not in include:
            continue
        if rule.source == TOP1:
            if "top20e" not in include:
                # the notebooks only compute top_aids2 under `if 'top20e' in INFER`
                raise ValueError("the top20c second-order rule needs top20e")
            src = _top1_new(contribs, source("all"))
        else:
            src = source(rule.source)
        contribs.append(rule_contributions(src, neighbours(matrices[rule.matrix]), rule, no, info))
    return contribs


def _history_branch(n_unique: pl.Expr, branch: Branch) -> pl.Expr:
    return {"auto": n_unique >= 20, "history": pl.lit(True), "covisit": pl.lit(False)}[branch]


def suggest_564(
    events: Events,
    matrices: dict[str, pl.LazyFrame | pl.DataFrame],
    target: Target,
    include: list[str] | None = None,
    branch: Branch = "auto",
    item_ct: int = 50,
    popular: list[int] | None = None,
) -> pl.LazyFrame:
    """model-564 `suggest_aids` (target="clicks") or `suggest_orders` (target="orders").

    `include` is the notebook's INFER list (None: all rules, as model-564-F). Returns
    `session, pos, aid, score`: item_ct rows per session, where score is what the -wgt
    notebook writes (-1 for history aids placed first in the covisit branch, -2 for
    `popular` padding).
    """
    include_set = set(ALL_564 if include is None else include)
    matrices = {k: v.lazy() for k, v in matrices.items()}
    events = add_subsessions(events)
    all_aids = source_set(events, "all")
    in_history = (
        all_aids.group_by("session")
        .agg(_hist=_history_branch(pl.len(), branch))
        .select("session", "_hist")
    )
    events = events.join(in_history, on="session")
    hist_events = events.filter("_hist").drop("_hist")
    cov_events = events.filter(~pl.col("_hist")).drop("_hist")

    if target == "orders":
        hist_rules, cov_rules, start = ORDERS_HISTORY_564, ORDERS_COVISIT_564, 0.5
        hist_extra = [
            constant_contributions(source_set(hist_events, "first_of_d"), 0.5, rule_no=1),
            constant_contributions(source_set(hist_events, "last_of_d"), 0.5, rule_no=2),
        ]
        first_rule = 3
    else:
        hist_rules, cov_rules, start = AIDS_HISTORY_564, AIDS_COVISIT_564, 0.1
        hist_extra = []
        first_rule = 1

    hist = rank_scores(total_scores(pl.concat(_contributions(
        hist_events, matrices,
        [history_contributions(hist_events, start, rule_no=0), *hist_extra],
        hist_rules, include_set, first_rule,
    ))))
    hist_out = hist.filter(pl.col("rnk") < item_ct).select("session", pos=pl.col("rnk"), aid="aid", score="score")

    cov_aids = source_set(cov_events, "all")
    cov_contribs = _contributions(cov_events, matrices, [], cov_rules, include_set, first_rule_no=0)
    cov_new = (
        rank_scores(
            total_scores(pl.concat(cov_contribs, how="vertical"))
            .join(cov_aids, on=["session", "aid"], how="anti")
        )
        if cov_contribs
        else pl.LazyFrame(schema={"session": pl.Int32, "aid": pl.Int32, "score": pl.Float64, "rnk": pl.UInt32})
    )
    n_hist = cov_aids.group_by("session").agg(n_hist=pl.len().cast(pl.UInt32))
    cov_out = pl.concat([
        cov_aids.select("session", pos=pl.col("src_idx"), aid="aid", score=pl.lit(-1.0)),
        cov_new.join(n_hist, on="session").select(
            "session", pos=pl.col("n_hist") + pl.col("rnk"), aid="aid", score="score"
        ),
    ]).filter(pl.col("pos") < item_ct)

    out = pl.concat([hist_out, cov_out])
    return _pad(out, popular or [], item_ct).sort("session", "pos")


def _pad(out: pl.LazyFrame, popular: list[int], item_ct: int) -> pl.LazyFrame:
    """`(result + top_x[:ITEM_CT-len(result)])[:ITEM_CT]`, scores -2 (top_x = [-2]*100)."""
    if not popular:
        return out
    counts = out.group_by("session").agg(n=pl.len().cast(pl.UInt32))
    pad = pl.LazyFrame(
        {"j": list(range(len(popular))), "aid": popular, "_k": [0] * len(popular)},
        schema={"j": pl.UInt32, "aid": pl.Int32, "_k": pl.UInt8},
    )
    padding = (
        # an equi-join on a constant, not how="cross": the GPU engine has no nested loop join
        counts.with_columns(_k=pl.lit(0, pl.UInt8)).join(pad, on="_k")
        .filter(pl.col("n") + pl.col("j") < item_ct)
        .select("session", pos=pl.col("n") + pl.col("j"), aid="aid", score=pl.lit(-2.0))
    )
    return pl.concat([out, padding])


# model-564 notebooks: W_VER -> (target, INFER, branch). model-564-F (all rules, history
# branch) runs both targets; the -wgt notebooks are the same runs' scores.
RUNS_564: dict[str, tuple[Target, list[str] | None, Branch]] = {
    "1": ("orders", ["top20orders"], "auto"),
    "3": ("orders", ["top20test"], "auto"),
    "4": ("orders", ["top20buy2buy", "top20buy2buy2"], "auto"),
    "5": ("orders", ["top20b", "top20c"], "auto"),
    "6": ("orders", ["top20e", "top20f"], "auto"),
    "7": ("orders", ["top20d", "top20buy"], "auto"),
    "20": ("orders", ["top20d", "top20buy"], "history"),
    "21": ("orders", ["top20test"], "history"),
    "22": ("orders", ["top20buy2buy", "top20buy2buy2"], "history"),
    "23": ("orders", ["top20b", "top20c"], "history"),
    "24": ("orders", ["top20d", "top20buy"], "covisit"),
    "25": ("orders", ["top20test"], "covisit"),
    "26": ("orders", ["top20buy2buy", "top20buy2buy2"], "covisit"),
    "27": ("orders", ["top20b", "top20c"], "covisit"),
    "31": ("orders", ["top20orders"], "covisit"),
    "32": ("orders", ["top20e", "top20f"], "covisit"),
    "33": ("clicks", ["top20"], "covisit"),
    "F_clicks": ("clicks", None, "history"),
    "F_orders": ("orders", None, "history"),
}
