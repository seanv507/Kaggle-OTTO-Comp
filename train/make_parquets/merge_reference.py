"""Pandas ports of Chris Deotte's Make-152/172/174 notebooks (steps A-G), used by
test_merge.py.

Transcribed as literally as possible from train/make_parquets/*.ipynb (cuDF -> pandas,
cupy -> numpy). Deliberate changes:
- Candidates come from save_candidates.py's long files (session, pos, aid, score) instead
  of the submission CSV pairs, so the notebooks' explode / np.tile(np.arange(100)) /
  merge-on-pos is already done: `cands[run]` is that file with session -> user,
  aid -> item. Feature files have their key as a column instead of the index.
- `type` is an integer code (event_types), not a string; ts is epoch seconds.
- Make-172-B's paths '.../../data/candidate_scores/submission_v564_F' (three dots) are
  read as the '../../' they are next to.
- Sorts are stable, so tied ts keep their input order.
- 152 is not split into parts A/B (the split only saved memory).
Step H (the clicks ranker's out-of-fold predictions) needs the trained rankers and is
not here.
"""

import numpy as np

from event_types import CARTS, CLICKS, ORDERS

TARGETS = {"clicks": CLICKS, "carts": CARTS, "orders": ORDERS}


def _f32_ints(df):
    for c in df.columns:
        if str(df[c].dtype) == "int32":
            df[c] = df[c].astype("float32")
    return df


def _final_types(users, int64="int32"):
    """The closing dtype loop: B, F, G cast int64 -> int32; D, E cast it -> float32."""
    for c in users.columns:
        t = str(users[c].dtype)
        if t == "int64":
            users[c] = users[c].astype(int64)
        if t in ("float64", "Int32", "Float32"):
            users[c] = users[c].astype("float32")
    return users


def _merge_feature(users, feats, name, on):
    df = _f32_ints(feats[name].set_index("aid" if on in ("item", "last") else "session"))
    return users.merge(df, left_on=on, right_index=True, how="left")


# --- A: candidates of model 709 + target ------------------------------------------------

def step_a(cands, labels, use, keep_positive_users):
    df = cands[f"709_{'orders' if use == 'carts' else use}"]
    users = df[["user", "pos", "item"]].copy()
    users["wgt"] = df["score"].astype("float32")
    test_labels = labels.loc[labels["type"] == TARGETS[use]]
    tmp = test_labels.ground_truth.explode().dropna().astype("int32").rename("item")
    test_users = test_labels[["session"]].astype("int32").copy()
    test_users.columns = ["user"]
    test_users = test_users.merge(tmp, left_index=True, right_index=True)
    test_users[use] = np.int8(1)
    users = users[["user", "item", "pos", "wgt"]]
    users = users.merge(test_users, on=["user", "item"], how="left").fillna(0)
    users[use] = users[use].astype("int8")
    if keep_positive_users:  # Make-172-A / 174-A
        tmp = users.groupby("user")[use].agg("sum")
        keep = tmp.loc[tmp > 0].index.values
        users = users.loc[users.user.isin(keep)].reset_index(drop=True)
    return users


# --- B: candidate scores of the other models + item/user features -----------------------

def score_sources(use):
    """(score column, run) in the order Make-152-B / 172-B / 174-B merge them."""
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


def step_b(users, cands, feats, use):
    for col, run in score_sources(use):
        dfA = cands[run][["user", "item", "score"]].rename(columns={"score": col})
        dfA[col] = dfA[col].astype("float32")
        dfA = dfA.loc[dfA[col] != -2]
        users = users.merge(dfA, on=["user", "item"], how="left").fillna(-3)
    for name in ["item12", "item13", "item4", "item10"]:
        users = _merge_feature(users, feats, name, "item")
    users = _merge_feature(users, feats, "user4", "user")
    return _final_types(users)


# --- C: user7 and sort -----------------------------------------------------------------

def step_c(users, feats):
    users = users.merge(feats["user7"].set_index("session"), left_on="user", right_index=True, how="left")
    return users.sort_values(["user", "pos"], kind="stable").reset_index(drop=True)


# --- D: embedding similarity to the last aid and the first 5 buys ----------------------

def _normalize(embed):
    z = np.sqrt(np.sum(embed * embed, axis=1))
    return embed / np.reshape(z, (-1, 1))


def step_d(users, events, embed):
    df = events.sort_values(["session", "ts"], kind="stable").reset_index(drop=True)
    tmp = df.groupby("session").aid.agg("last").rename("last")
    users = users.merge(tmp, left_on="user", right_index=True, how="left")
    x = users.item.values
    y = users["last"].values
    users = users.drop(["last"], axis=1)
    embed = _normalize(embed)
    users["z1"] = np.sum(embed[x,] * embed[y,], axis=1)
    users2 = users[["user", "item"]].copy()
    for k in range(5):
        tmp = df.loc[df["type"] != 0].reset_index(drop=True)
        tmp["n"] = tmp.groupby("session").aid.cumcount()
        tmp = tmp.loc[tmp.n == k]
        tmp = tmp[["session", "aid"]]
        tmp.columns = ["user", "last_buy"]
        users2 = users2.merge(tmp, on="user", how="left")
        has = users2.last_buy.notnull()
        x = users2.loc[has].item.to_numpy()
        y = users2.loc[has]["last_buy"].to_numpy().astype("int32")
        z = np.sum(embed[x,] * embed[y,], axis=1)
        users2[f"z{k + 2}"] = -1.0
        users2.loc[has, f"z{k + 2}"] = z
        users2 = users2.drop(["last_buy"], axis=1)
    users = users.merge(users2.drop_duplicates(["user", "item"]), on=["user", "item"], how="left")
    for c in ["z1"] + [f"z{x + 2}" for x in range(5)]:
        users[c] = users[c].astype("float32")
    return _final_types(users, int64="float32")


# --- E: more item/user features ----------------------------------------------------------

def step_e(users, feats):
    for name in ["item20", "item21", "item22"]:
        users = _merge_feature(users, feats, name, "item")
    users = _merge_feature(users, feats, "user10", "user")
    return _final_types(users, int64="float32")


# --- F (172/174): session-level similarities, buy features of the item and last aid ------

def step_f(users, events, embed, feats):
    df = events.sort_values(["session", "ts"], kind="stable")
    last = df.groupby("session").agg("last").reset_index()[["session", "aid"]]
    last.columns = ["user", "last"]
    users2 = users[["user", "item"]].copy()
    users2 = users2.merge(last, on="user", how="left")
    first = df.groupby("session").agg("first").reset_index()[["session", "aid"]]
    first.columns = ["user", "first"]
    users2 = users2.merge(first, on="user", how="left")
    df = events.sort_values(["session", "ts"], kind="stable").reset_index(drop=True)
    df["mx"] = df.groupby("session").ts.transform("max")
    df["mx"] = df.mx - df.ts
    df = df.loc[df.mx > 60 * 60 * 24]
    last2 = df.groupby("session").agg("last").reset_index()[["session", "aid"]]
    last2.columns = ["user", "last2"]
    users2 = users2.merge(last2, on="user", how="left")
    users2["last2"] = users2["last2"].fillna(users2["last"]).astype("int32")
    embed = _normalize(embed)
    e2 = embed[users2["last"].values,]
    users2["z10"] = np.sum(embed[users2["first"].values,] * e2, axis=1)
    users2["z11"] = np.sum(embed[users2["last2"].values,] * e2, axis=1)
    users2 = _merge_feature(users2, feats, "item40", "item")
    users2 = _merge_feature(users2, feats, "item41", "item")
    df = _f32_ints(feats["item41"].set_index("aid"))
    df.columns = [x.replace("12", "13") for x in df.columns]
    users2 = users2.merge(df, left_on="last", right_index=True, how="left")
    users2 = _merge_feature(users2, feats, "user20", "user")
    users2 = _merge_feature(users2, feats, "user21", "user")
    for c in users2.columns[5:]:
        users[c] = users2[c].values
    return _final_types(users)


# --- G (172/174): similarity to the last/first 5 aids and last 5 buys --------------------

def step_g(users, events, emb):
    users2 = users[["user", "item"]].copy()
    df = events.sort_values(["session", "ts"], ascending=[True, False], kind="stable").reset_index(drop=True)
    df["x"] = df.groupby("session").cumcount()
    for k in range(5):
        tmp = df.loc[df.x == k][["session", "aid"]]
        tmp.columns = ["user", f"last_{k}"]
        users2 = users2.merge(tmp, on="user", how="left")
        users2[f"last_{k}"] = users2[f"last_{k}"].fillna(-1)
    df = df.sort_values(["session", "ts"], ascending=[True, True], kind="stable").reset_index(drop=True)
    df["x"] = df.groupby("session").cumcount()
    for k in range(5):
        tmp = df.loc[df.x == k][["session", "aid"]]
        tmp.columns = ["user", f"first_{k}"]
        users2 = users2.merge(tmp, on="user", how="left")
        users2[f"first_{k}"] = users2[f"first_{k}"].fillna(-1)
    df = df.loc[df["type"] > 0]
    df = df.sort_values(["session", "ts"], ascending=[True, False], kind="stable").reset_index(drop=True)
    df["x"] = df.groupby("session").cumcount()
    for k in range(5):
        tmp = df.loc[df.x == k][["session", "aid"]]
        tmp.columns = ["user", f"buy_{k}"]
        users2 = users2.merge(tmp, on="user", how="left")
        users2[f"buy_{k}"] = users2[f"buy_{k}"].fillna(-1)
    emb = _normalize(emb)

    def build_dist(a0, a1):
        a0 = np.asarray(a0, dtype=np.int64)
        a1 = np.asarray(a1, dtype=np.int64)
        return np.sum(emb[np.clip(a0, 0, None)] * emb[np.clip(a1, 0, None)], 1)

    cols1 = [f"last_{x}" for x in range(5)]
    cols2 = [f"first_{x}" for x in range(5)]
    cols3 = [f"buy_{x}" for x in range(5)]
    cols = cols1 + cols2 + cols3
    for nm in cols:
        mask = users2[nm] != -1
        x = build_dist(users2.item.values, users2[nm].values)
        users2[nm] = users2[nm].astype("float64")
        users2.loc[mask, nm] = x[mask.to_numpy()]
    for c in cols:
        users2.loc[users2[c] == -1, c] = np.nan
    for k, _c in enumerate([cols1, cols2, cols3, cols]):
        users2[f"e_mean_{k}"] = users2[cols].mean(axis=1)  # sic: cols, not _c
        users2[f"e_std_{k}"] = users2[cols].std(axis=1)
    users2 = users2.fillna(-1)
    for c in users2.columns:
        if c not in ["user", "item"]:
            users2[c] = users2[c].astype("float32")
    for c in users2.columns[2:]:
        users[c] = users2[c].to_numpy()
    return _final_types(users)


# --- whole notebooks ---------------------------------------------------------------------

def make(model, cands, labels, events, feats, embed, giba_embed=None):
    """train_v152 (clicks), train_v172 (orders) or train_v174 (carts) before step H."""
    use = {"152": "clicks", "172": "orders", "174": "carts"}[model]
    users = step_a(cands, labels, use, keep_positive_users=model != "152")
    users = step_b(users, cands, feats, use)
    users = step_c(users, feats)
    users = step_d(users, events, embed)
    users = step_e(users, feats)
    if model != "152":
        users = step_f(users, events, embed, feats)
        users = step_g(users, events, embed if giba_embed is None else giba_embed)
    return users
