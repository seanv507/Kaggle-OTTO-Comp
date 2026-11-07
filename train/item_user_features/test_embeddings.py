import datetime

import numpy as np
import polars as pl
import pytest
import torch

import embeddings
from event_types import CLICKS
from test_features import ENGINES, EVENT_SCHEMA, _engine

T0 = datetime.datetime(2022, 8, 1)
S = datetime.timedelta(seconds=1)


def _events(rows):
    return pl.DataFrame([(s, a, T0 + i * S, CLICKS) for i, (s, a) in enumerate(rows)],
                        schema=EVENT_SCHEMA, orient="row")


@pytest.mark.parametrize("engine", ENGINES)
def test_next_pairs(engine):
    train = _events([(0, 5), (0, 6), (0, 5), (1, 9)])  # session 1: one event, no pair
    test = _events([(7, 1), (7, 2)])
    got = embeddings.next_pairs(train, test).collect(engine=_engine(engine))
    assert got.rows() == [(5, 6), (6, 5), (1, 2)]


def _cluster_pairs(n_pairs, seed):
    """aids 0-49 and 50-99 form two clusters; consecutive aids stay in their cluster."""
    rng = np.random.default_rng(seed)
    aid = rng.integers(0, 100, n_pairs)
    nxt = (aid // 50) * 50 + rng.integers(0, 50, n_pairs)
    return pl.DataFrame({"aid": aid, "aid_next": nxt}, schema={"aid": pl.Int32, "aid_next": pl.Int32})


@pytest.mark.parametrize("per_coordinate", [True, False])
def test_learns_clusters(per_coordinate):
    pairs = _cluster_pairs(40_000, seed=0)
    emb = embeddings.train_embeddings(pairs, n_aids=101, dim=8, epochs=5, lr=0.05, batch_size=1024,
                                      n_valid=4000, device="cpu", per_coordinate=per_coordinate)
    assert emb.shape == (101, 8) and emb.dtype == np.float32
    # same-cluster dot products beat cross-cluster ones
    dots = emb[:100] @ emb[:100].T
    same = np.add.outer(np.arange(100) // 50, np.arange(100) // 50) % 2 == 0
    assert dots[same].mean() > dots[~same].mean() + 1

    model = embeddings.MatrixFactorization(101, 8, per_coordinate)
    model.aid_factors.weight.data = torch.from_numpy(emb)
    held_out = torch.from_numpy(_cluster_pairs(4000, seed=1).to_numpy())
    # a shuffled negative is cross-cluster half the time, so ~0.75 is the ceiling
    assert embeddings.accuracy(model, held_out, 1024, "cpu", torch.Generator().manual_seed(0)) > 0.65


def test_per_coordinate_logits():
    """One logit per coordinate product, as the notebook's (batch, 1) Merlin columns gave."""
    model = embeddings.MatrixFactorization(3, 2, per_coordinate=True)
    model.aid_factors.weight.data = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    aid1, aid2 = torch.tensor([0, 1]), torch.tensor([1, 2])
    assert model(aid1, aid2).tolist() == [[3.0, -2.0], [1.5, -0.5]]
    # the notebook's code on (batch, 1) inputs gives the same logits
    reference = (model.aid_factors(aid1[:, None]) * model.aid_factors(aid2[:, None])).sum(dim=1)
    assert torch.equal(model(aid1, aid2), reference)
    model.per_coordinate = False
    assert model(aid1, aid2).tolist() == [1.0, 1.0]


def test_seeded_runs_repeat():
    pairs = _cluster_pairs(5000, seed=0)
    kw = dict(n_aids=100, dim=4, epochs=2, batch_size=512, device="cpu", seed=3)
    np.testing.assert_array_equal(embeddings.train_embeddings(pairs, **kw), embeddings.train_embeddings(pairs, **kw))
