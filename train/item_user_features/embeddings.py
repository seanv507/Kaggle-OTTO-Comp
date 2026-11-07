"""Item embeddings from consecutive aids: Chris Deotte's embeddings-for-train.ipynb (at
upstream commit 2bb8962) in plain PyTorch (the notebook fed the pairs with Merlin's
dataloader).

Each aid gets a `dim`-vector, trained so that an aid and the next aid in its session
score high (BCE with logits) and an aid and a shuffled next aid from the same batch score
low. Embedding rows are indexed by aid.

What the notebook really trained (`per_coordinate=True`, the default): Merlin's loader
yields each column as a (batch, 1) tensor, so in `(e1 * e2).sum(dim=1)` the sum runs
over that size-1 axis and every coordinate product e1[k] * e2[k] is its own logit,
not their dot product. With (batch, 1) inputs this port reproduces the notebook's
logged metrics (epoch 1 on the valid split: last-batch loss 0.612, in-sample accuracy
0.715 vs the notebook's 0.601, 0.717); with the dot product they are 1.21 and 0.705.
`per_coordinate=False` trains the dot-product factorisation the code reads as.

Other differences from the notebook:
- Batches are drawn from a per-epoch `torch.randperm` of the pairs, kept as int32 on the
  training device (Merlin shuffled the parquet partitions).
- The logged loss is the epoch mean (the notebook reset its meter inside the batch loop,
  so it printed the last batch's loss).
- Validation is optional and held out (`n_valid`); the notebook's valid_pairs.parquet
  was the last 10M *training* pairs, and the code writing it is commented out.
- The table has max(aid) + 1 rows (the notebook hard-coded 1855602 + 1).
"""

import time

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn


def next_pairs(*events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """`aid, aid_next` for every two consecutive events of a session, in row order.

    Each input must be sorted by session, ts; sessions must not span inputs (the
    notebook concatenates train and test)."""
    lf = pl.concat([e.lazy().select("session", "aid") for e in events])
    same = pl.col("session") == pl.col("session").shift(-1)
    return lf.select("aid", aid_next=pl.when(same).then(pl.col("aid").shift(-1))).drop_nulls()


class MatrixFactorization(nn.Module):
    """Logits for (aid1, aid2): shape (batch,) for the dot product, or (batch, n_factors)
    with `per_coordinate` (one logit per coordinate product, as the notebook trained)."""

    def __init__(self, n_aids: int, n_factors: int, per_coordinate: bool = True):
        super().__init__()
        self.aid_factors = nn.Embedding(n_aids, n_factors, sparse=True)
        self.per_coordinate = per_coordinate

    def forward(self, aid1: torch.Tensor, aid2: torch.Tensor) -> torch.Tensor:
        prod = self.aid_factors(aid1) * self.aid_factors(aid2)
        return prod if self.per_coordinate else prod.sum(dim=1)


def _batches(pairs: torch.Tensor, batch_size: int, device, shuffle: bool, generator=None):
    """Batches of `pairs`, shuffled where they live (on the GPU when training there; the
    generator must be on the same device)."""
    n = len(pairs)
    order = torch.randperm(n, generator=generator, device=pairs.device) if shuffle else torch.arange(n, device=pairs.device)
    for i in range(0, n, batch_size):
        b = pairs[order[i : i + batch_size]].to(device, non_blocking=True).long()
        yield b[:, 0], b[:, 1]


def _scores(model, aid1, aid2, generator):
    neg = aid2[torch.randperm(len(aid2), generator=generator, device=aid2.device)]
    return model(aid1, aid2), model(aid1, neg)


def _data_device(device: str, n_pairs: int, n_aids: int, dim: int) -> str:
    """`device` if the pairs fit there beside the model, else "cpu": the int32 pairs, an
    int64 permutation of them, and the embedding with SparseAdam's two moment tables, with
    a 1 GB margin."""
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    need = n_pairs * (2 * 4 + 8) + 3 * n_aids * dim * 4 + (1 << 30)
    return device if need < free else "cpu"


def train_embeddings(
    pairs: pl.DataFrame,
    n_aids: int | None = None,
    dim: int = 32,
    epochs: int = 20,
    lr: float = 0.1,
    batch_size: int = 65536,
    n_valid: int = 0,
    device: str | None = None,
    seed: int = 0,
    per_coordinate: bool = True,
) -> np.ndarray:
    """Train on `pairs` (aid, aid_next) and return the (n_aids, dim) float32 embeddings.

    `n_valid`: hold out the last n pairs and log the accuracy on them each epoch (a
    positive scores > 0.5 and a shuffled negative < 0.5, as the notebook).
    `per_coordinate`: see the module docstring."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    data = torch.from_numpy(pairs.select(pl.col("aid", "aid_next").cast(pl.Int32)).to_numpy())
    if n_aids is None:
        n_aids = int(data.max()) + 1
    train, valid = (data[:-n_valid], data[-n_valid:]) if n_valid else (data, None)
    data_device = _data_device(device, len(data), n_aids, dim)
    if data_device != "cpu":
        # the pairs (159M x 2 int32 = 1.3 GB on the valid split) live on the GPU, so batches
        # need no CPU gather or copy: on a pod with a 14-CPU quota the CPU-side batching
        # starved the GPU (391 s per epoch at 1% GPU use, 2026-11-07)
        train = train.to(data_device)
        valid = valid.to(data_device) if valid is not None else None
    elif device != "cpu":
        train = train.pin_memory()
    gen = torch.Generator(device=data_device).manual_seed(seed)  # batch order
    gen_neg = torch.Generator(device=device).manual_seed(seed + 1)  # in-batch negatives
    model = MatrixFactorization(n_aids, dim, per_coordinate).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    logger.info(f"{len(train):,} training pairs, {n_aids:,} aids x {dim}, device {device} (pairs on {data_device})")
    for epoch in range(epochs):
        t = time.perf_counter()
        model.train()
        loss_sum, n = 0.0, 0
        for aid1, aid2 in _batches(train, batch_size, device, shuffle=True, generator=gen):
            pos, neg = _scores(model, aid1, aid2, gen_neg)
            output = torch.cat([pos, neg])
            targets = torch.cat([torch.ones_like(pos), torch.zeros_like(pos)])
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(aid1)
            n += len(aid1)
        msg = f"{epoch + 1:02d}: train loss {loss_sum / n:.4f} (last batch {loss.item():.4f})"
        if valid is not None:
            msg += f", valid accuracy {accuracy(model, valid, batch_size, device, gen_neg):.4f}"
        logger.info(f"{msg} ({time.perf_counter() - t:.1f}s)")
    return model.aid_factors.weight.detach().cpu().numpy().astype("float32")


@torch.no_grad()
def accuracy(model, pairs: torch.Tensor, batch_size: int, device, generator=None) -> float:
    model.eval()
    hits, n = 0.0, 0
    for aid1, aid2 in _batches(pairs, batch_size, device, shuffle=False):
        pos, neg = _scores(model, aid1, aid2, generator)
        hits += ((pos.sigmoid() > 0.5).sum() + (neg.sigmoid() < 0.5).sum()).item()
        n += pos.numel() + neg.numel()  # per coordinate if the model is
    return hits / n
