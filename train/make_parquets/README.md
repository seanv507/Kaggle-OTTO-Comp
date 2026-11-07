# Merged training rows (Polars)

`save_merged.py` builds the rankers' training rows of Chris Deotte's Make-152 (clicks),
Make-172 (orders) and Make-174 (carts) notebooks, steps A-G: model 709's 100 candidates
per session with the target, the scores of the other candidate runs, the item/user
features and embedding similarities. Run it from the repo root:

    uv run python train/make_parquets/save_merged.py \
        --events data/valid_test.parquet --labels data/valid_test_labels.parquet \
        --candidates data/candidate_scores_valid --features data/item_user_features_valid \
        --out data/train_with_features_valid

Each model is written as `<out>/train_v<model>/part-NNN.parquet`, one file per session
chunk (`--chunk`); read the directory with `pl.scan_parquet`. The column order matches the
notebooks' (the rankers take `columns[2:]` minus the target as features): 155 columns for
152, 225 for 172 / 174 (226 with step H's `click_oof`).

- `merge.py`: the steps as lazy joins, with the embedding similarities in numpy.
- `merge_reference.py`: pandas transcriptions of the notebooks, which `test_merge.py`
  compares against. Its docstring lists the notebook quirks that are kept.

Not here:
- Step H (`Make-172-H`, `Make-174-H`): the clicks ranker's out-of-fold predictions as the
  feature `click_oof`. It needs the trained XGB-186 models, so it belongs with
  train/ranker_models.
- Giba's embeddings. Step G uses `embbedings_fm10.npy` from Giba's code ([here][1]) if
  given with `--giba-embed`; without it, like the notebooks, it falls back to
  `item_embed_32.npy` (Chris notes this scores slightly lower CV and LB).

[1]: https://github.com/titericz/otto2023_giba
