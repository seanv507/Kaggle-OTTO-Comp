# Rankers (XGBoost)

`train_rankers.py` trains the three XGBoost rankers of Chris Deotte's XGB-186-CLICKS,
XGB-406-CARTS and XGB-412-ORDERS notebooks on `save_merged.py`'s rows (5 folds by
session each) and prints OTTO recall@20 of the out-of-fold predictions. Run it from the
repo root:

    uv run python train/ranker_models/train_rankers.py \
        --rows data/train_with_features_valid --labels data/valid_test_labels.parquet \
        --out data/models_valid

Clicks runs first: its fold models add the `click_oof` feature to the carts and orders
rows (step H of Make-172/174, formerly `train/make_parquets/Make-17x-H.ipynb`).

- `rankers.py`: parameters, negative sampling, folds, training, `click_oof` and the
  recall metric. Its docstring lists the differences from the notebooks.
- The clicks recall is printed twice: on all rows, which reproduces the notebook's figure
  (full valid run: 0.55845 against Chris's 0.55835), and on the sampled rows only (all
  positives, 40% of the negatives), which flatters it (0.636 on the same run).

The CatBoost notebooks (`CAT-200-orders.ipynb`, `CAT-203-carts.ipynb`) are not ported:
Chris's LB inference uses the XGBoost models only.

The XGB notebooks (and Make-172/174-H) were removed after conversion. Read them at
upstream commit `2bb8962`, e.g. `git show 2bb8962:train/ranker_models/XGB-186-CLICKS.ipynb`.
