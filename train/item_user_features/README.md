# Item and user features (Polars)

`save_features.py` computes the 14 item and user feature files of Chris Deotte's
`item-features-*` / `user-features-*` notebooks, one parquet per notebook. Run it from
the repo root:

    uv run python train/item_user_features/save_features.py \
        --train data/valid_train.parquet --test data/valid_test.parquet \
        --out data/item_user_features_valid

On the local-validation split this takes ~80 s on the local machine (streaming engine,
12 GB peak RAM). The features only read the event files, not the covisitation matrices
or the candidates.

- `features.py`: one function per notebook, collected in `FEATURES`. The module docstring
  lists the notebooks' quirks that are kept.
- `features_reference.py`: pandas transcriptions of the notebooks, which
  `test_features.py` compares against.

`embeddings-for-train.ipynb` (32-d matrix factorisation embeddings of consecutive aids,
trained with PyTorch and Merlin) is not converted yet.
