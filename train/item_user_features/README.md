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

`save_embeddings.py` trains the 32-d item embeddings of `embeddings-for-train.ipynb` on
consecutive aids of the train and test sessions (plain PyTorch instead of Merlin's loader)
and writes `item_embed_32.npy`, indexed by aid:

    uv run python train/item_user_features/save_embeddings.py --out data/item_user_features_valid

An epoch takes ~36 s on the local RTX 2050 (149M pairs). `embeddings.py` explains a
quirk it reproduces: the notebook trained one logit per coordinate product, not the dot
product (`--dot` trains the dot product instead).

Chris's original notebooks (the `item-features-*` / `user-features-*` notebooks,
`embeddings-for-train.ipynb` and the driver `script.ipynb`) were removed after
conversion. Read them at upstream commit `2bb8962`, e.g.
`git show 2bb8962:train/item_user_features/item-features-4.ipynb`.
