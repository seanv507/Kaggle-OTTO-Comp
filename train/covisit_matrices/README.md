
# Covisitation matrices (Polars)

The 20 train covisitation matrices are computed by `save_covisits.py` (Polars, CPU or GPU engine), built from `covisits.py`. Run from the repo root:

    uv run python train/covisit_matrices/save_covisits.py

Chris Deotte's original notebooks (`cpu-90/95/99.ipynb`, the 17 `gpu-*.ipynb`, and the driver `script.ipynb`) were removed after conversion. Read them at upstream commit `2bb8962`, e.g. `git show 2bb8962:train/covisit_matrices/cpu-90.ipynb`.

Original note on the CPU vs GPU notebooks: [discussion][1]. The GPU co-visit took 1-2 minutes whereas the CPU code took 2-3 hours.

[1]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/365369
