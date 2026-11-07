# Candidates and scores (Polars)

`save_candidates.py` generates the candidate lists and their scores for every session,
one parquet file per run of Chris Deotte's original notebooks. Run it from the repo root:

    uv run python train/candidates/save_candidates.py \
        --events data/valid_test.parquet --covisits data/covisit_matrices_valid \
        --out data/candidate_scores_valid --labels data/valid_test_labels.parquet --chunk 250000

`--labels` prints the recall of each run. `--chunk` bounds memory: unchunked, the largest
runs (564_F_orders, 612_orders) need ~85 KB per session. `run_a100.sh` runs the matrices and
the candidates on the local-validation split on a cloud GPU pod.

- `candidates.py`: the rules of the five notebook families (564, 612/614, 620, 709) as
  Polars joins, and `RUNS`, one entry per notebook. Scores are summed in the notebooks'
  order, so candidates and scores match exactly; that needs a per-group cumulative sum
  the GPU engine lacks, so this runs on the CPU engines.
- `candidates_reference.py`: pandas ports of the notebooks, which `test_candidates.py`
  compares against.
- `popular.py`: the popular-item lists that pad short candidate lists.

Chris's original notebooks (the `model-564-*`, `model-612/614/620/709` notebooks and
their `-wgt` twins, `make-valid-with-d.ipynb`, `make-valid-lists.ipynb` and the driver
`script.ipynb`) were removed after conversion. Read them at upstream commit `2bb8962`, e.g.
`git show 2bb8962:train/candidates/model-564-1.ipynb`.
