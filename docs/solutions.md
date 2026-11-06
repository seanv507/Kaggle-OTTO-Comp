# Reference solutions: OTTO 3rd place

The 3rd-place team (Benny, Chris, Giba and Theo) built individual models and ensembled them by adding the ranks of each model's predictions for each session and target. This repo follows Chris's and Benny's parts. The summaries below come from their write-ups and repo READMEs.

## Chris Deotte: candidate rerank, LB 0.601

- Code: [cdeotte/Kaggle-OTTO-Comp](https://github.com/cdeotte/Kaggle-OTTO-Comp), local copy at `../Kaggle-OTTO-Comp`.
- [Write-up: "3rd Place - Using Only Rules Achieves LB 0.590!"](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)
- Base method: [candidate-rerank-model-lb-0-575](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575)

### Overview (from the repo README)

- CV 0.601, LB 0.601 (private 0.6012).
- For each target (click, cart, order) and each session, generate about 100 candidate aids from 20 covisitation matrices plus handcrafted rules (43 candidate scripts).
- Merge item, user and user-item features onto each (session, aid) pair.
- One XGBoost ranker per target (`XGB-186-CLICKS`, `XGB-406-CARTS`, `XGB-412-ORDERS`) picks the final 20. CatBoost rankers for carts and orders give LB 0.6016.
- Trains on the first 3.5 weeks and infers on all 4.5.
- Part of the make-parquets step needs Giba's `embbedings_fm10.npy` ([titericz/otto2023_giba](https://github.com/titericz/otto2023_giba)).

### The three steps beyond the public notebook

1. **Candidates:** take the top 50 to 100 from the rule-based `suggest_clicks`/`suggest_buys`, which sum covisitation neighbours over the session's aids, instead of the top 20.
2. **Interaction features:** for each session × candidate pair, the rule's covisitation count, extracted *per matrix* to give one feature each. Add simple item and user counts of clicks, carts and orders.
3. **20 covisitation matrices:** in place of the public notebook's 3. Rules alone with 20 matrices score LB 0.590, and the reranker adds about +0.011, giving 0.601.

### The 20 covisitation matrices

These use the write-up's names. He picked them from hundreds by local CV. Mapping these names to the notebook numbers in `train/covisit_matrices/` (and the `make_covisits_*` functions in `covisits.py`) is still to do.

`k` is the event's position in the session, and `ts_x`/`ts_y` are the two events' times. "Time decay" is weight `(1/2)**(|ts_x - ts_y| / 1 hour)`. Matrices with no type restriction pair every event type.

| Name | Pairs counted |
|---|---|
| `top_20` | the original "clicks" matrix |
| `top_20b` | consecutive events only, `\|k_y - k_x\| == 1` |
| `top_20c` | `\|k_y - k_x\| <= 2` |
| `top_20d` | carts/orders only, forward 1 to 3 positions |
| `top_20e` | `\|k_y - k_x\| <= 3`, time decay |
| `top_20f` | `\|k_y - k_x\| <= 6`, time decay |
| `top_20_orders` | original carts/orders matrix |
| `top_20_buy2buy` | original buy-to-buy matrix |
| `top_20_buy2buy2` | last 3 weeks, carts/orders only, time decay |
| `top_20_test` | last 3 weeks, forward in time, any type → carts/orders, time decay |
| `top_20_test2` | last 2 weeks, time decay |
| `top_20_buy` | forward up to 2 hours, any type → carts/orders, time decay |
| `top_20_new` | cold-start train sessions (first event more than 2 weeks after train start); first event → carts/orders |
| `top_20_new2` | as `top_20_new` but → any type, time decay |
| `top_40_day` | last week, forward in time, any type → carts/orders, time decay |
| `top_40_day2` | last week, time decay |
| `top_40_less` | short sessions: train < 6 events, test < 3 |
| `top_40_more` | long sessions: train > 6 events, test > 3 |
| `top_40_less2` | first event before 2pm, any type → carts/orders, time decay |
| `top_40_more2` | first event after 2pm, any type → carts/orders, time decay |

Each matrix took under a minute on GPU with cuDF (4 × V100 32 GB), which made searching hundreds of variants feasible.

## Benny Schifferer: XGBoost reranker with Transformer and GRU inputs, LB 0.601

- Code: local copy at `../Kaggle-Otto-Comp-Benedikt`, the user's fork `seanv507/Kaggle-Otto-Comp-Benedikt`.
- [Write-up](https://www.kaggle.com/competitions/otto-recommender-system/discussion/386497)
- Folders are numbered by step: `00_Preprocess`; feature engineering in `01a_FE_Word2Vec`, `01b_FE_CoVisit_1`, `01c_FE_CoVisit_2_Chris`, `01d_FE_GRU`, `01e_FE_Transformer` and `01f_FE_other`; then one model per target in `02_Clicks`, `03_Carts` and `04_Orders`.
- Uses cuDF, plus TensorFlow/Transformers4Rec and gensim for the sequence and embedding features.

### Features (~160)

- 7 of his own covisitation matrices, plus 10 of Chris's.
- Next-item scores from a Transformers4Rec model and a GRU. Both were trained with negative sampling, and the GRU was initialised from word2vec.
- Word2Vec embeddings, e.g. cosine similarity between the candidate and each viewed item, aggregated over the session.
- Session features (length, day, time).
- Candidate activity counts (clicks, carts and orders on the same day, previous 7 days, previous 13 days).
- Session × candidate scores aggregated several ways: max, mean, rank- or linearly weighted mean, sum, count.

### Candidates (about 100 to 150 per session)

- Clicks: the union of the top 80 from about 6 covisitation matrices, plus session history.
- Carts and orders: the union of the top 80 from all matrices plus history, keeping only candidates found by at least 2 sources.
- More candidates helped; adding more for carts and orders took LB from 0.599 to 0.601.

### Rankers

XGBoost, with hierarchical inputs. A click model trained on sessions with only click targets feeds its probability into the order model, and so does the cart model; only the order model used them. A neural-net reranker didn't beat XGBoost, and removing features always hurt CV.

### Data split (his biggest gain)

Weeks 1 to 3 are kept whole, and week 4 is split into 5 folds by session. For each fold, that fold's sessions are truncated and used to train the ranker. The other 80% of week 4 stays whole, and together with weeks 1 to 3 and the test sessions it is used for feature engineering. Running all 5 folds means 100% of week 4 is available for training, at the cost of slow, complicated pipeline runs. Generating training and submission features from the same data avoids distribution shift.

### Ablation

| Model | CV | LB |
|---|---|---|
| Chris's baseline | 0.567 | 0.575 |
| XGBoost trained on truncated week 4 | 0.576 | 0.583 |
| trained on 20% truncated week 4 | 0.585 | 0.591 |
| + activity history | 0.587 | 0.594 |
| trained on 100% (all folds) | 0.592 | 0.598 |
| + Word2Vec | 0.593 | 0.599 |
| + more cart/order candidates | | 0.601 |
