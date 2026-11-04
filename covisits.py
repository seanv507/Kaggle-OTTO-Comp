import datetime
from pathlib import Path
import polars as pl
from loguru import logger


aid_diff_filter = pl.col("aid")!=pl.col("aid_y")
within_14_days_filter = ((pl.col("ts")-pl.col("ts_y")).abs()) <= pl.duration(days=14)
start_train = datetime.datetime.fromisoformat('2022-07-31 22:00:00') 
end_train = datetime.datetime.fromisoformat('2022-08-28 21:59:59')
end_test = datetime.datetime.fromisoformat('2022-09-04 21:59:51') 
last_21_days_filter = (pl.col('ts')> end_test - pl.duration(days=21))
last_28_days_filter = (pl.col('ts')> end_test -pl.duration(days=28))
            # 1659304800   2022-07-31 22:00:00
            # 1661716799 "2022-08-28 21:59:59"	
            # 1662328791 '2022-09-04 21:59:51'


carts_orders_filter = pl.col("type").is_in(["carts","orders"])
within_1_day_filter = ((pl.col("ts")-pl.col("ts_y")).abs()) <= pl.duration(days=1)
hour_diff = (pl.col('ts')-pl.col('ts_y')).abs()/pl.duration(hours=1)
weight_93_ts = 1 + 3*(pl.col('ts') - start_train)/(end_test-start_train)

def session_self_join(events: pl.DataFrame):
    return events.join(events,on="session",suffix="_y")

def take_tail_30(events: pl.DataFrame):
    tail_30 = events.group_by("session").tail(30)
    logger.info(f'tail_30 {tail_30.height:,} of {events.height:,}')
    return tail_30

def filter(events: pl.DataFrame, filter):
    pre = events.height
    events = events.filter(filter)
    logger.info(f'selecting {events.height:,} from {pre:,} with {filter}')
    return events


def iterate_sessions(events: pl.DataFrame, session_chunk: int):
    max_session = events.get_column('session').max()
    n_sessions =  events.get_column('session').n_unique()
    logger.info(f'Unique sessions {n_sessions} max {max_session} chunk{session_chunk}')
    for i_gp, (gp, df) in enumerate(events.group_by(pl.col('session')//session_chunk)):
        logger.info(f'Chunk {i_gp} {gp} of {max_session//session_chunk}')
        yield i_gp, gp, df


def update_df_agg(df, df_agg):
    logger.info('joining results')
    if df_agg is None:
        df_agg = df
    else:
        df_agg = pl.concat((df_agg,df)).group_by(["aid", "aid_y"]).agg(pl.col('value').sum())
    return df_agg

def make_covisits_90(events: pl.DataFrame):
    session_chunk = 1_00_000
    logger.info(f'session_chunk {session_chunk:,}')
    events =  filter(events,carts_orders_filter)
    
    df_agg = None
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        df = (
            filter(session_self_join(df),
                   within_14_days_filter & aid_diff_filter)
            .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session
            .group_by(["aid", "aid_y"]).agg(pl.len().alias("value"))
        )
        df_agg = update_df_agg(df, df_agg)
    covisits_cpu_90 = (
        df_agg
        .sort(['aid','value'], descending=(False, True))
    )
    return covisits_cpu_90


def top_covisit(df: pl.DataFrame, top: int = 40):
    """For each aid keep the `top` aid_y with the highest `value`."""
    return df.select(
        pl.col("aid", "aid_y", "value")
        .top_k_by("value", k=top)
        .over("aid", mapping_strategy="explode")
    )

def make_covisits_95():

    session_chunk = 100_000
    logger.info(f'session_chunk {session_chunk:,}')
    events =  (
        pl.read_parquet("data/train.parquet")
        .set_sorted('session').pipe(take_tail_30)
    )
    df_agg = None
    dfs = {}
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        df = (
            filter(session_self_join(df), aid_diff_filter)
            .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
            .group_by(["aid", "aid_y"]).agg(
                ((1/2)**(hour_diff)*pl.when(pl.col('type')=='clicks').then(1).otherwise(4)).alias('value').sum()
            )
        )
        dfs[gp] = df
    return pl.concat(dfs.values())

def make_covisits_95_gpu():

    session_chunk = 100_000
    logger.info(f'session_chunk {session_chunk:,}')
    events =  (
        pl.read_parquet("data/train.parquet")
        .set_sorted('session').pipe(take_tail_30)
    )
    df_agg = None
    dfs = {}
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        df = df.lazy()
        df = (
            session_self_join(df).filter(aid_diff_filter)
            .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
            .group_by(["aid", "aid_y"]).agg(
                ((1/2)**(hour_diff)*pl.when(pl.col('type')=='clicks').then(1).otherwise(4)).alias('value').sum()
            )
        ).collect(engine='gpu')
        dfs[gp] = df
    return pl.concat(dfs.values())


def aggregate_covisits_95():
    for i in range(19):
        print(i)
        (
            pl.scan_parquet('data/out/covisits_95_unagg.parquet')
            .filter(pl.col('aid')//1_00_000 == i)
            .group_by(["aid", "aid_y"])
            .agg(pl.col('value').sum())
            .sort(['aid','value'],descending = [False, True])
            .sink_parquet((f'data/out/covisits_95/aid_{i:02d}00000.parquet'))
        )

def aggregate_covisits(ver: int):
    Path(f'data/out/covisits_{ver}/agg/').mkdir(parents=True, exist_ok=True)
    for i in range(19):
        print(i)
        (
            pl.scan_parquet(f'data/out/covisits_{ver}/unagg/aid_seg={i}/')
            .group_by(["aid", "aid_y"])
            .agg(pl.col('value').sum())
            .sort(['aid','value'],descending = [False, True])
            .sink_parquet(f'data/out/covisits_{ver}/agg/aid_seg={i:02d}.parquet')
        )


# with open(f'../../data/covisit_matrices/top_40_orders_carts_v{VER}.pkl', 'wb') as f:

# pandas CPU times: user 2h 56min 20s, sys: 42.6 s, total: 2h 57min 3s
# Wall time: 3h 27s

# using polars about 10 minutes

def make_covisits_99():
    session_chunk = 1_000_000
    events =  (
        filter(pl.read_parquet('data/train.parquet'),
            #1662328791
            last_28_days_filter & 
            carts_orders_filter)
    )
    df_agg = None
    
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        df = (
            filter(session_self_join(df), aid_diff_filter)
            .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
            .group_by(["aid", "aid_y"]).agg(
                ((1/2)**(hour_diff)).alias('value').sum()
            )
        )
        df_agg = update_df_agg(df, df_agg)
    df_agg = df_agg.sort(['aid','value'],descending=[False, True])
    return df_agg


# Chunks:   0%|          | 0/120 [00:00<?, ?it/s]

# CPU times: user 9min 25s, sys: 8.75 s, total: 9min 33s
# Wall time: 9min 29s


#with open(f'../../data/covisit_matrices/top_40_buy2buy_v{VER}.pkl', 'wb') as f:


def make_covisits_93():
    events =  pl.read_parquet('data/train.parquet').group_by("session").tail(30)
    dfs = {}
    session_chunk = 500_000
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(aid_diff_filter & within_1_day_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            (weight_93_ts*(1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
# takes around 5 minutes on cpu (including save and regroup)
#df_agg_p = df_agg.partition_by('aid_seg',maintain_order=False, include_key=True, as_dict=True)
# memory error

    # df_agg.write_parquet(pl.PartitionBy('data/out/covisits_93/')
    
    #     .group_by(["aid", "aid_y"])
    #     .agg(pl.col('value').sum())
    #     .sort(['aid','value'], descending=[False, True])
    # )
    # return df_agg
# run on gpu
# CPU times: user 55.5 s, sys: 41.1 s, total: 1min 36s
# Wall time: 1min 40s
# output top 80


            # df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
            # df['wgt'] = df['wgt'] * ((1/2)**( (df.ts_x - df.ts_y).abs() /60/60))
            # df = df[['aid_x','aid_y','wgt']]


def make_covisits_115():
    events =  pl.read_parquet('data/train.parquet')
    events = filter(events, last_21_days_filter)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(aid_diff_filter & within_1_day_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
# CPU times: user 2min 31s, sys: 1min 29s, total: 4min 1s
# Wall time: 4min 11s

# with open(f'../../data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl'
y_after_x_filter = (pl.col('ts_y') > pl.col('ts'))

def make_covisits_116(events: pl.DataFrame):
    events =  pl.read_parquet('data/train.parquet')
    events = filter(events, last_28_days_filter & carts_orders_filter)
    # type weight 0 for clicks, otherwise 1
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(y_after_x_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
#with open(f'../../data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl', 'wb') as f:
       # pickle.dump(df.to_dict(), f)
    
def make_covisits_155(test_events: pl.DataFrame):
    events = filter(test_events, carts_orders_filter)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(y_after_x_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff/6)).alias('value').sum()
        )
    )
    
def make_covisits_157(test_events: pl.DataFrame):
    events=test_events
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff/6)).alias('value').sum()
        )
    )
        # top 80

session_length = pl.col('aid').count().over('session').alias('x')
def make_covisits_165(train_events:pl.DataFrame, test_events: pl.DataFrame):
    train_events = train_events.filter(session_length <=6)
    test_events = test_events.filter(session_length <=3)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(filter(df, carts_orders_filter),on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
# top 40

def make_covisits_166(train_events:pl.DataFrame, test_events: pl.DataFrame):
    train_events = train_events.filter(session_length > 6)
    test_events = test_events.filter(session_length > 3)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(filter(df, carts_orders_filter),on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
# top 40


#     "cell_type": "code",
# -   "execution_count": 3,
# +   "execution_count": null,
#     "id": "d487dbd4",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "aid_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_95_tail_30 =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .group_by(\"session\").tail(30)\n",
# -    "    \n",
# -    ")\n",
# -    "dfs = {}\n",
# -    "for gp, df in cpu_95_tail_30.group_by(pl.col('session')//10_000):\n",
# -    "    dfs[gp] = (\n",
# -    "#covisits_95_lzy = (\n",
# -    "    df\n",
# -    "    .join(df,on=\"session\",suffix=\"_y\")\n",
# -    "    .filter(aid_diff_filter)\n",
# -    "    .unique(subset = [\"session\",\"aid\", \"aid_y\"]) # 1 pair per session (?random time then if duplicate)\n",
# -    "    .group_by([\"aid\", \"aid_y\"]).agg(\n",
# -    "        ((1/2)**((pl.col('ts')-pl.col('ts_y'))/pl.duration(hours=1))*pl.when(pl.col('type')=='clicks').then(1).otherwise(4)).alias('weight').sum()\n",
# -    "    )\n",
# -    ")"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -1998,13 +2775,11 @@
#    },
#    {
#     "cell_type": "code",
# -   "execution_count": 4,
# +   "execution_count": null,
#     "id": "267c55ad",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "covisits_95 = pl.concat(dfs.values())"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -2259,30 +3034,7 @@
#     "id": "b8ecb411",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "id_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_99_28_days =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .filter(\n",
# -    "        #1662328791\n",
# -    "        (pl.col('ts')> datetime.datetime.fromisoformat('2022-09-04 21:59:51') -pl.duration(days=28)) & \n",
# -    "        pl.col('type').is_in(['carts','orders'])\n",
# :
#     "outputs": [],
# -   "source": [
# -    "covisits_95 = pl.concat(dfs.values())"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -2259,30 +3034,7 @@
#     "id": "b8ecb411",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "id_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_99_28_days =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .filter(\n",
# -    "        #1662328791\n",
# -    "        (pl.col('ts')> datetime.datetime.fromisoformat('2022-09-04 21:59:51') -pl.duration(days=28)) & \n",
# -    "   
