import argparse

import datetime
from pathlib import Path
import numpy as np
import polars as pl
import polars.selectors as cs
from beartype import beartype
from labels import ground_truth


@beartype
def split_events(events: pl.DataFrame)->tuple[pl.DataFrame, pl.DataFrame]:
    # accumulate backwards all the labels at each event
    events = ground_truth(events)
    events = truncate_events(events)    
    labels = events.pipe(get_last_labels)
    events = events.drop(cs.ends_with('_labels'))
    
    return events, labels

def truncate_events(events) -> pl.DataFrame: 
    rng = np.random.default_rng()
    session_lengths = (
        events.group_by('session').len()
        .with_columns(split=pl.col('len').map_batches(lambda x: rng.integers(1,x), return_dtype=pl.Int64))
    )
    events = events.join(session_lengths.select(['session','split']), on='session',maintain_order='left' )
    events = events.filter(pl.int_range(pl.len()).over('session')<pl.col('split')).drop('split')
    return events


def get_last_labels(events: pl.DataFrame):
    labels = events.select('session',cs.ends_with('_labels')).group_by('session').tail(1)
    return labels

# @beartype
# def create_kaggle_testset(events: pl.DataFrame):
    
#         session = session.to_dict()
#         splitted_events, labels = split_events(session['events'])
#         last_labels.append({'session': session['session'], 'labels': labels})
#         splitted_sessions.append({'session': session['session'], 'events': splitted_events})

#     with open(sessions_output, 'w') as f:
#         for session in splitted_sessions:
#             f.write(json.dumps(session) + '\n')

#     with open(labels_output, 'w') as f:
#         for label in last_labels:
#             f.write(json.dumps(label, cls=setEncoder) + '\n')


#@beartype
def trim_events(events:pl.DataFrame, max_ts: datetime.datetime) -> pl.DataFrame:
    events_trimmed = events.filter(pl.col('ts') < max_ts)
    return events_trimmed


@beartype
def get_max_ts(df: pl.DataFrame) -> datetime.datetime:
    max_ts = df.get_column('ts').max()
    return max_ts


@beartype
def filter_unknown_items(events: pl.DataFrame, known_items: set[int])->pl.DataFrame:
    session_length_filter = pl.col('aid').len().over('session') >= 2
    known_item_filter = pl.col('aid').is_in(known_items)
    orig_height = events.height
    events = events.filter(known_item_filter)
    print(f'dropped {orig_height -events.height} of {orig_height}')
    orig_height = events.height
    events = events.filter(session_length_filter)
    print(f'dropped {orig_height - events.height} of {orig_height}')
    return events


@beartype
def train_test_split(events: pl.DataFrame, max_ts: datetime.datetime, test_days: int):
    split_ts = max_ts - pl.duration(days=test_days)
    after_split_filter = (pl.col('ts').first().over('session') > split_ts)
    test_events, train_events = (
        events
        .with_columns(is_after=after_split_filter)
        .partition_by('is_after', include_key=False)
    )
    # dont filter test sessions of 1 event? yes, but in filter_unknown
    train_events = trim_events(train_events, split_ts)
    session_length_filter = pl.col('aid').len().over('session') >= 2
    train_events = train_events.filter(session_length_filter)
    train_items = set(train_events.get_column('aid').unique().to_list())
    # remove aids in test data sets that weren't in train
    test_events = filter_unknown_items(test_events, train_items)
    return train_events, test_events, train_items


@beartype
def main(train_set: Path, output_path: Path, days: int, seed: int):
    # random.seed(seed)
    max_ts = get_max_ts(train_set)

    events = pl.read_parquet(train_set)
    
    train_events, test_events = train_test_split(events, max_ts, days)
    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / 'train_sessions.parquet'
    test_file_full = output_path / 'test_sessions_full.parquet'
    train_events.write_parquet(train_file)
    test_events.write_parquet(test_file_full)

    test_sessions = pl.read_parquet(test_file_full)
    test_sessions_file = output_path / 'test_sessions.parquet'
    test_labels_file = output_path / 'test_labels.parquet'
    # create_kaggle_testset(test_sessions, test_sessions_file, test_labels_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--days', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.train_set, args.output_path, args.days, args.seed)
