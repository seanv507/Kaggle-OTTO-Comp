import datetime

import polars as pl
from polars.testing import assert_frame_equal

import lb_data

SCHEMA = {"session": pl.Int32, "aid": pl.Int32, "ts": pl.Datetime("ms"), "type": pl.UInt8}


def _events(rows):
    return pl.DataFrame([(s, a, datetime.datetime.fromisoformat(t), y) for s, a, t, y in rows], schema=SCHEMA, orient="row")


def test_lb_events_drop_week_1_and_shift_back_7_days():
    train = _events([
        (1, 10, "2022-07-31 22:00:00", 0),
        (1, 11, "2022-08-07 21:59:59.999", 1),  # last ms of week 1: dropped
        (1, 12, "2022-08-07 22:00:00", 0),  # first of week 2: kept
        (2, 13, "2022-08-28 21:59:59", 2),
    ])
    test = _events([(3, 14, "2022-08-28 22:00:00", 0), (3, 15, "2022-09-04 21:59:51", 1)])
    lb_train, lb_test = (f.collect() for f in lb_data.lb_events(train, test))
    assert_frame_equal(lb_train, _events([(1, 12, "2022-07-31 22:00:00", 0), (2, 13, "2022-08-21 21:59:59", 2)]))
    assert_frame_equal(lb_test, _events([(3, 14, "2022-08-21 22:00:00", 0), (3, 15, "2022-08-28 21:59:51", 1)]))


def test_shift_keeps_weekday_and_hour():
    assert lb_data.SHIFT.days % 7 == 0 and lb_data.SHIFT.seconds == 0
