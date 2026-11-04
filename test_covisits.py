import polars as pl
from polars.testing import assert_frame_equal

from covisits import top_covisit


def test_top_covisit_keeps_top_k_per_aid():
    df = pl.DataFrame(
        {
            "aid": [1, 1, 1, 2, 2, 3],
            "aid_y": [10, 11, 12, 20, 21, 30],
            "value": [1.0, 3.0, 2.0, 5.0, 4.0, 1.0],
        }
    )
    expected = pl.DataFrame(
        {
            "aid": [1, 1, 2, 2, 3],
            "aid_y": [11, 12, 20, 21, 30],
            "value": [3.0, 2.0, 5.0, 4.0, 1.0],
        }
    )
    result = top_covisit(df, top=2)
    assert_frame_equal(
        result.sort("aid", "value", descending=[False, True]),
        expected,
    )
