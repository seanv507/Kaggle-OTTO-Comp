
from beartype import beartype
import polars as pl


@beartype
def ground_truth(events: pl.DataFrame):
    # clicks using as of join
    # click label is aid of first click after
    events = (
        events.with_columns(
            click_labels = (
                pl.when(pl.col('type')=='clicks').then(pl.col('aid')).otherwise(None).shift(-1)
                .reverse().cumulative_eval(pl.element().implode().list.drop_nulls().list.first()).reverse().over('session',)
            ),
            cart_labels = (
                pl.when(pl.col('type')=='carts').then(pl.col('aid')).otherwise(None).shift(-1)
                .reverse().cumulative_eval(pl.element().implode().list.drop_nulls()).reverse().over('session',)
            ),
            order_labels = (
                pl.when(pl.col('type')=='orders').then(pl.col('aid')).otherwise(None).shift(-1)
                .reverse().cumulative_eval(pl.element().implode().list.drop_nulls()).reverse().over('session',)
            )
        # .with_columns(
        #     cart_labels = (
        #         pl.col('cart_label1').reverse().cumulative_eval(pl.element().implode().list.drop_nulls()).reverse()
        #         )
        # .over('session',mapping_strategy='join')
        )
    )
    events = drop_last_event_in_session(events)
    return events


def drop_last_event_in_session(events: pl.DataFrame):
    events = events.filter(pl.int_range(pl.len()).over("session")  < (pl.len().over("session") - 1))
    return events


def test_gt():
    events = pl.DataFrame(
        {
            'session': [0,0,0, 1,1,1],
            'type': ['clicks', 'carts', 'carts', 'orders','clicks','orders'],
            'aid': [1,2,3, 4,5,6],}
    )

    #expected = events.with_columns(cart_labels=pl.Series([[2],[2],[]]))
    actual = ground_truth_pl(events)
    return  actual
                                   