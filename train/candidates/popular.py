"""Popular aids used to pad candidate lists to ITEM_CT (score -2 in the -wgt files).

The candidate notebooks hard-code `top_clicks` / `top_orders` ("COMPUTED FROM TEST DATA"):
the 100 most frequent aids per type in the local-validation test week (valid_test).
Chris's lists match `value_counts()` of valid_test exactly, except for the order of aids
with equal counts, which pandas leaves arbitrary. `top_carts` is defined but never used
(the carts candidates are the orders list).

- CHRIS_TOP_CLICKS / CHRIS_TOP_ORDERS: the literals, for exact parity on Chris's data.
- `popular_from_events`: the same rule on any events (ties: lowest aid first), for
  other data such as the leaderboard test week.
"""

import polars as pl

from event_types import CLICKS, ORDERS

CHRIS_TOP_CLICKS = [
    485256, 1460571, 108125, 1551213, 33343, 613493, 876493, 152547, 184976, 1236775,
    1406660, 836852, 331708, 1531805, 29735, 554660, 634452, 986164, 959208, 832192,
    332654, 620545, 1645990, 166037, 102345, 1116095, 811371, 322370, 1733943, 1043508,
    1019736, 756588, 801774, 231487, 861401, 1502122, 1030009, 329725, 1624436, 1022566,
    1006198, 496180, 1685214, 1853288, 530377, 659399, 1604220, 1102089, 1586171,
    1497089, 754412, 1765072, 1109824, 1264313, 881286, 1294924, 199409, 1581568,
    385065, 195381, 558573, 1196256, 1782099, 612920, 435253, 923948, 508883, 1126038,
    1647563, 1508062, 1182614, 544144, 247240, 982423, 57315, 1628069, 674590, 1798916,
    892871, 909449, 337471, 721034, 1734305, 95488, 1722991, 1754984, 819288, 1462420,
    836707, 1796103, 42628, 642804, 122983, 321547, 1278671, 714524, 1603001, 409620,
    636101, 1702657,
]

CHRIS_TOP_ORDERS = [
    876493, 1406660, 1236775, 166037, 1460571, 1531805, 836852, 634452, 923948, 1043508,
    832192, 801774, 258353, 332654, 1596897, 29735, 544144, 1581568, 1006198, 1462420,
    331708, 231487, 642804, 986164, 1116095, 329725, 1022566, 1604220, 1733943, 289406,
    631899, 162064, 122983, 247240, 1436280, 954951, 1445562, 373490, 1568011, 172856,
    756588, 184976, 1534690, 508883, 1310373, 530377, 1457846, 1196256, 321547, 982423,
    892871, 1441266, 1294924, 1685214, 1609228, 576949, 714524, 56279, 1102089, 314210,
    1125638, 1768724, 1734305, 898836, 1188425, 145332, 688602, 527209, 1647157, 42628,
    1628069, 1722991, 409620, 102416, 1476166, 1734475, 1182614, 926412, 988767, 137514,
    500609, 479970, 165160, 443425, 302096, 1257293, 823143, 856779, 1030009, 304807,
    1164832, 78262, 1025795, 1133209, 881286, 1620020, 742601, 108125, 558573, 1107897,
]

CHRIS = {"clicks": CHRIS_TOP_CLICKS, "orders": CHRIS_TOP_ORDERS}
TARGET_TYPE = {"clicks": CLICKS, "orders": ORDERS}


def popular_from_events(events: pl.LazyFrame | pl.DataFrame, target: str, n: int = 100) -> list[int]:
    """The `n` most frequent aids among `target` events (count descending, then aid)."""
    return (
        events.lazy()
        .filter(pl.col("type") == TARGET_TYPE[target])
        .group_by("aid")
        .len()
        .sort(["len", "aid"], descending=[True, False])
        .head(n)
        .collect()["aid"]
        .to_list()
    )
