"""Integer codes for the `type` column in data/*.parquet (stored as UInt8)."""

CLICKS = 0
CARTS = 1
ORDERS = 2

TYPE_NAMES = {CLICKS: "clicks", CARTS: "carts", ORDERS: "orders"}
