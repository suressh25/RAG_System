import sys
import importlib

try:
    import pysqlite3

    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
    sys.modules["sqlite3.dbapi2"] = sys.modules["pysqlite3.dbapi2"]
except ImportError:
    raise ImportError(
        "pysqlite3 is required but not installed. Run `pip install pysqlite3-binary`"
    )
