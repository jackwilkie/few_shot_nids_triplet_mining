"""
Functions for appending to csv file in python

Created on: 25/11/2023
"""

import pandas as pd
import os
from typing import Union, List
import csv
import json


def dir_exists(path):
    dir_path = os.path.dirname(path)

    if dir_path:
        if os.path.exists(dir_path):
            return True
        else:
            return False
    else:
        return True


def append_csv(
    data: Union[List[dict], dict], path: str, quick_add: bool = False
) -> None:
    # ensure list of dicts
    data = [data] if not isinstance(data, list) else data

    # --- 🔧 serialize nested objects safely ---
    def serialize(v):
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v)  # safe, consistent
        return v

    data = [{k: serialize(v) for k, v in row.items()} for row in data]

    df = pd.DataFrame(data)

    # ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if quick_add:
        df.to_csv(
            path,
            mode="a",
            index=False,
            header=not os.path.exists(path),
            quoting=csv.QUOTE_ALL,  # 🔥 critical fix
            escapechar="\\",
        )
    else:
        if os.path.exists(path):
            df_existing = pd.read_csv(path, engine="python")
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_ALL,  # 🔥 critical fix
            escapechar="\\",
        )


def read_csv(path):
    return pd.read_csv(os.path.expanduser(path))


class NoContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
