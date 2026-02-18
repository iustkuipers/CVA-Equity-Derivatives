import os
import json
import pandas as pd
from datetime import datetime


class OutputWriter:

    @staticmethod
    def ensure_directory(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data: dict, filepath: str):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
