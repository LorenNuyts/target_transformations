from typing import List

import pandas as pd


def skewed_columns(df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
    skew_per_column = df.skew()
    skewed_cols = skew_per_column[abs(skew_per_column) > threshold].index
    return skewed_cols
