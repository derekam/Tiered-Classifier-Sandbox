import pandas as pd
import csv
import numpy as np
from pandas import DataFrame


# Read the csv file into a pandas Dataframe
def read_csv_full(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes)
    return df.set_index("id")


def read_csv_predictions(file_path) -> DataFrame:
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        dtypes = {x: np.float16 for x in column_names if
                  x.startswith(('prediction', 'probability'))}
    return pd.read_csv(file_path, dtype=dtypes).set_index("id")
