import pandas as pd
import numpy as np


def make_baseline_submission(*args):
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    df_train[[f"ft_{i + 1}" for i in range(5)]] = np.random.normal(
        size=(df_train.shape[0], 5)
    )
    df_test[[f"ft_{i + 1}" for i in range(5)]] = np.random.normal(
        size=(df_test.shape[0], 5)
    )

    df_train.to_csv("output/train.csv", index=False)
    df_test.to_csv("output/test.csv", index=False)
