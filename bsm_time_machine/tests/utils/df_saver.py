"""
a util file for generating test dataframe pickles.

example invocation: 
for case: $ python utils/df_saver.py --dir filter_days --name 100_trading_days --is_case True
for expected: $ python utils/df_saver.py --dir filter_days --name 100_trading_days

"""
import argparse
import os

import pandas as pd
import numpy as np


def main(name: str, dir: str, is_case: bool):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(f"../data/{dir}")
    prefix = "cases/" if is_case else "expected/"
    print("prefix:", prefix, "is_case", is_case)
    df = generate_df()
    if df is None:
        print("aborted")
        return
    df.to_pickle(prefix + name + ".pkl")


def generate_df() -> pd.DataFrame():
    """modify this code to chage what the df should look like..."""
    df = pd.DataFrame(np.resize(np.arange(5), 100), columns=["day"])
    print(df)
    if input("does the below df look correct? [y/n]:") != "y":
        return
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="df_saver",
        description="",
    )
    parser.add_argument("--name", type=str)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--is_case", type=bool, default=False)
    args = parser.parse_args()
    main(args.name, args.dir, args.is_case)
