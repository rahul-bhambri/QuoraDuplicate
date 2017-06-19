import pandas as pd


if __name__ == "__main__":
    df_train = pd.read_csv('train.csv').fillna("")
    df_test = pd.read_csv('./df_test_with_qid.csv').fillna("")
