import pickle
import pandas as pd

# Setting up pagerank features!
df_train = pd.read_csv('./train.csv').fillna("")
df_test = pd.read_csv('./df_test_with_qid.csv').fillna("")
print 'dataframes loaded'

with open('pagerank.pickle', 'rb') as handle:
    pr = pickle.load(handle)
print 'pagerank loaded'

with open('qid_graph.pickle', 'rb') as handle:
    graph = pickle.load(handle)
print 'graph loaded'


def get_pagerank_value(row):
    return pd.Series({
        "q1_pr": pr[row["qid1"]],
        "q2_pr": pr[row["qid2"]]
    })


# pagerank_feats = df_test.progress_apply(get_pagerank_value, axis=1)
# pagerank_feats_1 = df_test[:390000].apply(get_pagerank_value, axis=1)


def pagerank_neighbor_features(row):
    q1_neighbors = graph[row["question1"]]
    q2_neighbors = graph[row["question2"]]
    common_neighbors = set(q1_neighbors).intersection(q2_neighbors)
    common_neighbors_pr = sum([pr[j] for j in common_neighbors])
    q1_neighbors_pr = sum([pr[j] for j in q1_neighbors])
    q2_neighbors_pr = sum([pr[j] for j in q2_neighbors])
    if q1_neighbors_pr + q2_neighbors_pr - common_neighbors_pr == 0:
        return 0
    else:
        return common_neighbors_pr / (q1_neighbors_pr + q2_neighbors_pr - common_neighbors_pr)


def get_pr_ratio(row):
    if row["q1_pr"] != 0 and row["q2_pr"] != 0:
        return max(row["q1_pr"] / row["q2_pr"], row["q2_pr"] / row["q1_pr"])
    else:
        return 0


def real_testing(gen_filename, df_with_qs=None, res_file=None):
    # Required for initial setup!
    old_filename = './old/' + gen_filename
    df_with_qs.to_csv(old_filename, index=False)
    print 'old file dumped!'

    dataframe_modified = df_with_qs
    dataframe_modified["pr_neighbor_ratio"] = df_with_qs.apply(pagerank_neighbor_features, axis=1)
    dataframe_modified["pr_diff"] = abs(dataframe_modified["q1_pr"] - dataframe_modified["q2_pr"])
    dataframe_modified["pr_avg"] = (dataframe_modified["q1_pr"] + dataframe_modified["q2_pr"]) / 2
    dataframe_modified["pr_ratio"] = dataframe_modified.apply(get_pr_ratio, axis=1)
    if res_file is None:
        new_filename = "./new/" + gen_filename
    else:
        new_filename = "./new/" + res_file
    dataframe_modified.to_csv(new_filename, index=False)
    print 'new file dumped'


# %reset_selective dataframe_modified

real_testing('test_1.csv', df_test[:390000])