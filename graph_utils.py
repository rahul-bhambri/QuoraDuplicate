import pandas as pd
import networkx as nx
import numpy as np


def neighbor_intersection(row, graph):
    q1_neighbors = graph[row["question1"]]
    q2_neighbors = graph[row["question2"]]

    common_neighbors = set(q1_neighbors).intersection(q2_neighbors)

    return len(common_neighbors)


def neighbor_features(row, graph):
    q1_neighbors = graph[row["question1"]]
    q2_neighbors = graph[row["question2"]]

    common_neighbors = set(q1_neighbors).intersection(q2_neighbors)

    neighbor_ratio = len(common_neighbors) / (len(q1_neighbors) + len(q2_neighbors) - len(common_neighbors))
    neighbor_dissimilar = len(q1_neighbors) + len(q2_neighbors) - len(common_neighbors)

    return pd.Series({
        "neighbor_ratio": neighbor_ratio,
        "neighbor_dissimilar": neighbor_dissimilar
    })


def get_q1_second_degree_freq(row, graph):
    q1_neighbors = graph[row["question1"]]

    q1_second_degree_neighbors = []
    for i in q1_neighbors:
        q1_second_degree_neighbors += graph[i]

    return len(set(q1_second_degree_neighbors))


def get_q2_second_degree_freq(row, graph):
    q2_neighbors = graph[row["question2"]]

    q2_second_degree_neighbors = []
    for i in q2_neighbors:
        q2_second_degree_neighbors += graph[i]

    return len(set(q2_second_degree_neighbors))


def second_degree_intersection(row, graph):
    q1_neighbors = graph[row["question1"]]
    q2_neighbors = graph[row["question2"]]

    q1_second_degree_neighbors = []
    for i in q1_neighbors:
        q1_second_degree_neighbors += graph[i]

    q2_second_degree_neighbors = []
    for i in q2_neighbors:
        q2_second_degree_neighbors += graph[i]

    common_second_degree_neighbors = set(q1_second_degree_neighbors).intersection(set(q2_second_degree_neighbors))

    return len(common_second_degree_neighbors)


def generate_positive_graph(row, pos_graph):
    hash_key1 = row["question1"]
    hash_key2 = row["question2"]

    if row["is_duplicate"] == 1:
        if hash_key1 not in pos_graph:
            pos_graph[hash_key1] = [hash_key2]
        elif hash_key1 in pos_graph:
            pos_graph[hash_key1].append(hash_key2)

        if hash_key2 not in pos_graph:
            pos_graph[hash_key2] = [hash_key1]
        elif hash_key2 in pos_graph:
            pos_graph[hash_key2].append(hash_key1)


def generate_graph_table(row, graph):
    hash_key1 = row["question1"]
    hash_key2 = row["question2"]

    if hash_key1 not in graph:
        graph[hash_key1] = [hash_key2]
    elif hash_key1 in graph:
        graph[hash_key1].append(hash_key2)

    if hash_key2 not in graph:
        graph[hash_key2] = [hash_key1]
    elif hash_key2 in graph:
        graph[hash_key2].append(hash_key1)


def generate_qid_graph_table(row, qid_graph):
    hash_key1 = row["qid1"]
    hash_key2 = row["qid2"]

    if hash_key1 not in qid_graph:
        qid_graph[hash_key1] = [hash_key2]
    elif hash_key1 in qid_graph:
        qid_graph[hash_key1].append(hash_key2)

    if hash_key2 not in qid_graph:
        qid_graph[hash_key2] = [hash_key1]
    elif hash_key2 in qid_graph:
        qid_graph[hash_key2].append(hash_key1)




def get_weighted_edge_score(row):
    q1_words = row["question1"].lower().split()
    q2_words = row["question2"].lower().split()

    # modify this!
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))

    common_words_score = np.sum([weights.get(w, 0) for w in common_words])
    all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum(
        [weights.get(w, 0) for w in q2_words]) - common_words_score

    return common_words_score / all_words_score


def generate_edge_scores(row):
    node1 = row["qid1"]
    node2 = row["qid2"]

    if node1 not in edge_weights:
        edge_weights[node1] = {}

    if node2 not in edge_weights:
        edge_weights[node2] = {}

    edge_weight = get_weighted_edge_score(row)

    edge_weights[node1][node2] = edge_weight
    edge_weights[node2][node1] = edge_weight





def weighted_neighbor_features(row, qid_graph, edge_weights):
    q1_neighbors = qid_graph[row["qid1"]]
    q2_neighbors = qid_graph[row["qid2"]]

    common_neighbors = set(q1_neighbors).intersection(q2_neighbors)

    common_neighbors_weight = sum([edge_weights[row["qid1"]][j] for j in common_neighbors])
    q1_weight = sum([edge_weights[row["qid1"]][j] for j in q1_neighbors])
    q2_weight = sum([edge_weights[row["qid2"]][j] for j in q2_neighbors])

    edge_weight = edge_weights[row["qid1"]][row["qid2"]]

    if q1_weight + q2_weight - common_neighbors_weight == 0:
        return pd.Series({
            "edge_weight": edge_weight,
            "edge_weight_ratio": 0,
            "weighted_neighbor_ratio": 0
        })

    edge_weight_ratio = edge_weight / (q1_weight + q2_weight - common_neighbors_weight)
    weighted_neighbor_ratio = common_neighbors_weight / (q1_weight + q2_weight - common_neighbors_weight)

    return pd.Series({
        "edge_weight": edge_weight,
        "edge_weight_ratio": edge_weight_ratio,
        "weighted_neighbor_ratio": weighted_neighbor_ratio
    })


def pagerank(graph):
    pr = {i: 1 / len(graph) for i in graph}

    for iter in range(0, 20):
        print iter
        for node in graph:
            local_pr = 0

            for neighbor in graph[node]:
                local_pr += pr[neighbor] / len(graph[neighbor])

            pr[node] = 0.15 / len(pr) + 0.85 * local_pr

    return pr


def get_pagerank_value(row, pr):
    return pd.Series({
        "q1_pr": pr[row["question1"]],
        "q2_pr": pr[row["question2"]]
    })


def run_kcore_max(core_filepath):
    df_cores = pd.read_csv(core_filepath, index_col="qid")

    df_cores.index.names = ["qid"]

    df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)

    df_cores[['max_kcore']].to_csv("question_max_kcores.csv")  # with index
    return df_cores


def get_kcores(df_train, df_test):
    core_filepath = "question_kcores.csv"

    NB_CORES = 15

    g = make_nx_graph(df_train, df_test)
    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
    print("df_output.shape:", df_output.shape)

    for k in range(2, NB_CORES + 1):
        print k

        fieldname = "kcore{}".format(k)

        ck = nx.k_core(g, k=k).nodes()
        df_output[fieldname] = 0

        df_output.ix[df_output.qid.isin(ck), fieldname] = k

    df_output.to_csv(core_filepath, index=None)
    run_kcore_max(core_filepath)


def get_train_max_kcore():
    kcore_df = pd.read_csv("question_max_kcores.csv")

    df_train = pd.read_csv('./train.csv').fillna("")

    q1_kcore = pd.merge(df_train, kcore_df, left_on='qid1', right_on='qid', how="inner")
    q2_kcore = pd.merge(df_train, kcore_df, left_on='qid2', right_on='qid', how="inner")

    train_cores = pd.concat([q1_kcore.max_kcore, q2_kcore.max_kcore], axis=1)
    train_cores.columns = ["q1_kcore", "q2_kcore"]

    return train_cores


def get_test_max_kcore():
    kcore_df = pd.read_csv("question_max_kcores.csv")

    df_test = pd.read_csv('./df_test_with_qid.csv').fillna("")

    q1_kcore = pd.merge(df_test, kcore_df, left_on='qid1', right_on='qid', how="inner")
    q2_kcore = pd.merge(df_test, kcore_df, left_on='qid2', right_on='qid', how="inner")

    test_cores = pd.concat([q1_kcore.max_kcore, q2_kcore.max_kcore], axis=1)
    test_cores.columns = ["q1_kcore", "q2_kcore"]

    return test_cores