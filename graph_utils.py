import pandas as pd


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

