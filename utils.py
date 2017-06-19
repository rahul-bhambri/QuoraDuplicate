import pandas as pd


def submit(p_test):
    sub = pd.DataFrame()

    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test

    sub.to_csv('simple_xgb.csv', index=False)


def get_inverse_freq(inverse_freq, count, min_count=2):
    if count < min_count:
        return 0
    else:
        return inverse_freq


def get_tf(text):
    tf = {}

    for word in text:
        tf[word] = text.count(word) / len(text)

    return tf


def tuple_similarity(q1_words, q2_words):
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = len(set(q1_words).intersection(set(q2_words)))
    all_words = len(set(q1_words).union(set(q2_words)))

    return common_words / all_words


def get_ne_score(row):
    q1_words = str(row.question1).lower().split()
    q2_words = str(row.question2).lower().split()

    # all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words])

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = set([str(i) for i in q1_ne])
    q2_ne = set([str(i) for i in q2_ne])

    if len(q1_ne) == 0:
        q1_ne_ratio = 0
    else:
        q1_ne_ratio = len(q1_ne) / len(q1_words)

    if len(q2_ne) == 0:
        q2_ne_ratio = 0
    else:
        q2_ne_ratio = len(q2_ne) / len(q2_words)

    common_ne = list(q1_ne.intersection(q2_ne))
    # common_ne_weights = np.sum([weights.get(w, 0) for w in common_ne])

    if len(q1_ne) + len(q2_ne) == 0:
        common_ne_score = 0
    else:
        common_ne_score = len(common_ne) / (len(q1_words) + len(q2_words) - len(common_ne))

    return pd.Series({
        "q1_ne_ratio": q1_ne_ratio,
        "q2_ne_ratio": q2_ne_ratio,
        "ne_diff": abs(q1_ne_ratio - q2_ne_ratio),
        "ne_score": common_ne_score
    })


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


def weighted_neighbor_features(row):
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


def pagerank():
    pr = {i: 1 / len(graph) for i in graph}

    for iter in range(0, 20):
        print iter
        for node in graph:
            local_pr = 0

            for neighbor in graph[node]:
                local_pr += pr[neighbor] / len(graph[neighbor])

            pr[node] = 0.15 / len(pr) + 0.85 * local_pr

    return pr


def get_pagerank_value(row):
    return pd.Series({
        "q1_pr": pr[row["question1"]],
        "q2_pr": pr[row["question2"]]
    })


def basic_nlp(row):
    # q1_tf = get_tf(q1_words)
    # q2_tf = get_tf(q2_words)

    q1_words = str(row.question1).lower().split()
    q2_words = str(row.question2).lower().split()

    # modify this!
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))

    common_words_score = np.sum([weights.get(w, 0) for w in common_words])
    all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum(
        [weights.get(w, 0) for w in q2_words]) - common_words_score

    hamming_score = sum(1 for i in zip(q1_words, q2_words) if i[0] == i[1]) / max(len(q1_words), len(q2_words))

    jacard_score = len(common_words) / (len(q1_words) + len(q2_words) - len(common_words))
    cosine_score = len(common_words) / (pow(len(q1_words), 0.5) * pow(len(q2_words), 0.5))

    bigrams_q1 = set(ngrams(q1_words, 2))
    bigrams_q2 = set(ngrams(q2_words, 2))
    common_bigrams = len(bigrams_q1.intersection(bigrams_q2))
    if common_bigrams == 0:
        bigram_score = 0
    else:
        bigram_score = common_bigrams / (len(bigrams_q1.union(bigrams_q2)))

    trigrams_q1 = set(ngrams(q1_words, 3))
    trigrams_q2 = set(ngrams(q2_words, 3))
    common_trigrams = len(trigrams_q1.intersection(trigrams_q2))
    if common_trigrams == 0:
        trigram_score = 0
    else:
        trigram_score = common_trigrams / (len(trigrams_q1.union(trigrams_q2)))

        # sequence1 = get_word_bigrams(q1_words)
    # sequence2 = get_word_bigrams(q2_words)

    # try:
    #     simhash_diff = Simhash(sequence1).distance(Simhash(sequence2))/64
    # except:
    #     simhash_diff = 0.5

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = set([str(i) for i in q1_ne])
    q2_ne = set([str(i) for i in q2_ne])

    if len(q1_ne) == 0:
        q1_ne_ratio = 0
    else:
        q1_ne_ratio = len(q1_ne) / len(q1_words)

    if len(q2_ne) == 0:
        q2_ne_ratio = 0
    else:
        q2_ne_ratio = len(q2_ne) / len(q2_words)

    common_ne = list(q1_ne.intersection(q2_ne))
    # common_ne_weights = np.sum([weights.get(w, 0) for w in common_ne])

    if len(q1_ne) + len(q2_ne) == 0:
        common_ne_score = 0
    else:
        common_ne_score = len(common_ne) / (len(q1_words) + len(q2_words) - len(common_ne))

    pos_hash = {}
    common_pos = []

    for word in q1:
        if word.tag_ not in pos_hash:
            pos_hash.update({word.tag_: [word.text]})
        else:
            pos_hash[word.tag_].append(word.text)

    for word in q2:
        if word.tag_ not in pos_hash:
            continue
        if word.text in pos_hash[word.tag_]:
            common_pos.append(word.text)

    common_pos_score = np.sum([weights.get(w, 0) for w in common_pos])
    all_pos_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum(
        [weights.get(w, 0) for w in q2_words]) - common_pos_score

    q1_pronouns_count = 0
    q2_pronouns_count = 0

    for word in q1:
        if str(word.tag_) == "PRP":
            q1_pronouns_count += 1

    for word in q2:
        if str(word.tag_) == "PRP":
            q2_pronouns_count += 1

    pronouns_diff = abs(q1_pronouns_count - q2_pronouns_count)

    q1_nc = q1.noun_chunks
    q2_nc = q2.noun_chunks

    q1_nc = set([str(i) for i in q1_nc])
    q2_nc = set([str(i) for i in q2_nc])

    common_nc = len(q1_nc.intersection(q2_nc))

    if len(q1_nc) + len(q2_nc) == 0:
        common_nc_score = 0
    else:
        common_nc_score = common_nc / (len(q1_nc) + len(q2_nc) - common_nc)

    fw_q1 = q1_words[0]
    fw_q2 = q2_words[0]

    if fw_q1 == fw_q2 and fw_q1 in question_types:
        question_type_same = 1
    else:
        question_type_same = 0

    try:
        q1_quotes = len(re.findall(r'\"(.+?)\"', row["question1"]))
    except:
        q1_quotes = 0

    try:
        q2_quotes = len(re.findall(r'\"(.+?)\"', row["question2"]))
    except:
        q2_quotes = 0

    # if len(q1_ne) == 0:
    #     q1_ne_hash_freq = 1
    # else:
    #     hash_key1 = hash("-".join(set([str(i).lower() for i in q1_ne])))

    #     if hash_key1 not in hash_table_ne:
    #         q1_ne_hash_freq = 1
    #     else:
    #         q1_ne_hash_freq = hash_table_ne[hash_key1]

    # if len(q2_ne) == 0:
    #     q2_ne_hash_freq = 1
    # else:
    #     hash_key2 = hash("-".join(set([str(i).lower() for i in q2_ne])))

    #     if hash_key2 not in hash_table_ne:
    #         q2_ne_hash_freq = 1
    #     else:
    #         q2_ne_hash_freq = hash_table_ne[hash_key2]

    try:
        q1_sents = len(nltk.tokenize.sent_tokenize(row.question1))
    except:
        q1_sents = 1
    try:
        q2_sents = len(nltk.tokenize.sent_tokenize(row.question2))
    except:
        q2_sents = 1

    q1_exclaim = sum([1 for i in str(row.question1) if i == "!"])
    q2_exclaim = sum([1 for i in str(row.question2) if i == "!"])

    q1_question = sum([1 for i in str(row.question1) if i == "?"])
    q2_question = sum([1 for i in str(row.question2) if i == "?"])

    hash_key1 = hash(str(row["question1"]).lower())
    if hash_key1 in hash_table:
        q1_hash_freq = hash_table[hash_key1]
    else:
        q1_hash_freq = 1

    hash_key2 = hash(str(row["question2"]).lower())
    if hash_key2 in hash_table:
        q2_hash_freq = hash_table[hash_key2]
    else:
        q2_hash_freq = 1

    # if hash_key1 in pos_hash_table:
    #     q1_dup_ratio = pos_hash_table[hash_key1]/q1_hash_freq
    # else:

    #     q1_dup_ratio = 0

    # if hash_key2 in pos_hash_table:
    #     q2_dup_ratio = pos_hash_table[hash_key2]/q2_hash_freq
    # else:
    #     q2_dup_ratio = 0

    spacy_sim = q1.similarity(q2)

    return pd.Series({

        "weighted_word_match_ratio": common_words_score / all_words_score,
        "weighted_word_match_diff": all_words_score - common_words_score,
        "weighted_word_match_sum": common_words_score,
        "jacard_score": jacard_score,
        "hamming_score": hamming_score,
        "cosine_score": cosine_score,
        "bigram_score": bigram_score,
        "trigram_score": trigram_score,
        "pos_score": common_pos_score / all_pos_score,
        # "simhash_diff": simhash_diff,
        "question_type_same": question_type_same,
        "q1_stops": len(set(q1_words).intersection(stops)) / len(q1_words),
        "q2_stops": len(set(q2_words).intersection(stops)) / len(q2_words),
        "q1_len": len(str(row.question1)),
        "q2_len": len(str(row.question2)),
        "len_diff": abs(len(str(row.question1)) - len(str(row.question2))),
        "len_avg": (len(str(row.question1)) + len(str(row.question2))) / 2,
        "q1_sents": q1_sents,
        "q2_sents": q2_sents,
        "sents_diff": abs(q1_sents - q2_sents),
        "q1_words": len(q1_words),
        "q2_words": len(q2_words),
        "words_diff": abs(len(q1_words) - len(q2_words)),
        "words_avg": (len(q1_words) + len(q2_words)) / 2,
        "q1_caps_count": sum([1 for i in str(row.question1) if i.isupper()]),
        "q2_caps_count": sum([1 for i in str(row.question2) if i.isupper()]),
        "q1_exclaim": q1_exclaim,
        "q2_exclaim": q2_exclaim,
        "exclaim_diff": abs(q1_exclaim - q2_exclaim),
        "q1_question": q1_question,
        "q2_question": q2_question,
        "question_diff": abs(q1_question - q2_question),
        "ne_score": common_ne_score,
        "nc_score": common_nc_score,
        "q1_ne_ratio": q1_ne_ratio,
        "q2_ne_ratio": q2_ne_ratio,
        "ne_diff": abs(q1_ne_ratio - q2_ne_ratio),
        "q1_quotes": q1_quotes,
        "q2_quotes": q2_quotes,
        "quotes_diff": abs(q1_quotes - q2_quotes),
        # "q1_ne_hash_freq": q1_ne_hash_freq,
        # "q2_ne_hash_freq": q2_ne_hash_freq,
        # "chunk_hash_diff": abs(q1_ne_hash_freq - q2_ne_hash_freq),
        "q1_hash_freq": q1_hash_freq,
        "q2_hash_freq": q2_hash_freq,
        "q_freq_avg": (q1_hash_freq + q2_hash_freq) / 2,
        "freq_diff": abs(q1_hash_freq - q2_hash_freq),
        "spacy_sim": spacy_sim,
        "q1_pronouns_count": q1_pronouns_count,
        "q2_pronouns_count": q2_pronouns_count,
        "pronouns_diff": pronouns_diff
        # "q1_dup_ratio": q1_dup_ratio,
        # "q2_dup_ratio": q2_dup_ratio,
        # "q1_q2_dup_ratio_avg": (q1_dup_ratio + q2_dup_ratio)/2
    })





# def pos_neighbor_intersection(row):

#     if row["question1"] in pos_graph and row["question2"] in pos_graph:
#         q1_neighbors = pos_graph[row["question1"]]
#         q2_neighbors = pos_graph[row["question2"]]

#         common_neighbors = set(q1_neighbors).intersection(q2_neighbors)

#         return len(common_neighbors)/(len(q1_neighbors) + len(q2_neighbors) - len(common_neighbors))

#     else:
#         return 0

def get_word_bigrams(words):
    ngrams = []

    for i in range(0, len(words)):
        if i > 0:
            ngrams.append("%s %s" % (words[i - 1], words[i]))

    return ngrams


def generate_hash_freq(row):
    hash_key1 = hash(row["question1"].lower())
    hash_key2 = hash(row["question2"].lower())

    if hash_key1 not in hash_table:
        hash_table[hash_key1] = 1
    else:
        hash_table[hash_key1] += 1

    if hash_key2 not in hash_table:
        hash_table[hash_key2] = 1
    else:
        hash_table[hash_key2] += 1


def generate_duplicate_freq(row):
    hash_key1 = hash(row["question1"].lower())
    hash_key2 = hash(row["question2"].lower())

    if hash_key1 not in pos_hash_table and row["is_duplicate"] == 1:
        pos_hash_table[hash_key1] = 1
    elif hash_key1 not in pos_hash_table and row["is_duplicate"] == 0:
        pos_hash_table[hash_key1] = 0
    elif hash_key1 in pos_hash_table and row["is_duplicate"] == 1:
        pos_hash_table[hash_key1] += 1
    # elif hash_key1 in pos_hash_table and row["is_duplicate"] == 0:
    #     pass

    if hash_key2 not in pos_hash_table and row["is_duplicate"] == 1:
        pos_hash_table[hash_key2] = 1
    elif hash_key2 not in pos_hash_table and row["is_duplicate"] == 0:
        pos_hash_table[hash_key2] = 0
    elif hash_key2 in pos_hash_table and row["is_duplicate"] == 1:
        pos_hash_table[hash_key2] += 1
        # elif hash_key1 in pos_hash_table and row["is_duplicate"] == 0:
        #     pass



def augment_rows():
    new_graph = graph

    for q1 in graph:

        q2_list = graph[q1]

        for i in q2_list:
            for j in q2_list:
                if i != j:
                    if j not in graph[i]:
                        new_graph[i].append(j)

                        # new_df_train = df_train[["question1", "question2", "is_duplicate"]]

                        # for i in new_graph:


def generate_ne_freq(row):
    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = "-".join(set([str(i).lower() for i in q1_ne]))
    q2_ne = "-".join(set([str(i).lower() for i in q2_ne]))

    hash_key1 = hash(q1_ne)
    hash_key2 = hash(q2_ne)

    if hash_key1 not in hash_table_ne:
        hash_table_ne[hash_key1] = 1
    else:
        hash_table_ne[hash_key1] += 1

    if hash_key2 not in hash_table_ne:
        hash_table_ne[hash_key2] = 1
    else:
        hash_table_ne[hash_key2] += 1


def oversample(x_train):
    neg_train = x_train[x_train.is_duplicate == 0]
    pos_train = x_train[x_train.is_duplicate == 1]

    # Oversampling negative class
    #     p = 0.1742
    p = 0.165

    #     scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1 #How much times greater is the train ratio than actual

    #     while scale > 1:
    #         neg_train = pd.concat([neg_train, neg_train])
    #         scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

    #     scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

    #     while ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1 > 1e-6:
    #         neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    #         scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

    # Number of neg_trains that should be added -- (((len(pos_train)/p) - len(x_train))/len(neg_train) - 1)*len(neg_train)
    neg_train = pd.concat([neg_train, neg_train])
    #     neg_train = pd.concat([neg_train, neg_train[:197531]]) #When p is 0.1742
    neg_train = pd.concat([neg_train, neg_train[:245307]])  # When p is 0.165

    return pd.concat([pos_train, neg_train])


# When plotted a histogram of degrees, only -1,1 and 2 are observed. Which means either you're max 2 degree separated or you're separate(with 5 as a cutoff).
# Add (number of second degree connections) and its intersection as a feature
def bfs(q_node, q_search, separation):
    if separation > 5:
        return -1

    if len(graph[q_node]) > 0:

        shortest_res = 32768

        if q_search in graph[q_node]:
            return separation
        else:

            for i, j in enumerate(graph[q_node]):

                if i > 5:
                    return shortest_res

                bfs_res = bfs(j, q_search, separation + 1)

                if bfs_res != -1 and bfs_res < shortest_res:
                    shortest_res = bfs_res

            return shortest_res

    else:
        return -1


def initialize_bfs(row):
    q1 = row["question1"]
    q2 = row["question2"]

    shortest_res = 32768

    for i in graph[q1]:
        if i != q2:
            res = bfs(i, q2, 1)

            if res != -1 and res < shortest_res:
                shortest_res = res

    if shortest_res == 32768:
        return -1
    else:
        return shortest_res


def augment_test(row):
    global new_df_test

    # map q1 with dups of q2
    if row["question2"] in pos_graph:
        new_rows = pd.DataFrame()
        q2_dups = pos_graph[row["question2"]]
        new_rows["question2"] = [i for i in q2_dups]
        new_rows["question1"] = row["question1"]
        new_rows["test_id"] = row["test_id"]
        new_df_test = pd.concat([new_df_test, new_rows])

    # map q2 with dups of q1
    if row["question1"] in pos_graph:
        new_rows = pd.DataFrame()
        q1_dups = pos_graph[row["question1"]]
        new_rows["question1"] = [i for i in q1_dups]
        new_rows["question2"] = row["question2"]
        new_rows["test_id"] = row["test_id"]
        new_df_test = pd.concat([new_df_test, new_rows])


def get_pr_ratio(row):
    if row["q1_pr"] != 0 and row["q2_pr"] != 0:
        return max(row["q1_pr"] / row["q2_pr"], row["q2_pr"] / row["q1_pr"])
    else:
        return 0


# def run_xgb(x_train, x_valid, y_train, y_valid):
def run_xgb(x_train, x_label):
    # x_train = pd.concat([pos_train, neg_train]) #Concat positive and negative
    # y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist() #Putting in 1 and 0

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.05
    params['max_depth'] = 5
    params['silent'] = 1
    params['min_child_weight'] = 0
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.8
    params['nthread'] = 13
    # params['scale_pos_weight'] = 0.36

    d_train = xgb.DMatrix(x_train, label=x_label)

    watchlist = [(d_train, 'train')]

    bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, verbose_eval=50)

    return bst


def run_rf(x_train, x_label):
    h2o.init(max_mem_size="2G")
    h2o.remove_all()

    rf = H2ORandomForestEstimator(
        model_id="rf_model_1",
        ntrees=500,
        max_depth=10,
        stopping_rounds=2,
        stopping_tolerance=0.01,
        score_each_iteration=True,
        seed=3000000
    )

    hf_train = h2o.H2OFrame(pd.concat([x_train, x_label], axis=1))

    rf.train(hf_train.col_names[:-1], hf_train.col_names[-1], training_frame=hf_train)

    return rf


def run_gbm(x_train, x_label):
    h2o.remove_all()
    h2o.init(max_mem_size="2G")

    gbm = H2OGradientBoostingEstimator(
        ntrees=1000,
        learn_rate=0.3,
        max_depth=10,
        sample_rate=0.7,
        col_sample_rate=0.7,
        stopping_rounds=2,
        stopping_tolerance=0.001,  # 10-fold increase in threshold as defined in rf_v1
        score_each_iteration=True,
        model_id="gbm_covType_v3",
        seed=2000000
    )

    hf_train = h2o.H2OFrame(pd.concat([x_train, x_label], axis=1))

    gbm.train(hf_train.col_names[:-1], hf_train.col_names[-1], training_frame=hf_train)

    return gbm


def run_tsne(pos_train, neg_train, x_test_feat):
    x_train = pd.concat([pos_train, neg_train])  # Concat positive and negative
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()  # Putting in 1 and 0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    df_subsampled = x_train[0:3000]
    X = MinMaxScaler().fit_transform(df_subsampled[['z_len1', 'z_len2', 'z_words1', 'z_words2', 'word_match']])
    # y = y_train['is_duplicate'].values

    tsne = TSNE(
        n_components=3,
        init='random',  # pca
        random_state=101,
        method='barnes_hut',
        n_iter=200,
        verbose=2,
        angle=0.5
    ).fit_transform(X)

    trace1 = go.Scatter3d(
        x=tsne[:, 0],
        y=tsne[:, 1],
        z=tsne[:, 2],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            color=y_train,
            colorscale='Portland',
            colorbar=dict(title='duplicate'),
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.75
        )
    )

    data = [trace1]
    layout = dict(height=800, width=800, title='3d embedding with engineered features')
    fig = dict(data=data, layout=layout)
    py.plot(data, filename='3d_bubble')


def validate(training):
    training_res = training.pop("is_duplicate")
    x_train, x_valid, y_train, y_valid = train_test_split(training, training_res, test_size=0.2, random_state=4242,
                                                          stratify=training_res)

    return (x_train, x_valid, y_train, y_valid)


def get_jarowinkler(row):
    try:
        return distance.get_jaro_distance(row.question1, row.question2, winkler=True, scaling=0.1)
    except:
        return 0


def real_testing(gen_filename, df_with_qs=None, res_file=None):
    # Required for initial setup!
    # row = df_with_qs.progress_apply(basic_nlp, axis = 1)
    old_filename = './old/' + gen_filename
    # row.to_csv(old_filename, index = False)

    # REPLACE ROW WITH DF_MODIFIED
    #     if gen_filename == "x_train.csv":
    #         kcore_features = get_train_max_kcore()
    # # #         s2v_features = pd.read_csv('train_s2v_features.csv')
    # # #         df_with_qid = pd.read_csv('df_train_with_max_qid.csv') #max_qid overfits
    #     else:
    #         kcore_features = get_test_max_kcore()
    # #         s2v_features = pd.read_csv('test_s2v_features.csv')
    # #         df_with_qid = pd.read_csv('df_test_with_max_qid.csv') #max_qid overfits

    # #     row = pd.concat([dataframe, s2v_features], axis = 1)
    # #     row = pd.concat([dataframe, df_with_qid.max_qid], axis = 1)

    #     row = pd.concat([dataframe, kcore_features], axis = 1)

    #     row["neighbor_intersection"] = df_with_qs.apply(neighbor_intersection, axis = 1)

    #     neighbor_feats = df_with_qs.apply(neighbor_features, axis = 1)
    #     row = pd.concat([dataframe, neighbor_feats], axis = 1)

    #     dataframe_modified = dataframe.drop("neighbor_dissimilar", axis = 1)
    #     dataframe_modified = dataframe_modified.drop("neighbor_ratio", axis = 1)

    #     q1_second_degree_freq = df_with_qs.apply(get_q1_second_degree_freq, axis = 1)
    #     q2_second_degree_freq = df_with_qs.apply(get_q2_second_degree_freq, axis = 1)
    #     row["second_degree_avg"] = (q1_second_degree_freq + q2_second_degree_freq)/2
    #     row["second_degree_diff"] = abs(q1_second_degree_freq - q2_second_degree_freq)
    #     row["second_degree_intersection"] = df_with_qs.apply(second_degree_intersection, axis = 1)
    #     row["separation"] = df_with_qs.progress_apply(initialize_bfs, axis = 1)

    #     weighted_neighbor_feats = df_with_qs.progress_apply(weighted_neighbor_features, axis = 1)
    #     weighted_neighbor_feats.index = dataframe.index
    #     row = pd.concat([dataframe, weighted_neighbor_feats], axis = 1)

    #     dataframe_modified["pr_diff"] = abs(dataframe_modified["q1_pr"] - dataframe_modified["q2_pr"])
    #     dataframe_modified["pr_avg"] = (dataframe_modified["q1_pr"] + dataframe_modified["q2_pr"])/2

    #     dataframe_modified["pr_ratio"] = dataframe_modified.apply(get_pr_ratio, axis = 1)

    #     dataframe_modified = pd.read_csv(old_filename).fillna("")
    #     dataframe_modified["pr_neighbor_ratio"] = df_with_qs.progress_apply(pagerank_neighbor_features, axis = 1)

    dataframe_modified = pd.read_csv(old_filename).fillna("")
    dataframe_modified["jarowinkler"] = df_with_qs.progress_apply(get_jarowinkler, axis=1)

    if res_file is None:
        new_filename = "./new/" + gen_filename
    else:
        new_filename = "./new/" + res_file

    dataframe_modified.to_csv(new_filename, index=False)
    # %reset_selective dataframe_modified


def pred_n_submit(x_test, res_filename):
    #     XGB
    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)
    sub = pd.DataFrame()
    sub['is_duplicate'] = p_test
    sub.to_csv(res_filename + "_xgb.csv", index=False)

    #     Keras Dense Neural net
    #     res_keras = model.predict(np.array(x_test))
    #     np.savetxt(res_filename + "_keras.csv", res_keras, delimiter=",", header = "is_duplicate")

    #     H2O Random Forest
    #     hf_test = h2o.H2OFrame(x_test)
    #     rf_pred = rf.predict(hf_test)
    #     h2o.export_file(rf_pred, res_filename + "_rf.csv")

    #   H2O GBM
    #     hf_test = h2o.H2OFrame(x_test)
    #     gbm_pred = gbm.predict(hf_test)
    #     h2o.export_file(gbm_pred, res_filename + "_gbm.csv")


