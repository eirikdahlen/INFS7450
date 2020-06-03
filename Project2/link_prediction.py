import networkx as nx
from node2vec import Node2Vec
from numpy import dot
from numpy.linalg import norm
from node2vec.edges import HadamardEmbedder
from random import choice
from sklearn.neural_network import MLPClassifier


def load_graph(input_file=None, input_list=None):
    """
    Creates a Networkx graph based on a file or a list of edges
    :param input_file: str, filepath
    :param input_list: list, list of edges in tuple format
    :return: NetworkX graph
    """
    G = nx.Graph()
    if input_file:
        with open(input_file, 'r') as file:
            for line in file.readlines():
                line = line.strip().split(" ")
                G.add_edge(line[0], line[1])
    elif input_list:
        G.add_edges_from(input_list)
    return G


def merge_data_sets(filepath_pos, filepath_neg, labelling=False):
    """
    Merge the negative and positive datasets
    Add labelling if needed
    :param filepath_pos: str, filepath to the positive edges
    :param filepath_neg: str, filepath to the negative edges
    :param labelling: boolean, if it should create a label list
    :return: two lists, one for the merged dataset and one label set
    """
    data_set = []
    labels = []
    with open(filepath_pos, 'r') as pos_file:
        for line in pos_file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
            if labelling:
                labels.append(1)
    with open(filepath_neg, 'r') as neg_file:
        for line in neg_file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
            if labelling:
                labels.append(0)
    return data_set, labels


def get_neighbors(graph):
    """
    Get the neighbors of each node from a NetworkX graph in a dictionary
    :param graph: NetworkX graph
    :return: dict, key=node, value=set of neighbour nodes
    """
    neighbor_dict = dict()
    for node in graph:
        neighbor_dict[node] = set(graph[node])
    return neighbor_dict


def jaccard_similarity(validation_pairs, neighbors):
    """
    Calculate jaccard similarity
    :param validation_pairs: list, The node pairs in the validation/test set
    :param neighbors: dict, neighbor dict
    :return: dict, scores, key=node pair, value=jaccard similarity
    """
    scores = dict()
    for pair in validation_pairs:
        author1 = pair[0]
        author2 = pair[1]
        intersection = neighbors[author1].intersection(neighbors[author2])
        union = neighbors[author1].union(neighbors[author2])
        scores[author1 + ' ' + author2] = len(intersection) / len(union)
    return scores


def adamic_adar(G, edges):
    """
    Calculate Adamic Adar index
    :param G: NetworkX graph, the training graph
    :param edges: list, validation/test pairs
    :return: dict, scores, key=node pair, value=Adamic-Adar similarity
    """
    scores = dict()
    adamic_adar_index = nx.adamic_adar_index(G, edges)
    for author1, author2, index in adamic_adar_index:
        scores[author1 + ' ' + author2] = index
    return scores


def cosine_similarity(author1, author2):
    return dot(author1, author2) / (norm(author1) * norm(author2))


def embedding_similarity(model, validation_pairs):
    """
    Calculate similarity of node embeddings using cosine similarity
    :param model: node2vec model
    :param validation_pairs: list, validation/test set of pairs
    :return: dict, scores, key=node pair, value=Cosine Similarity
    """
    scores = dict()
    for pair in validation_pairs:
        author1 = pair[0]
        author2 = pair[1]
        scores[author1 + ' ' + author2] = cosine_similarity(model.wv[author1], model.wv[author2])
    return scores


def classify_embeddings(model, training_set, test_set, training_labels):
    """
    Calculate scores using edge embeddings and a binary classifier
    :param model: Node2vec model
    :param training_set: list, whole training set
    :param test_set: list, whole test set
    :param training_labels: list, labels for each pair in training set
    :return: dict, scores, key=node pair, value=probability for being labelled 1
    """
    print("Embedding...")
    # Using Hadamard product for the embedding vectors
    edge_embeddings = HadamardEmbedder(keyed_vectors=model.wv)
    x_train_embedded = [edge_embeddings[pair] for pair in training_set]
    x_test = [edge_embeddings[pair] for pair in test_set]

    # Using MLPClassifier from sklearn as binary classifier
    classifier = MLPClassifier(random_state=1)
    print("Classifying...")
    classify = classifier.fit(x_train_embedded, training_labels)

    predict = classify.predict_proba(x_test)
    score = dict()
    for i in range(len(test_set)):
        node1 = test_set[i][0]
        node2 = test_set[i][1]
        score[node1 + ' ' + node2] = predict[i][1]
    return score

def node2vec_embedding(
    G_training,
    dimensions=64,
    walk_length=10,
    num_walks=10,
    p=1,
    q=1.2
):
    """
    Train a node2vec model using word2vec, skip-gram and negative sampling
    :param G_training: NetworkX Graph, training set
    :param dimensions: int, dimensions of vector
    :param walk_length: int, length of each walk
    :param num_walks: int, number of walks
    :param p: float, return parameter
    :param q: float, in-out parameter
    :return: node2vec model
    """
    node2vec = Node2Vec(
        G_training,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q
    )
    print("Fitting node2vec model...")
    # Using skip-gram algorithm and negative sampling
    # instead of hierarchical softmax
    model = node2vec.fit(window=5, min_count=1, sg=1, hs=0)
    return model


def get_sorted_top_k(results, top_k=100):
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]


def get_reversed_pair(pair):
    pair_list = pair[0].split(" ")
    return pair_list[1] + ' ' + pair_list[0]


def evaluate(calculated_results, ground_truth, measure=''):
    """
    Calculates the accuracy of the results
    :param calculated_results: list, top-100 link predictions
    :param ground_truth: list, 100 positive links
    :param measure: str, what kind of measure used
    """
    score = 0
    truth_pairs = dict.fromkeys(pair[0] + ' ' + pair[1]
                                for pair in ground_truth.edges())
    for pair in calculated_results:
        pair_reversed = get_reversed_pair(pair)
        if pair[0] in truth_pairs or pair_reversed in truth_pairs:
            score += 1
    accuracy = (score / len(truth_pairs)) * 100
    print("The accuracy is %f %% when using measure: %s" % (accuracy, measure))


def generate_neg_links(
        G_train,
        G_val_pos,
        G_val_neg,
        G_test,
        length_multiplier=5
):
    """
    Generate negative node pairs that is not in any of the other datasets
    :param G_train: NetworkX graph, training set
    :param G_val_pos: NetworkX graph, positive validation set
    :param G_val_neg: NetworkX graph, negative validation set
    :param G_test: NetworkX graph, test set
    :param length_multiplier: int, how many times the new dataset should be bigger
    :return: list, negative links generated
    """
    neg_links = []
    size = 0
    length = len(G_train.edges()) * length_multiplier
    while size < length:
        node1 = choice(list(G_train.nodes()))
        node2 = choice(list(G_train.nodes()))
        if not G_train.has_edge(node1, node2) \
                and not G_val_pos.has_edge(node1, node2) \
                and not G_val_neg.has_edge(node1, node2) \
                and not G_test.has_edge(node1, node2):
            neg_links.append(node1 + ' ' + node2 + '\n')
            size += 1
    return neg_links


def process_dict(top_100_list):
    results = []
    for pair in top_100_list:
        results.append(pair[0] + '\n')
    return results


def write_to_file(input, filename):
    with open(filename, 'w') as file:
        file.writelines(input)


def main():
    # --------------- PREPARE DATA ------------------
    files = {'TRAINING': 'data/training.txt',
             'VAL_POS': 'data/val_positive.txt',
             'VAL_NEG': 'data/val_negative.txt',
             'TEST': 'data/test.txt',
             'EXAMPLE': 'data/example.txt',
             'TRAIN_NEG_5': 'data/training_negative5.txt',
             'TRAIN_NEG_1': 'data/training_negative1.txt'}
    G_training = load_graph(files['TRAINING'])
    G_val_pos = load_graph(files['VAL_POS'])
    G_val_neg = load_graph(files['VAL_NEG'])
    G_test = load_graph(files['TEST'])
    test_set = [pair for pair in G_test.edges()]
    # neg_links = generate_neg_links(G_training, G_val_pos, G_val_neg, G_test, length_multiplier=1)
    # write_to_file(neg_links, "data/training_negative1.txt")
    validation_pairs, _ = merge_data_sets(
        files['VAL_POS'], files['VAL_NEG'], labelling=True
    )
    training_set, training_labels = merge_data_sets(
        files['TRAINING'], files['TRAIN_NEG_5'], labelling=True
    )
    G_train_complete = load_graph(input_list=training_set)

    # ------------ NEIGHBORHOOD BASED METHODS ---------------
    neighbors = get_neighbors(G_training)

    jaccard = jaccard_similarity(validation_pairs, neighbors)
    top_100_jaccard = get_sorted_top_k(jaccard)
    evaluate(top_100_jaccard, G_val_pos, 'Jaccard')

    adamic = adamic_adar(G_training, validation_pairs)
    top_100_adamic = get_sorted_top_k(adamic)
    evaluate(top_100_adamic, G_val_pos, 'Adamic Adar')

    # ----------------- EMBEDDING ----------------------
    n2v_model_classifier = node2vec_embedding(G_train_complete)
    n2v_model_cos_sim = node2vec_embedding(G_training)
    classified_score = classify_embeddings(n2v_model_classifier, training_set, validation_pairs, training_labels)
    embedding_sim = embedding_similarity(n2v_model_cos_sim, validation_pairs)

    top_100_n2v_classifier = get_sorted_top_k(classified_score)
    top_100_n2v_sim = get_sorted_top_k(embedding_sim)

    evaluate(top_100_n2v_classifier, G_val_pos, 'Node2Vec Classifier')
    evaluate(top_100_n2v_sim, G_val_pos, 'Node2Vec Cos-Sim')

    # ----------------- TEST SET --------------------
    # Because Adamic Adar gave the best results on the validation set,
    # I will use it to do link predictions on the test set as well
    adamic = adamic_adar(G_training, test_set)
    top_100_adamic = get_sorted_top_k(adamic)
    processed_result = process_dict(top_100_adamic)
    write_to_file(processed_result, 'result.txt')


if __name__ == '__main__':
    main()
