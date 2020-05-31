import networkx as nx

def load_graph(input_file):
    G = nx.Graph()
    with open(input_file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            G.add_edge(line[0], line[1])
    return G

def merge_validation_sets(filepath_pos, filepath_neg):
    validation_set = []
    with open(filepath_pos, 'r') as pos_file:
        for line in pos_file.readlines():
            line = line.strip().split(" ")
            validation_set.append((line[0], line[1]))
    with open(filepath_neg, 'r') as neg_file:
        for line in neg_file.readlines():
            line = line.strip().split(" ")
            validation_set.append((line[0], line[1]))
    return validation_set

def get_neighbors(graph):
    neighbor_dict = dict()
    for node in graph:
        neighbor_dict[node] = set(graph[node])
    return neighbor_dict

def jaccard_similarity(validation_pairs, neighbors):
    scores = dict()
    for pair in validation_pairs:
        author1 = pair[0]
        author2 = pair[1]
        intersection = neighbors[author1].intersection(neighbors[author2])
        union = neighbors[author1].union(neighbors[author2])
        scores[author1 + ' ' + author2] = len(intersection) / len(union)
    return scores

def adamic_adar(G, edges):
    scores = dict()
    adamic_adar_index = nx.adamic_adar_index(G, edges)
    for author1, author2, index in adamic_adar_index:
        scores[author1 + ' ' + author2] = index
    return scores

def get_sorted_top_k(results, top_k=100):
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

def get_reversed_pair(pair):
    pair_list = pair[0].split(" ")
    return pair_list[1] + ' ' + pair_list[0]

def evaluate(calculated_results, ground_truth, measure=''):
    score = 0
    truth_pairs = dict.fromkeys(pair[0] + ' ' + pair[1] for pair in ground_truth.edges())
    for pair in calculated_results:
        pair_reversed = get_reversed_pair(pair)
        if pair[0] in truth_pairs or pair_reversed in truth_pairs:
            score += 1
    accuracy = (score/len(truth_pairs))*100
    print("The accuracy is %f %% when using measure: %s" % (accuracy, measure))
    return accuracy



def main():
    files = {'TRAINING': 'data/training.txt',
             'VAL_POS': 'data/val_positive.txt',
             'VAL_NEG': 'data/val_negative.txt',
             'TEST': 'data/test.txt',
             'EXAMPLE': 'data/example.txt'}
    G_training = load_graph(files['TRAINING'])
    G_val_pos = load_graph(files['VAL_POS'])
    validation_pairs = merge_validation_sets(files['VAL_POS'], files['VAL_NEG'])
    neighbors = get_neighbors(G_training)
    jaccard = jaccard_similarity(validation_pairs, neighbors)
    adamic = adamic_adar(G_training, validation_pairs)
    top_100_jaccard = get_sorted_top_k(jaccard)
    top_100_adamic = get_sorted_top_k(adamic)
    accuracy = evaluate(top_100_jaccard, G_val_pos, 'Jaccard')
    accuracy = evaluate(top_100_adamic, G_val_pos, 'Adamic Adar')






if __name__ == '__main__':
    main()
