import networkx as nx
from collections import deque
import time
import numpy as np


def create_graph(input_file):
    file = open(input_file, 'r')
    G = nx.Graph()
    for line in file.readlines():
        line = line.strip()
        line = line.split(" ")
        G.add_edge(int(line[0]), int(line[1]))
    return G


def bfs(G, source):
    """
    Breadth First Search in undirected graph, storing values for sigma
    to use in betweenness centrality calculation
    :param G: The graph
    :param source: The source node we are doing BFS on
    :return: The stack, predecessors dict and sigma of this source node
    """
    stack = []
    queue = deque([source])
    predecessors = {}  # A dict is used for faster lookups
    for n in G:  # Fill predecessor-dict with empty lists for later
        predecessors[n] = []
    sigma = dict.fromkeys(G, 0.0)  # All sigmas are 0.0 initially
    distance = dict.fromkeys(G, -1.0)  # All distances are -1.0 initially
    sigma[source] = 1.0
    distance[source] = 0.0

    while len(queue) > 0:
        vertex = queue.popleft()
        stack.append(vertex)
        for neighbour in G[vertex]:   # Go through all neighbours of the node
            if distance[neighbour] < 0:  # If not visited
                queue.append(neighbour)
                distance[neighbour] = distance[vertex] + 1
            if distance[neighbour] == distance[vertex] + 1:
                # This means that "vertex" is a predecessor of "neighbour" on a
                # shortest path from source node to "neighbour"
                sigma[neighbour] += sigma[vertex]
                predecessors[neighbour].append(vertex)
    return stack, predecessors, sigma


def accumulation_backprop(results, source, stack, predecessors, sigma):
    """
    Working backwards with the stack created in BFS.
    Calculate the delta and add that to the results dict.

    :param results: Dictionary of results, to be altered
    :param source: The node we are backproping for
    :param stack: Stack of nodes
    :param predecessors: Predecessor dictionary
    :param sigma: Number of shortest paths from source node to vertex
    :return: Updated result after the backprop
    """
    delta = dict.fromkeys(stack, 0.0)  # Establish the delta values in a dict
    while len(stack) > 0:
        neighbour = stack.pop()
        for vertex in predecessors[neighbour]:
            delta[vertex] += ((sigma[vertex] / sigma[neighbour])
                              * (1 + delta[neighbour]))
        if neighbour != source:
            results[neighbour] += delta[neighbour]
    return results


def betweenness_centrality(G, normalize=True, top_k=10):
    """
    Implementation of Brandes algorithm for calculating betweenness centrality
    1) Iterate through all nodes in the network
    2) Do BFS for each node, calculate distances and shortest paths count, push the nodes to a stack
    3) Visit nodes in stack and calculate betweenness centrality
    4) Normalize results

    :param G: NetworkX graph
    :param normalize: boolean, if the results should be normalized or not
    :param top_k: Top k numbers of results that should be returned
    :return: Dictionary of top-k results with highest betweenness centrality
    """
    nodes = G
    # Create a dict where {node_1: 0.0, node_2: 0.0, ...etc}
    results = dict.fromkeys(G, 0.0)
    for source in nodes:
        stack, predecessors, sigma = bfs(G, source)  # Run BFS for each node
        results = accumulation_backprop(
            results,
            source,
            stack,
            predecessors,
            sigma)  # Do the backpropagation accumulation
    top_k_results_sorted = sorted_dictionary_on_value(
        results)[:top_k]  # Sort results and fetch top-k results
    if normalize:
        return normalize_results(
            top_k_results_sorted, len(results), "b")
    return top_k_results_sorted


def page_rank(
        G,
        alpha=0.85,
        beta=0.15,
        max_iterations=100,
        tolerance=1.0e-4,
        normalize=True,
        top_k=10):
    """
    Implementation of PageRank algorithm based on formula from slides in INFS7450.
    1) Assign adjacency matrix as a dict of dicts: this store a node as key and
    its neighbours/outgoing links as values in the dictionary.
    2) Create degree matrix and invert it
    3) Create c matrix where the entries are 1.0
    4) Do iterations of pagerank where the next c is calculated based on previous c and
    the neighbours' outgoing links, as well as alpha and beta
    5) Stop when the L2 norm is small enough

    :param G: An undirected networkX graph
    :param alpha: damping factor
    :param beta: personalization factor
    :param max_iterations: maximal number of iterations that should be performed
    :param tolerance: The threshold for when to stop iterations
    :param normalize: boolean, normalizes the results if True
    :param top_k: Top k numbers of results that should be returned
    :return: A dictionary containing nodes and their pagerank values
    """
    A = nx.to_dict_of_dicts(G)  # Use that A=A^T in an undirected graph
    N = len(A)
    D = dict(G.degree)  # Create the degree matrix
    for node in D.keys():
        D[node] = 1 / D[node]  # The inverse of D is the reciprocals for each node
    c = dict.fromkeys(A, 1.0)  # The start vector, all values are 1.0

    for i in range(max_iterations):
        print("Runde: ", i)
        c_previous = c
        c = dict.fromkeys(c_previous, 0)
        for node in c:
            for neighbour in A[node]:
                # Don't need to multiply by A[node][neighbour] here as they will
                # be 1.0 and not change the results
                c[neighbour] += (alpha * D[node] * c_previous[node])
            c[node] += beta

        err = (np.linalg.norm([c[n] - c_previous[n]
                               for n in c]))  # The L2 norm
        if err < N * tolerance:
            c = sorted_dictionary_on_value(c)[:top_k]
            if normalize:
                return normalize_results(c, N, "p")
            return c
    print("Did not converge")
    return {}


def normalize_results(results, n, measure):
    """
    Normalize the results of the calculation
    :param results: top-k results from the calculation
    :param n: number of nodes in the network
    :param measure: Which measure calculation is used, b=betweenness, p=pagerank
    :return: top-k results normalized
    """
    normalized_results = []
    for r in results:
        if measure == "b":
            normalized_results.append((r[0], r[1] * (1 / ((n - 1) * (n - 2)))))
        elif measure == "p":
            normalized_results.append((r[0], r[1] / n))
        else:
            break
    return normalized_results


def sorted_dictionary_on_value(dictionary, reverse=True):
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=reverse)


def test_ground_truth(node_measure_function, top_k=10):
    return sorted_dictionary_on_value(node_measure_function)[:top_k]


def results_to_file(results, mode="w"):
    """
    Writes the results to a file.
    File consists of nodes, separated with " ".
    First row is results from betweenness, second row from pagerank
    :param results: Result array of top_k nodes
    :param mode: mode for the file operations
    """
    file = open("results.txt", mode)
    if mode == "a":
        file.write("\n")
    for node, value in results:
        file.write(str(node) + " ")
    file.close()
    print("File successfully written")


def main():
    G = create_graph('3_data.txt')
    ticks1 = time.time()
    # ----------------- Betweenness Centrality -----------------
    betweenness = betweenness_centrality(G)
    print(betweenness)
    ticks2 = time.time()
    print(test_ground_truth(nx.betweenness_centrality(G)))
    print("Time Betweenness: ", ticks2 - ticks1)

    # ----------------- PageRank -----------------
    ticks3 = time.time()
    pagerank = page_rank(G)
    print(pagerank)
    ticks4 = time.time()
    print("Time PageRank: ", ticks4 - ticks3)
    print(test_ground_truth(nx.pagerank(G)))

    # # ----------------- Write results to file -----------------
    results_to_file(betweenness)
    results_to_file(pagerank, mode="a")


main()

