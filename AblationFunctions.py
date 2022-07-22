# AblationFunctions.py>
import numpy as np
import neuprint
from neuprint import NeuronCriteria as NC
import networkx as nx
import pandas as pd
import copy
from tqdm import tqdm


def preprocess_df(table):
    mb_graph = nx.from_pandas_edgelist(table, source='a.bodyId', target='b.bodyId',
                                       edge_attr=['weight', 'a.type', 'b.type'], create_using=nx.DiGraph())
    mb_ids = np.unique(list(table["a.bodyId"].unique()) + list(table["b.bodyId"].unique()))
    mb_neurons = neuprint.fetch_neurons(NC(bodyId=mb_ids))

    type_dict_mb = {}
    for i in range(mb_neurons[0].shape[0]):
        row = mb_neurons[0].iloc[i]
        type_dict_mb[row.bodyId] = row.type
    nx.set_node_attributes(mb_graph, type_dict_mb, "type")
    return mb_graph, type_dict_mb


def type_count(type_dict_mb):
    df = pd.DataFrame(type_dict_mb, index=[0]).T
    kc_ct = 0
    pn_ct = 0
    mbon_ct = 0
    apl_ct = 0
    orn_ct = 0
    for i in range(df.shape[0]):
        if "PN" in df.iloc[i][0]:
            pn_ct += 1
        if "KC" in df.iloc[i][0]:
            kc_ct += 1
        if "MBON" in df.iloc[i][0]:
            mbon_ct += 1
        if "ORN" in df.iloc[i][0]:
            orn_ct += 1
        if "APL" in df.iloc[i][0]:
            apl_ct += 1


def preferential_detachment(weight_matrix, decay_vector, n_passes, p, spread_rate):
    weights_mask = weight_matrix > 0

    sh = weight_matrix.shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            if i == j:
                weights_mask[i, j] = 1
    osum = np.sum(weight_matrix)
    n_nodes = sh[0]
    avg1, avg2 = [], []
    decay_matrix = weights_mask

    for i in range(n_passes):
        # Transmit decay
        decay_matrix = weights_mask * decay_vector

        # binomial distribution of where to decrement weights

        decay_prob = p * (1 / n_nodes) * weights_mask + (1 - p) * (decay_matrix) / osum

        decay_choose = np.random.random(sh) < decay_prob
        #         if i == 0:
        #             print(np.mean(decay_choose), np.mean(decay_prob), 'decay')
        decay_vector = (decay_vector + np.sum(decay_choose, axis=1)) * spread_rate
        # Maximum of one decrement per time step
        weight_matrix = weight_matrix - decay_choose
        weight_matrix[weight_matrix < 0] = 0
    return weight_matrix, decay_vector


def binomial_detachment(weight_matrix, decay_vector, n_passes, p, spread_rate):
    # Degeneration model has base prob that an edge is deleted, p

    weights_mask = weight_matrix > 0

    sh = weight_matrix.shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            if i == j:
                weights_mask[i, j] = 1
    n_nodes = sh[0]
    osum = np.sum(weight_matrix)
    decay_matrix = weights_mask
    for i in range(n_passes):

        decay_matrix = weights_mask * decay_vector
        if np.sum(decay_matrix) == 0:
            decay_chance = weights_mask * p / n_nodes
        else:
            decay_chance = weights_mask * p / n_nodes + (1 - p) * decay_matrix / osum
        #         print(weights_mask * p / n_nodes, (1-p) * spread_rate * decay_matrix / np.sum(decay_matrix))
        #         if i == 0:
        #             print(np.sum(decay_chance > p / n_nodes), decay_chance.shape[0]*decay_chance.shape[1], np.mean((1-p) * decay_matrix / osum))
        decay_chance[decay_chance > 1] = 1
        binomial_delete = np.random.binomial(weight_matrix, decay_chance)

        weight_matrix = weight_matrix - binomial_delete
        #         print(decay_vector.shape, binomial_delete.shape)
        decay_vector = (decay_vector + np.sum(binomial_delete, axis=1)) * spread_rate

    return weight_matrix, decay_vector


def random_detachment(weight_matrix, n_passes, p):
    # Every synapse has an equal chance to be deleted, p
    sh = weight_matrix.shape
    weights_mask = weight_matrix > 0
    #     print(sh, weight_matrix)
    for i in range(n_passes):
        #         print(weight_matrix)
        delete = np.random.binomial(weight_matrix, p)

        weight_matrix = weight_matrix - delete

    return weight_matrix


def perform_ablation(graph, func, n_passes, weight_key, decay_vector):

    A = nx.adjacency_matrix(graph, weight=weight_key).toarray()

    if func == random_detachment:
        return func(A, n_passes, 0.02)
    else:
        return func(A, decay_vector, n_passes, 0.05, 1)
