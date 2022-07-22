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


def test_network_degeneration(G, func, n_passes, weight_key, sum_min, b_or_c="B", decomp=False, dv=np.zeros(1),
                              spread_rate=1):
    print("start", weight_key)
    G_decomp = G
    nodelist = G_decomp.nodes()
    decomp_df = pd.DataFrame()
    names = list(G_decomp.nodes)
    i = 0
    m = nx.adjacency_matrix(G_decomp).toarray()
    sum_num = np.sum(m)
    edges = np.sum(m > 0)
    n_edges = edges
    o_sum_num = sum_num

    while edges / n_edges > sum_min:
        if b_or_c == "B":
            G_b = nx.betweenness_centrality(G_decomp, weight=weight_key)
            data_df = pd.DataFrame(G_b, index=["Betweenness"]).T
        elif b_or_c == "s":
            G_b = nx.current_flow_betweenness_centrality(G_decomp.to_undirected(), weight=weight_key)
            data_df = pd.DataFrame(G_b, index=["Current"]).T
        elif b_or_c == "d":
            G_bi = nx.in_degree_centrality(G_decomp)
            G_bo = nx.out_degree_centrality(G_decomp)
            data_df = pd.DataFrame(G_bi, index=["In-Degree Cenrality"]).T
            data_df["Out-Degree Cenrality"] = G_bo
        else:
            G_b = nx.eigenvector_centrality(G_decomp, weight=weight_key)
            data_df = pd.DataFrame(G_b, index=["Eigen Cenrality"]).T
        data_df["Degree"] = data_df.index.map(G_decomp.degree)
        neighbor_deg = nx.average_neighbor_degree(G_decomp, weight=weight_key, source='in', target='out')
        data_df["Neighbor Degree"] = data_df.index.map(neighbor_deg)
        data_df["Density"] = nx.density(G_decomp)
        data_df["Transitivity"] = nx.transitivity(G_decomp)
        data_df["Reciprocity"] = nx.reciprocity(G_decomp, nodes=nodelist)
        data_df["Iter"] = i

        A = nx.adjacency_matrix(G_decomp, weight=weight_key).toarray()
        sum_num = np.sum(A)
        edges = np.sum(A > 0)
        data_df["Sum"] = sum_num
        data_df["Sum Fraction"] = sum_num / o_sum_num

        if not decomp:
            print(sum_num, np.sum(A > 0), sum_num / o_sum_num, np.max(A), np.min(A))

            A = func(A, n_passes)

        else:
            print(sum_num, np.sum(A > 0), sum_num / o_sum_num, np.max(A), np.min(A), np.sum(dv), np.sum(dv > 0))

            A, dv = func(A, dv, n_passes, .05, spread_rate)
        ad = pd.DataFrame(A, columns=nodelist, index=nodelist)

        G_decomp = nx.from_pandas_adjacency(ad, create_using=nx.DiGraph())

        decomp_df = pd.concat([decomp_df, data_df])
        i += 1
    return decomp_df


def preferential_detachment(weight_matrix, decay_vector, n_passes, p=.05, spread_rate=1):
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


def binomial_detachment(weight_matrix, decay_vector, n_pass, p=.1, spread_rate=1):
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
    for i in range(n_pass):

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


def random_detachment(weight_matrix, n_passes, p=.02):
    # Every synapse has an equal chance to be deleted, p
    sh = weight_matrix.shape
    weights_mask = weight_matrix > 0
    #     print(sh, weight_matrix)
    for i in range(n_passes):
        #         print(weight_matrix)
        delete = np.random.binomial(weight_matrix, p)

        weight_matrix = weight_matrix - delete

    return weight_matrix

