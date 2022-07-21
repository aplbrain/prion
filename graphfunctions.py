"""
Graph functions:
    This module serves to define and describe the functions used in graphfunctions.py.
    The functions used in graphfunctions.py are designed to use the Neuprint Janelia website.

    @Author: Anirejuoritse Egbe lax18christian@gmail.com
    @Date: 7/21/2022
    @Credit: REDD Department of Johns Hopkins Applied Physics Laboratory
"""
# graphfunctions.py>
import networkx as nx
import numpy as np


def edgeColor(graph):
    """
    Adds color to edge.

    ORN edges must feedforward to PN nodes, PN edges must feedforward to KC nodes,
    APL edges must feedforward to PN nodes, KC edges must feedforward to APL and MBON,
    and show recurrent connection to other KC nodes.
    :param graph: graph of nodes from table dataframe
    :return: updated graph with color assigned to all edges
    """
    for node in graph.nodes():
        for u, v in graph.edges(node):
            if "PN" in u:
                if 'KC' in v:
                    graph[u][v]['color'] = 'blue'
                elif 'ORN' in v:
                    graph[u][v]['color'] = 'green'
                else:
                    graph[u][v]['color'] = 'black'
            elif "KC" in u:
                if 'APL' in v:
                    graph[u][v]['color'] = 'red'
                elif 'PN' in v:
                    graph[u][v]['color'] = 'blue'
                elif 'MBON' in v:
                    graph[u][v]['color'] = 'orange'
                elif 'KC' in v:
                    graph[u][v]['color'] = 'pink'
                else:
                    graph[u][v]['color'] = 'black'
            elif "MBON" in u:
                if 'KC' in v:
                    graph[u][v]['color'] = 'orange'
                else:
                    graph[u][v]['color'] = 'black'
            elif "APL" in u:
                if 'KC' in v:
                    graph[u][v]['color'] = 'red'
                else:
                    graph[u][v]['color'] = 'black'
            elif "ORN" in u:
                if 'PN' in v:
                    graph[u][v]['color'] = 'green'
                else:
                    graph[u][v]['color'] = 'black'


def removeExcess(graph):
    """
    Removes and keeps track of nodes and edges that don't feedforward in the corrct path

    :param graph: graph of nodes from table dataframe
    :return graph: updated graph with new nodes and edges that feedforward in the right path
    """
    connections = {}
    # connections = [v for u, v, z in graph.edges.data('color') if z == 'black']
    for u, v, z in graph.edges.data('color'):
        if z == 'black':
            connections.setdefault(u, []).append(v)
    graph.remove_edges_from([(u, v) for u, v, z in graph.edges.data('color') if z == 'black'])
    graph.remove_nodes_from([node for node in graph.nodes() if graph.nodes()[node] == {}])
    graph.remove_nodes_from([node for node in graph.nodes() if graph.degree(node) == 0])
    return graph, connections


def nodeStructure(graph):
    """
    Adds color, labels , and size as node attributes based on supertype.

    :param graph: graph of nodes from table dataframe
    :return graph: updated graph with new node attributes
    """
    for node in graph.nodes():
        if graph.nodes()[node]['labels'] == 'APL':
            graph.add_node(node, color="red")
            graph.add_node(node, labels2='APL')
        elif graph.nodes()[node]['labels'] == 'KC':
            graph.add_node(node, color="blue")
            graph.add_node(node, labels2='KC')
        elif graph.nodes()[node]['labels'] == 'MBON':
            graph.add_node(node, color="orange")
            graph.add_node(node, labels2='MBON')
        elif graph.nodes()[node]['labels'] == 'PN':
            graph.add_node(node, color="green")
            graph.add_node(node, labels2='PN')
        elif graph.nodes()[node]['labels'] == 'ORN':
            graph.add_node(node, color="purple")
            graph.add_node(node, labels2='ORN')
    nx.set_node_attributes(graph, name='size', values=dict(graph.degree(weight='weight')))


def nodeShapes(graph):
    """Add node shape as node attribute in graph.

    Node size is based on the degree of node, which is the number of edges a node has.
    After finding the maximum node size, an array was created from value 10 to max node
    size,evenly spaced by the type of neurons in our dataframe. Each value in array is
    associated with a shape and for each node, a shape was given based on its weight
    being below a value in the array.
    :param graph: graph of nodes from table dataframe
    :return graph: graph with shape depicting node weight as an attribute
    """

    maxweight = 0
    node_weight = dict(graph.degree(weight='weight'))
    for k, v in node_weight.items():
        if v > maxweight:
            maxweight = v
    node_shapes = np.linspace(10, maxweight, 5)
    for node in graph.nodes():
        for i in node_shapes:
            if graph.nodes()[node]['size'] <= i:
                if i == node_shapes[0]:
                    graph.add_node(node, shape="o")
                    break
                elif i == node_shapes[1]:
                    graph.add_node(node, shape="D")
                    break
                elif i == node_shapes[2]:
                    graph.add_node(node, shape="8")
                    break
                elif i == node_shapes[3]:
                    graph.add_node(node, shape="s")
                    break
                elif i == node_shapes[4]:
                    graph.add_node(node, shape="p")
                    break

    return graph, node_shapes


def typeCount(graph):
    """Count the number of nodes in each supertype."""

    pn_node_size = 0
    apl_node_size = 0
    mbon_node_size = 0
    kc_node_size = 0
    orn_node_size = 0
    for node in graph.nodes():
        if graph.nodes()[node]['labels'] == 'PN':
            pn_node_size = pn_node_size + 1
        elif graph.nodes()[node]['labels'] == 'APL':
            apl_node_size = apl_node_size + 1
        elif graph.nodes()[node]['labels'] == 'MBON':
            mbon_node_size = mbon_node_size + 1
        elif graph.nodes()[node]['labels'] == 'KC':
            kc_node_size = kc_node_size + 1
        elif graph.nodes()[node]['labels'] == 'ORN':
            orn_node_size = orn_node_size + 1
    return pn_node_size, apl_node_size, mbon_node_size, kc_node_size, orn_node_size


def nodeCoordinates(graph):
    """
    Create node coordinates in graph to model a neural network.

    ORN and MBON nodes are positioned in a single column, PN ndoes are positioned
    in three columns, APL is a point, and KC nodes are arranged in a circle.
    :param graph: graph of nodes from table data frame
    :return graph: updated graph with node positions as a node attribute
    """
    fit = 300

    pn_node_size, apl_node_size, mbon_node_size, kc_node_size, orn_node_size = typeCount(graph)

    pn_split = round(pn_node_size / 2)
    pn_split2 = pn_node_size - pn_split
    mbon_spacing = np.linspace(0, fit, mbon_node_size, endpoint=True)
    pn_spacing = np.linspace(0, fit, pn_split, endpoint=True)
    pn_spacing1 = np.linspace(0, fit, pn_split2, endpoint=True)
    final_pn_spacing = np.concatenate((pn_spacing, pn_spacing1), axis=0)
    apl_spacing = np.linspace(fit * 0.50, fit, apl_node_size, endpoint=True)
    orn_spacing = np.linspace(0, fit, orn_node_size, endpoint=True)
    index = 0
    index1 = 0
    index2 = 0
    index3 = 1
    index4 = 0
    theta = 2 * np.pi / kc_node_size  # used for creating shell layout of KC nodes

    # create a dictionary where key = nodes and value = node positions in graph
    for node in graph.nodes():
        # must create three columns for PN nodes because of amount compared to other supertypes
        if graph.nodes()[node]['labels'] == 'PN':
            if index < pn_split:
                a = 0
            else:
                a = 20
            graph.add_node(node, node_positions=(fit * 0.25 + a, final_pn_spacing[index]))
            index += 1
        elif graph.nodes()[node]['labels'] == 'ORN':
            graph.add_node(node, node_positions=(fit * 0.10, orn_spacing[index4]))
            index4 += 1
        elif graph.nodes()[node]['labels'] == 'APL':
            graph.add_node(node, node_positions=(apl_spacing[index1], fit * 0.75))
            index1 += 1
        elif graph.nodes()[node]['labels'] == 'MBON':
            graph.add_node(node, node_positions=(fit * 0.75, mbon_spacing[index2]))
            index2 += 1
        elif graph.nodes()[node]['labels'] == 'KC':
            xpos = fit / 10 * np.cos(theta * index3)
            ypos = fit / 10 * np.sin(theta * index3)
            graph.add_node(node, node_positions=(fit * 0.50 + xpos, fit * 0.50 + ypos))
            index3 += 1
    return graph


def nodelabelcoordinates(graph, types):
    """Shift Node labels to stop covering up nodes in figure.

    :param graph: graph with nodes from table dataframe
    :return pos_right, pos_right_labels: dictionary for positions of node labels, dictionary for labels of node
    """

    pos_right = {}
    right_off = -5
    pos_right_labels = {}
    for i in types:
        for k, v in graph.nodes(data="node_positions"):
            if i in k:
                pos_right[k] = (v[0] + right_off, v[1])
                pos_right_labels[k] = graph.nodes()[k]['labels2']
                break
            elif i in k:
                pos_right[k] = (v[0] + right_off, v[1])
                pos_right_labels[k] = graph.nodes()[k]['labels2']
                break
            elif i in k:
                pos_right[k] = (v[0] + right_off, v[1])
                pos_right_labels[k] = graph.nodes()[k]['labels2']
                break
            elif i in k:
                pos_right[k] = (v[0] + right_off, v[1])
                pos_right_labels[k] = graph.nodes()[k]['labels2']
                break
            elif i in k:
                pos_right[k] = (v[0] + right_off, v[1])
                pos_right_labels[k] = graph.nodes()[k]['labels2']
                break
        if i == 4:
            break
    return pos_right, pos_right_labels


def edgeOpacity(graph):
    """
    Change opacity of edges in graph.

    All the nodes are separated by the supertype and each node weight is divided by the
    supertype total weight. The nodes in each supertype are arranged in ascending order.
    An array is created for each supertype and the array is evenly spaced by number
    of nodes per supertype from 0 to 1. This was done because the max value for opacity
    in nx.draw_edges is 1. Each node is assigned a value for its supertype array, with
    the value representing opacity compared to other nodes in supertype. Finally, the
    edges in each node are given that opacity value.
    :param graph: graph with nodes from table dataframe
    :return: graph whose edges are normalized by node weight
    """
    kc_total_weight = 0
    pn_total_weight = 0
    orn_total_weight = 0
    mbon_total_weight = 0
    apl_total_weight = 0
    node_weight = dict(graph.degree(weight='weight'))

    # find the total weight for each type
    for k, v in node_weight.items():
        if "KC" in k:
            kc_total_weight += v
        elif "PN" in k:
            pn_total_weight += v
        elif "ORN" in k:
            orn_total_weight += v
        elif "MBON" in k:
            mbon_total_weight += v
        elif "APL" in k:
            apl_total_weight += v

    kc_alphas = {}
    pn_alphas = {}
    apl_alphas = {}
    mbon_alphas = {}
    orn_alphas = {}

    # find the percentage of weight for each node by its supertype total weight
    for k, v in node_weight.items():
        if "KC" in k:
            kc_alphas[k] = v / kc_total_weight
        elif "PN" in k:
            pn_alphas[k] = v / pn_total_weight
        elif "APL" in k:
            apl_alphas[k] = v / apl_total_weight
        elif "MBON" in k:
            mbon_alphas[k] = v / mbon_total_weight
        elif "ORN" in k:
            orn_alphas[k] = v / orn_total_weight

    # arrange dictionary in ascending order to organize nodes from the least weight to most
    kc_alphas_arranged = dict(sorted(kc_alphas.items(), key=lambda x: x[1]))
    pn_alphas_arranged = dict(sorted(pn_alphas.items(), key=lambda x: x[1]))
    apl_alphas_arranged = dict(sorted(apl_alphas.items(), key=lambda x: x[1]))
    mbon_alphas_arranged = dict(sorted(mbon_alphas.items(), key=lambda x: x[1]))
    orn_alphas_arranged = dict(sorted(orn_alphas.items(), key=lambda x: x[1]))
    pn_node_size, apl_node_size, mbon_node_size, kc_node_size, orn_node_size = typeCount(graph)

    # normalize the nodes to value of one for each type
    orn_opacity = np.linspace(0, 1, orn_node_size, endpoint=True)
    apl_opacity = np.linspace(0, 1, apl_node_size, endpoint=True)
    mbon_opacity = np.linspace(0, 1, mbon_node_size, endpoint=True)
    pn_opacity = np.linspace(0, 1, pn_node_size, endpoint=True)
    kc_opacity = np.linspace(0, 1, kc_node_size, endpoint=True)
    kc_opacity_dict = {}
    kc_index = 0
    apl_opacity_dict = {}
    apl_index = 0
    mbon_opacity_dict = {}
    mbon_index = 0
    pn_opacity_dict = {}
    pn_index = 0
    orn_opacity_dict = {}
    orn_index = 0
    # create dictionary of all nodes in each neuron type that places nodes in order of increasing weight
    for k, v in kc_alphas_arranged.items():
        kc_opacity_dict[k] = kc_opacity[kc_index]
        kc_index += 1
    for k, v in pn_alphas_arranged.items():
        pn_opacity_dict[k] = pn_opacity[pn_index]
        pn_index += 1
    for k, v in apl_alphas_arranged.items():
        apl_opacity_dict[k] = apl_opacity[apl_index]
        apl_index += 1
    for k, v in mbon_alphas_arranged.items():
        mbon_opacity_dict[k] = mbon_opacity[mbon_index]
        mbon_index += 1
    for k, v in orn_alphas_arranged.items():
        orn_opacity_dict[k] = orn_opacity[kc_index]
        orn_index += 1

    # add opacity as edge attribute depending on node
    for k, v in kc_opacity_dict.items():
        for e in graph.edges:
            if k == e[0]:
                graph.add_edge(e[0], e[1], opacity=v)
    for k, v in pn_opacity_dict.items():
        for e in graph.edges:
            if k == e[0]:
                graph.add_edge(e[0], e[1], opacity=v)
    for k, v in orn_opacity_dict.items():
        for e in graph.edges:
            if k == e[0]:
                graph.add_edge(e[0], e[1], opacity=v)
    for k, v in apl_opacity_dict.items():
        for e in graph.edges:
            if k == e[0]:
                graph.add_edge(e[0], e[1], opacity=v)
    for k, v in mbon_opacity_dict.items():
        for e in graph.edges:
            if k == e[0]:
                graph.add_edge(e[0], e[1], opacity=v)
    return graph


def removednodelist(connections):
    """Create a dictionary of all the nodes removed from graph

    :param connections: dictionary of nodes removed from graph along with their edges
    :return connections_dict: dictionary of nodes that don't feedforward correctly
    """

    connections_dict = {}
    for k, v in connections.items():
        connections_dict[k] = v

    connections_dict = {k: [vi for vi in v if k == 'PN' and 'KC' in vi] for k, v in connections_dict.items()}
    connections_dict = {k: [vi for vi in v if k == 'APL' and 'KC' in vi] for k, v in connections_dict.items()}
    connections_dict = {k: [vi for vi in v if k == 'KC' and 'MBON' in vi] for k, v in connections_dict.items()}
    connections_dict = {k: [vi for vi in v if k == 'PN' and 'KC' in vi] for k, v in connections_dict.items()}

    empty = []
    for k, v in connections_dict.copy().items():
        if v == empty:
            connections_dict.pop(k)
    return connections_dict
