"""
Benchmark-Prion Neurodegeneration Study:
    This module serves to provide an example of how to use the functions in graphfunctions.py.
    This module creates a dataframe from the Neuprint Janelia website based on neuron connections
    in the Mushroom Body to create graph that shows the neurons' feedforward paths.

    @Author: Anirejuoritse Egbe lax18christian@gmail.com
    @Date: 7/21/2022
    @Credit: REDD Department of Johns Hopkins Applied Physics Laboratory
"""

import neuprint
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from graphfunctions import *

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" \
        ".eyJlbWFpbCI6ImxheDE4Y2hyaXN0aWFuQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUFUWEFKd002NG9MQmtBYnc3T1JES2JQbmRHWGRNdDgxMEE1ZUxUTGphUno9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MzY1NzYzMDJ9.mvOOHTXT5emyB2yzh47uNmwYZL1SQOdoWgkFHuajVuU "

client = neuprint.Client('https://neuprint.janelia.org', token=token, dataset='hemibrain:v1.2.1')

w_thresh = 5
is_thresh = f"""AND w.weightHP >= {w_thresh}"""
types = ["PN", "KC", "APL", "MBON", "ORN"]  # neuron types in the MB
wheres_thresh = []

table = pd.DataFrame()
for c1 in types:
    for c2 in types:
        where = f"""(a.type CONTAINS "{c1}") AND (b.type CONTAINS "{c2}") {is_thresh}"""
        q = f" MATCH (a :`hemibrain_Neuron`)-[w:ConnectsTo]->(b:`hemibrain_Neuron`) WHERE {where} RETURN a.bodyId, " \
            f"a.type, b.bodyId, b.type, w.weight "

        lh_table = neuprint.fetch_custom(q)
        lh_table["Supertype_pre"] = c1
        lh_table["Supertype_post"] = c2
        table = pd.concat([table, lh_table])
table = table.rename(columns={"w.weight": "weight"})
table.columns = table.columns.str.replace(' ', '')

# create a graph where the source nodes are subtypes, target nodes are their supertypes in the post-synapse,
# and labels & weight are node attributes and color is edge attribute
G = nx.Graph()
G = nx.from_pandas_edgelist(table, 'a.type', 'b.type')
labels_dictionary = table.set_index('a.type').to_dict()['Supertype_pre']  # matches subtype neurons to its supertypes
weights_dictionary = table.set_index('a.type').to_dict()['weight']
nx.set_node_attributes(G, name="labels", values=labels_dictionary)

# make color an edge attribute to color edges in graph
nx.set_edge_attributes(G, name='color', values=1)
edgeColor(G)

# remove all the extra edges that are colored black and track nodes that don't feedforward properly
G, connections = removeExcess(G)
colors = [G[u][v]['color'] for u, v in G.edges()]

# add node attributes to graph
nodeStructure(G)

# assign a node shape depending on if node weight falls within a certain range
G, node_shapes = nodeShapes(G)
shapes = ['o', '8', 'D', 'p', 's']

# creating a figure using subplot to add legend
fit = 300
fig = plt.figure(figsize=(fit, fit))
ax = fig.add_subplot(1, 1, 1)

# position nodes on figure
nodeCoordinates(G)

# add edges to figure
edgeOpacity(G)
edges_opacity = nx.get_edge_attributes(G, 'opacity')
edge_widths = list(edges_opacity.values())
new_edge_widths = []
for i in edge_widths:
    new_edge_widths.append(i*6)
for e in G.edges:
    nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'node_positions'), width=new_edge_widths, alpha=edges_opacity[e], edge_color=colors)

# position node labels to not cover node
offset, node_labels = nodelabelcoordinates(G, types)
nx.draw_networkx_labels(G, pos=offset, labels=node_labels, horizontalalignment='right',
                        font_size=fit, font_weight='bold')

# position nodes on figure. Need for loop because node_shape in nx.draw_networkx_nodes uses one shape at a time
for shape in set(shapes):
    node_list = [node for node in G.nodes() if G.nodes()[node]['shape'] == shape]
    nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G, 'node_positions'), nodelist=node_list,
                           node_size=[(G.nodes()[node]['size'])*fit*20 for node in node_list], node_color=[G.nodes()[node]['color'] for node in node_list],
                           node_shape=shape)

# add text to figure
ax.set_title('Feedforward Network of Neurons in Mushroom Body', fontsize=fit, fontweight='bold')
connections = removednodelist(connections)
txt = "Fig 1: The figure above shows a networkx graph showing the feedforward neuron connections in the Mushroom Body " \
      "of the Drosophila fly. The neuron types (APL=red, KC=blue, MBON=orange, PN=green, RN=purple) are shown on the " \
      "graph and edges are colored by connections to specific neuron type. The pink represents the recourrent connection " \
      "between KC nodes. The nodes that failed to feedforward to their proper place are: " \

final_txt = txt + str(connections)
ax.set_xlabel(final_txt, size=fit, wrap=True)

# create legend for node shapes
legend_elements = [Line2D([0], [0], marker='o', markersize=fit * 0.60, label=str(node_shapes[0])),
                   Line2D([0], [0], marker='D', markersize=fit * 0.60, label=str(node_shapes[1])),
                   Line2D([0], [0], marker='8', markersize=fit * 0.60, label=str(node_shapes[2])),
                   Line2D([0], [0], marker='s', markersize=fit * 0.60, label=str(node_shapes[3])),
                   Line2D([0], [0], marker='p', markersize=fit * 0.60, label=str(node_shapes[4]))]
legend = ax.legend(handles=legend_elements, loc='upper center', title='Node Weight Scale', fontsize=fit * 0.60)
legend_title_size = fit * 0.60
legend_title = str(legend_title_size)
legend.get_title().set_fontsize(legend_title)
fig.savefig('NeurodegenerationTask2_Final_New.png')
