from groovi.seq.constants import *
from groovi.graph.graphs import GrooviGraph

import dgl
from enum import Enum
from matplotlib.colors import LinearSegmentedColormap
from pyvis.network import Network
import random

ARROWSIZE=.10
HORIZONTAL_SPACING = 100
EDGE_WIDTH = 3

class EDGE_LENGTHS(Enum):
    SHORT = 10
    MEDIUM = 30
    LONG = 40

def visualize_groovi_graph(g: GrooviGraph, path:str, additional_info: dict = None, node_feature_color_code = NodeFeature.RD, stagger_edges = False) -> None:
    '''
    Draws a genome graph, made to look linear. Turn stagger_edges on to add randomness for debugging purposes.
    Inputs:
        g: GrooviGraph to visualize
        path: path to save the html file
        additional_info: dictionary containing additional information to display on hover 
        - additional_info['edge_labels_str']: list of strings, one for each edge, to display as edge labels
        node_feature_color_code: NodeFeature enum, which node feature to use for coloring nodes
        stagger_edges: bool, whether to add randomness to edge lengths and curvatures, for debugging usage
    '''
    def get_stagger():
        if not stagger_edges: return 0
        return random.random()
    
    straight_edges = set([EDGE_RELATION_MAP[relation] for relation in EdgeRelation if relation in [EdgeRelation.REF, EdgeRelation.SEG]])
    adj_edges = set([EDGE_RELATION_MAP[relation] for relation in EdgeRelation if relation in [EdgeRelation.ORD12, EdgeRelation.ORD21, EdgeRelation.STRAND_FLIP]])

    net = Network(notebook=True, directed=True, cdn_resources='remote')
    g = g.get_dgl_graph()
    n = g.num_nodes()
    # Feature normalization -> color map
    cmap = LinearSegmentedColormap.from_list("light_blue_white_light_red", ["#53ADCA", "#FAFAFA", "#EC5B5B"])
    index_color = NODE_FEATURE_MAP[node_feature_color_code]
    min_val = g.ndata['feat'][:, index_color].min().item()
    max_val = g.ndata['feat'][:, index_color].max().item()
    normalized_features = (g.ndata['feat'][:,index_color] - min_val) / (max_val - min_val)
    normalized_rd_colors = cmap(normalized_features.numpy())[:, :3]  # R,G,B exclude opacity

    # Add nodes to the PyVis network
    for node_id in range(n):
        # If node has any features, you can add them here as attributes (optional)
        node_features = g.ndata['feat'][node_id]
        title = f'Features: {node_features}'
        if 'position' in g.ndata:
            title += f'\n location: {g.ndata["position"][node_id]}'
        if 'chr_id' in g.ndata:
            title += f'\n chr_id: {g.ndata["chr_id"][node_id]}'

        r_c,g_c,b_c = normalized_rd_colors[node_id]*255
        r_c,g_c,b_c = int(r_c), int(g_c), int(b_c)
        color = f'rgb({r_c},{g_c},{b_c})'
        net.add_node(node_id, label=str(node_id), title=title, color=color, x=node_id * HORIZONTAL_SPACING, y=0, fixed={'x': True, 'y': False}, physics=True)

    # Add edges to the PyVis network
    u, v, m = g.edges(form='all')

    all_eids = set()
    all_nids = set()
    for u, v, edge_id in zip(u, v, m):
        edge_type = g.edata['rel_type'][edge_id].item()
        title = f"Edge Type: {g.edata['rel_type'][edge_id]} \n Data: {g.edata['feat'][edge_id]} \n Edge ID: {edge_id.item()}"
        if edge_type < 3: 
            all_eids.add(edge_id.item())
            all_nids.add(u.item())
            all_nids.add(v.item())
        if additional_info and 'edge_label' in additional_info and edge_id < len(additional_info['edge_label']):
            edge_label = additional_info['edge_label'][edge_id]
            title += f'\n Edge Labeledge_label: {edge_label}'

        if 'labels' in g.edata:
            edge_class = g.edata['labels'][edge_id]
            title += f'\n Edge label: {edge_class}'
        
        if 'distance' in g.edata:
            title += f'\n Distance: {g.edata["distance"][edge_id].item()}'

        color = EDGE_COLOR_MAP[g.edata['rel_type'][edge_id].item()]

        if edge_type in straight_edges: # reference and segment edges are straight
            if stagger_edges:
                smooth = {'type': 'dynamic'}
            else:
                smooth = {'type': 'straightCross'}
            length = EDGE_LENGTHS.SHORT.value + get_stagger()
        elif edge_type in adj_edges:
            smooth = {'type': 'curvedCW', 'roundness': 0.5 + get_stagger()*0.5}
            length = EDGE_LENGTHS.MEDIUM.value + get_stagger()
        else: #pd
            smooth = {'type': 'curvedCW', 'roundness': 0.8 + get_stagger()*0.2}
            length = EDGE_LENGTHS.LONG.value + get_stagger()
        net.add_edge(u.item(), v.item(), color=color, title=title, length=length, smooth=smooth, width=EDGE_WIDTH, arrowsize=ARROWSIZE)

    net.toggle_physics(True)
    net.set_edge_smooth('dynamic')
    net.show(path)
