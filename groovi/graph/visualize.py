from groovi.graph.constants import *
from groovi.graph.graphs import GrooviGraph, GrooviNode
from groovi.seq.constants import *

from enum import Enum
from pyvis.network import Network
import random

NODE_SIZE = 200
ARROWSIZE= NODE_SIZE // 2500
HORIZONTAL_SPACING = NODE_SIZE*2

EDGE_WIDTH = NODE_SIZE//50

class EDGE_LENGTHS(Enum):
    SHORT = NODE_SIZE*4
    MEDIUM = NODE_SIZE*6
    LONG = NODE_SIZE*8

EDGE_COLOR_MAPCOLORS = {i: color for i, color in enumerate(['mediumpurple', 'pink', 'lightgreen', 'darkseagreen', 'lightblue', 'lightgray', 'yellow', 'pink', "blue", "red"])}

def visualize_groovi_graph(g: GrooviGraph, path: str, candidate_src= None, candidate_dst=None, expected_labels=None, predicted_labels=None):
    vis_format = path.split('.')[-1]
    if vis_format in ['html']:
        visualize_groovi_graph_pyvis(g, path, candidate_src, candidate_dst, expected_labels, predicted_labels)
    else:
        raise ValueError(f'{vis_format} not supported')
    
def visualize_groovi_graph_pyvis(g: GrooviGraph, path:str, candidate_src = None, candidate_dst = None, expected_labels = None, predicted_labels = None, stagger_edges = False) -> None:
    '''
    Draws a genome graph, made to look linear. Turn stagger_edges on to add randomness for debugging purposes.
    Inputs:
        g: GrooviGraph to visualize
        path: path to save, ensure it ends in html
    '''
    # Randomness option, used for debugging overlapping edges
    def get_stagger():
        if not stagger_edges: return 0
        return random.random() * 0.1
    
    # init
    net = Network(notebook=True, directed=True, cdn_resources='remote')
    straight_edges = set([EDGE_FEATURE_TO_ID[relation] for relation in EdgeFeature if relation in [EdgeFeature.REF, EdgeFeature.SEG]])
    adj_edges = set([EDGE_FEATURE_TO_ID[relation] for relation in EdgeFeature if relation in [EdgeFeature.LL, EdgeFeature.LR, EdgeFeature.RL, EdgeFeature.RR]])
    # Nodes & Annotations
    n = g.dgl_graph.num_nodes()
    for node_id in range(n):
        display_label = []
        ndata = g.dgl_graph.ndata
        n_features = ndata[FEAT][node_id]
        ht = ndata["HT"][node_id]

        for node_feature_name, node_feature_idx in NODE_FEATURE_TO_ID.items():
            feature_value = n_features[node_feature_idx].item()
            display_label.append(f'{node_feature_name.value}: {feature_value:.2f}')
        
        display_label.append(f'HT: {ht}')
        
        node: GrooviNode = g.id_to_node[node_id]
        position = node.ref_pos

        position_str = f'Position: ({position})'
        node_id_str = f'Node ID: ({node_id})'
        chr_name_str = f'Chr: {node.chr_name}'

        display_label.extend([position_str, node_id_str, chr_name_str])
        # apply padding
        display_label = [
            '   ' + line + '   ' for line in display_label
        ]
        display_label = '\n' + '\n'.join(display_label) + '\n'

        net.add_node(node_id, label=display_label, x=node_id * HORIZONTAL_SPACING, y=0, fixed={'x': True, 'y': True}, physics=True, size=NODE_SIZE, shape='circle')

    # Edges & Annotations
    u, v, m = g.dgl_graph.edges(form='all')
    edge_feature_list = list(EdgeFeature)
    for u, v, edge_id in zip(u, v, m):
        edata = g.dgl_graph.edata
        edge_type = edata[FEAT_TYPE][edge_id].item()
        edge_feature = edata[FEAT][edge_id].item()
        src_node: GrooviNode = g.id_to_node[u.item()]
        dst_node: GrooviNode = g.id_to_node[v.item()]
        distance = abs(dst_node.ref_pos-src_node.ref_pos)

        edge_type_str = f'Type: {edge_feature_list[edge_type].value}'
        edge_feature_str = f'Evidence: {edge_feature:.2f}'
        distance_str = f'Distance: {distance}'

        display_label = "\n".join([edge_type_str, edge_feature_str, distance_str])

        color = EDGE_COLOR_MAPCOLORS[edge_type]

        if edge_type in straight_edges: # reference and segment edges are straight
            if stagger_edges:
                smooth = {'type': 'curvedCW', 'roundness': 0.1 + get_stagger()}
                length = EDGE_LENGTHS.SHORT.value + get_stagger()
            else:
                smooth = {'type': 'straightCross'}
                length = EDGE_LENGTHS.SHORT.value + get_stagger()
        elif edge_type in adj_edges:
            smooth = {'type': 'curvedCW', 'roundness': 0.5 + get_stagger()}
            length = EDGE_LENGTHS.MEDIUM.value
        else: #pd
            smooth = {'type': 'curvedCW', 'roundness': 0.6 + get_stagger()}
            length = EDGE_LENGTHS.LONG.value + get_stagger()
        net.add_edge(u.item(), v.item(), color=color, label=display_label, title=display_label, length=length, smooth=smooth, width=EDGE_WIDTH, arrowsize=ARROWSIZE)

    if candidate_src is not None:
        for i, (u, v) in enumerate(zip(candidate_src, candidate_dst)):
            display_label = []
            if expected_labels is not None: display_label.append(f'Expected: {expected_labels[i]}')
            if predicted_labels is not None: display_label.append(f'Predicted: {predicted_labels[i]}')
            display_label = '\n'.join(display_label)
            color = 'orange'
            length = EDGE_LENGTHS.MEDIUM.value + get_stagger()
            net.add_edge(u.item(), v.item(), color=color, label=display_label, length=EDGE_LENGTHS.MEDIUM.value, smooth={'type': 'curvedCW', 'roundness': 0.7 + get_stagger()*0.2}, width=EDGE_WIDTH, arrowsize=ARROWSIZE)
        
    net.toggle_physics(True)
    net.set_edge_smooth('dynamic')

    # search feature
    search_injection = """
        <div style="position: absolute; top: 15px; left: 10px; z-index: 9;">
            <input type="text" id="search-bar" placeholder="Search by node id or position (press enter)"
                style="padding: 6px; border-radius: 4px; border: 1px solid black; width: 300px;">
            <div id="search-info" style="margin-top: 4px; background: white;"></div>
        </div>
        <script>
            let search_idx = 0;
            const search_bar_obj = document.getElementById('search-bar');
            const search_bar_info = document.getElementById('search-info');

            search_bar_obj.addEventListener('input', event => {
                search_idx = 0;
            })

            search_bar_obj.addEventListener('keydown', event => {
                if (event.key === 'Enter') {
                    query = event.target.value;
                    const matches = network.body.data.nodes.get().filter(node => node.label.indexOf(query) !== -1)

                    if (matches.length > 0) {
                        let node = matches[search_idx % matches.length];
                        network.selectNodes([node.id]);
                        network.focus(node.id, { scale: .5, animation: false });
                        search_bar_info.innerHTML = `Showing result ${(search_idx % matches.length) + 1} of ${matches.length} for "${query}"`;
                        search_idx += 1;
                    } else {
                        search_bar_info.innerHTML = `No results found for "${query}"`;
                    }
                }
            })
        </script>
    """

    # Generate HTML, save to file
    html_content = net.generate_html()
    final_html = html_content.replace('</body>', search_injection + '</body>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    