from groovi.graph.graphs import GrooviGraph
from groovi.graph.visualize import visualize_groovi_graph
from groovi.seq.constants import *

import pickle
import os

class GraphHandler:
    def __init__(self, config):
        self.config = config
        self.graphs: list[GrooviGraph] = []
        self.summary: list[dict] = []
    
    def add_graphs(self, graphs: list[GrooviGraph]) -> None:
        assert isinstance(graphs, list), "Input must be a list of GrooviGraph instances."
        self.graphs.extend(graphs)

    def load(self, path: str, store=False) -> list[GrooviGraph]:
        '''
        Load graphs from a file. If save is True, the loaded graphs are added to the handler internally.
        '''
        loaded_graphs = self._load_graphs(path)
        if store:
            self.graphs.extend(loaded_graphs)
        return loaded_graphs

    def save(self) -> None:
        # Save Graphs Together
        save_graph_path = os.path.join(self.config.graph_dir, f"graphs_all.pkl")
        self._save_graphs(save_graph_path, self.graphs)

        # Save Graphs, separated by connected componnents
        for i, graph in enumerate(self.graphs):
            save_path = os.path.join(self.config.graph_dir, f"graph_{i+1}.pkl")
            self._save_graphs(save_path, [graph])
        
        if self.config.visualize_every:
            for i, graph in enumerate(self.graphs):
                vis_path = os.path.join(self.config.img_dir, f"graph_{i+1}.html")
                visualize_groovi_graph(graph, vis_path)

        self._write_summary()
    
    def save_single(self, graph: GrooviGraph, filename: str) -> None:
        save_path = os.path.join(self.config.graph_dir, filename)
        self._save_graphs(save_path, [graph])

    def _save_graphs(self, save_path: str, graphs: list[GrooviGraph]):
        with open(save_path, 'wb') as f:
            pickle.dump(graphs, f)

    def _load_graphs(self, save_path: str) -> list[GrooviGraph]:
        with open(save_path, 'rb') as f:
            graphs = pickle.load(f)
            return graphs
        
    def _write_summary(self):
        self._update_summary(self.graphs)

        summary_lines = []
        for i, entry in enumerate(self.summary):
            summary_lines.append(f"--- Summary Entry {i+1} ---")
            for key, value in entry.items():
                summary_lines.append(f"{key}: {value}")

        summary = "\n".join(summary_lines)
        with open(self.config.stats_path, 'w') as f:
            f.write(summary)

    def _update_summary(self, graphs: list[GrooviGraph]) -> None:
        for graph in graphs:
            summary_entry = {
                'num_nodes': graph.num_nodes(),
                'num_edges': graph.num_edges(),
                'chr_ids': graph.get_chr_ids_included(as_string=True),
                'vcf': graph.vcf_path,
                'candidate_src': graph.get_candidate_edge_ids()[0],
                'candidate_dst': graph.get_candidate_edge_ids()[1],
                # ... add more statistics if needed
            }
            self.summary.append(summary_entry)




