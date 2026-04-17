from groovi.graph.constants import *
from groovi.graph.graphs import GrooviGraph

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

class SubgraphProfile:
    def __init__(self, g: GrooviGraph, node_ids: list[int]):
        self.g = g
        self.node_ids = node_ids
    
    def get_profile(self) -> str:
        node_profile_strs = []
        for nid in self.node_ids:
            node_profile = ["_"] * len(NodeFeature)
            for node_feature in NodeFeature:
                d_id: int = NODE_FEATURE_TO_ID[node_feature]
                if self.g.dgl_graph.ndata[FEAT][nid][d_id] > 0:
                    node_profile[d_id] = node_feature.value
            node_profile_str = f'[{" ".join(node_profile)}]'
            node_profile_strs.append(node_profile_str)
        edge_profile_str = f"{' '.join(node_profile_strs)}"
        return edge_profile_str
    
class DatasetStats:
    def __init__(self, config):
        self.config = config

        self.style_overrides = {
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 14
        }
    
    def write_edge_profiles(self, graphs: list[GrooviGraph], all_sv_class_matches: list[list], tag: str):
        profiles = {}
        for g, class_matches in zip(graphs, all_sv_class_matches):
            assert len(class_matches) == len(g.candidate_src) == len(g.candidate_dst)
            for match, src_id, dst_id in zip(class_matches, g.candidate_src.tolist(), g.candidate_dst.tolist()):
                node_ids_of_interest = sorted(set([src_id, src_id + 1, dst_id - 1, dst_id]))
                profile = SubgraphProfile(g, node_ids_of_interest).get_profile()
                if match not in profiles: profiles[match] = {}
                if profile not in profiles[match]: profiles[match][profile] = 0
                profiles[match][profile] += 1

        with open(os.path.join(self.config.stats_dir, f"{tag}_profiles.txt"), "w") as f:
            summary = []
            for class_type in profiles:
                summary.append(f"{class_type}")
                for profile, count in sorted(profiles[class_type].items(), key=lambda x: x[1], reverse=True):
                    summary.append(f"\t {profile} : {count} ")
            f.write("\n".join(summary))

    def plot_node_feat_distribution(self, graphs: list[GrooviGraph], tag: str, log_y: bool = False):
        num_features = len(NodeFeature)

        all_feats = [g.dgl_graph.ndata[FEAT].cpu() for g in graphs if g.dgl_graph.num_nodes() > 0]
        all_feats = torch.concat(all_feats, dim=0).numpy()

        with plt.style.context(self.style_overrides):
            fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 5), squeeze=False)
            axes = axes.flatten()
            color_pallete = sns.color_palette("husl", num_features)

            for feat in NodeFeature:
                i = NODE_FEATURE_TO_ID[feat]
                feature_data = all_feats[:, i]
                
                _, bin_edges = np.histogram(feature_data, bins='auto')

                sns.histplot(
                    feature_data, 
                    bins=bin_edges,  
                    kde=True, 
                    ax=axes[i], 
                    color=color_pallete[i]
                )
                
                axes[i].set_title(f'{feat.value} Distribution')
                axes[i].set_xlabel('Feature Value',)
                
                if log_y:
                    axes[i].set_yscale('log')
                    axes[i].set_ylabel('Count (Log Scale)')
                else:
                    axes[i].set_ylabel('Count')
                
            fig.suptitle(f'Node Feature Distributions for {tag} Dataset {"logscale" if log_y else ""}')
            plt.tight_layout()
            
            suffix = "_logY" if log_y else ""
                
            plot_path = os.path.join(self.config.stats_dir, f"node_feat_dist{suffix}_{tag}.png")
            plt.savefig(plot_path)
            plt.close(fig)
    
    def plot_edge_class_distribution(self, graphs: list[GrooviGraph], all_sv_class_matches: list[list], tag: str):
        stats = {}
        for g, sv_class_matches in zip(graphs, all_sv_class_matches):
            for class_match in sv_class_matches:
                stats[class_match] = stats.get(class_match, 0) + 1
        
        sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)
        classes, frequencies = zip(*sorted_stats)
        
        with plt.style.context(self.style_overrides):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(classes, frequencies, color='skyblue', edgecolor='black')
            
            ax.set_xlabel('Edge Classes')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{tag} Dataset Edge Class Frequencies')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Ensure the image directory exists
            plot_path = os.path.join(self.config.stats_dir, f"edge_class_frequencies_{tag}.png")
            plt.savefig(plot_path)
            plt.close(fig)

    def write_label_summaries(self, label_summaries: list[list[str]], tag: str):
        summary_path = os.path.join(self.config.stats_dir, f"label_summary_{tag}.txt")
        with open(summary_path, "w") as f:
            f.write("Label Summary:\n")
            for label_summary in label_summaries:
                f.write('\n'.join(label_summary) + '\n')
    
    def write_stats(self, graphs, tag, all_sv_class_matches=None, label_summaries=None):
        # stats that require graphs only
        self.plot_node_feat_distribution(graphs, tag)

        # stats that require edge labels
        if all_sv_class_matches is not None:
            self.plot_edge_class_distribution(graphs, all_sv_class_matches, tag)
            self.write_edge_profiles(graphs, all_sv_class_matches, tag)
        
        # label summary 
        if label_summaries:
            self.write_label_summaries(label_summaries, tag)
            