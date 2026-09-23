from groovi.seq.io import *
import groovi.seq.sv
import datetime
import logging
import os
from pathlib import Path
import shutil
import sys
import torch
import yaml
from groovi.utils.logging import create_logger
from groovi.utils.utils import seed_everything, limit_cores
from groovi.graph.constants import *
from groovi.graph.transforms import *

SHARED_DEFAULTS = {
    'logging_level': "INFO",
    'class_set': "CPX10",
    'node_feature_set': 'A',
    'edge_feature_set': 'A',
    "seed": 12345,
    "n_cpus": 20,
    "gpu_ids": [],
    'candidate_support_threshold': 1,
    "transforms": ["NORMALIZE_FEATURES"],
    "feature_norm_vmax": 200,
    'feature_norm_vmin': 0,
    'node_feature_norm': "LOG1P",
    'edge_feature_norm': "RATIO",
    "features_to_mask": [],
    "use_cached_graphs": False
}

GENERATE_DEFAULTS = {
    'save_graphs': True,
    'visualize_graphs': False,
    'cluster_radius': 20,
    'cluster_support_threshold': 1,
    'cluster_support_criteria': 'sum',
    'node_stats_buffer': 20,
    'node_stats_buffer_overlap': 0,
    'min_clipped_len': 10,
    'min_discord_len': 40,
    'min_mapq': 10,
    'max_discordance': 0.02,
    'patch_alns': True,
    'recover_ins': True,
    'recover_clipping': True,
    'recover_subfrag': True,
    'recover_missing_segs': True,
    'interchrom': False,
    'tiny_segment_threshold': 5,
    'cluster': True,
    'use_cached_junctions': False,
    'aln_batch_size': 500000,
}

MODEL_DEFAULTS = {
    "rgcn_hidden_dim": [256, 128],
    "mlp_hidden_dim": [128, 64, 64],
    "mlp_dropout": 0.3,
    "rgcn_dropout": 0.3,
    "update": "WConcat",
    'node_feature_proj_dim': None,
    'edge_feature_proj_dim': None,
    'augment_nodes_rgcn': True,
    'node_augmentation': "DIFF", #RATIO
    'edge_representation': 'SUPPORT',
    'rgcn_layer': "Default", #Custom
    'rgcn_custom_update': True, # only for Custom layer
}

TRAIN_ENGINE_DEFAULTS = {
    "pretrained_model": None,
    "num_epochs": 100,
    "validation_ratio": 0.1,
    "batch_size": 1,
    'metrics_report_interval': 10,
    "graph_visualization_interval": 10, # post transformation visualization frequency
    'model_checkpoint_interval': 10000,
    'learning_rate': 0.0001,
    'learning_rate_decay_interval': 5,
    'learning_rate_decay_factor': 1,
}

STITCHING_DEFAULTS = {
    'cluster_support_threshold': 2, # overrides default generation cluster_support_threshold
    "breakpoint_threshold": 500,
    "sep_frag_recs": True,
    "unmatched_edge_file": None,
    "filter_overlaps": False,
    "deduplicate_calls": True,
    "gap_file": None,
    "minimum_length": 50,
    "use_cached_predictions": False,
    "prioritize_evidence": True,
    'maximum_length': 1000000,
    "minimum_fragment_length": 30,
    'duplication_threshold': 10,
}

GENERATE_DEFAULTS = {**SHARED_DEFAULTS, **GENERATE_DEFAULTS}
TRAIN_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **TRAIN_ENGINE_DEFAULTS}
DISCOVER_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **GENERATE_DEFAULTS, **STITCHING_DEFAULTS}
DISCOVER_DEFAULTS.update({
    'save_graphs': False,
    'use_cached_graphs': False,
})

class Config:
    def __init__(self, config_yaml):
        with open(config_yaml, "r") as file:
            config_data = yaml.safe_load(file)
        self.__dict__.update(config_data)
        self.validate_params()

        # -------- setup output directory structure
        self.experiment_dir = str(Path(config_yaml).parent.resolve())
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.stats_dir = os.path.join(self.experiment_dir, "stats")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.stats_dir).mkdir(parents=True, exist_ok=True)

        # -------- setup logging
        self.log_file = os.path.join(self.log_dir, 'main.log')
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName(self.logging_level),
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])

        # -------- additional logs
        self.stitched_filter_logger = create_logger("post_stitching_filter", os.path.join(self.log_dir, 'post_stitching_filter.log'), log_to_console=False)

        # -------- setup shared data/model configs
        self.classes = groovi.seq.sv.TYPE_SET_TO_TYPES[groovi.seq.sv.SVTypeSet[self.class_set]]
        self.n_classes = len(self.classes)
        self.class2label = {sv_class: i for i, sv_class in enumerate(self.classes)}
        self.node_feature_set = NODE_FEATURE_SETS[self.node_feature_set]
        self.node_aln_feature_set = list(set(self.node_feature_set) - set(AUX_NODE_FEATURES))
        self.edge_feature_set = EDGE_FEATURE_SETS[self.edge_feature_set]
        self.n_node_features = len(self.node_feature_set)
        self.n_edge_types = len(self.edge_feature_set)
        self.node_feat2id = {feat: i for i, feat in enumerate(self.node_feature_set)}
        self.edge_feat2id = {feat: i for i, feat in enumerate(self.edge_feature_set)}
        self.id2edge_feat = {i: feat for feat, i in self.edge_feat2id.items()}
        self.transforms = [TRANSFORMS(self)[t] for t in self.transforms]

    def set_defaults(self, default_values_dict):
        for k, v, in default_values_dict.items():
            if not hasattr(self, k): self.__setattr__(k, v)

    def validate_params(self):
        for param_name in set(self.required_params):
            assert hasattr(self, param_name), f"Parameter '{param_name}' is missing in the configuration"
        if hasattr(self, "cluster_support_criteria"):
            assert self.cluster_support_criteria in ("sum", "max"), \
                f"'cluster_support_criteria' must be 'sum' or 'max', got '{self.cluster_support_criteria}'"

    def __str__(self):
        s = "==== Config ====\n\t"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class GenerateConfig(Config):
    def __init__(self, config_yaml):
        self.set_defaults(GENERATE_DEFAULTS)
        self.required_params = ["bam", "fa"]
        super().__init__(config_yaml)

        # -------- setup data inputs
        self.fai = "%s.fai" % self.fa
        self.fai_index = ChrFAIndex.load(self.fai, all=True)

        # -------- setup the output directory structure
        self.graph_dir = os.path.join(self.experiment_dir, "graphs")
        self.img_dir = os.path.join(self.experiment_dir, "vis")
        Path(self.graph_dir).mkdir(parents=True, exist_ok=True)
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        self.stats_file_path = os.path.join(self.log_dir, "stats.txt")
        self.bkp_raw_vcf_file_path = os.path.join(self.log_dir, "groovi_bkps_raw.vcf")
        self.bkp_cluster_vcf_file_path = os.path.join(self.log_dir, "groovi_bkps_cluster.vcf")
        self.validate_params()
        logging.info(self)


class DiscoverConfig(Config):
    def __init__(self, config_yaml):
        self.set_defaults(DISCOVER_DEFAULTS)
        self.required_params = ["bam", "fa", "model"]
        super().__init__(config_yaml)

        # -------- setup data inputs
        self.fai = "%s.fai" % self.fa
        self.fai_index = ChrFAIndex.load(self.fai, all=True)

        # -------- setup the output directory structure
        self.graph_dir = os.path.join(self.experiment_dir, "graphs")
        self.img_dir = os.path.join(self.experiment_dir, "vis")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        Path(self.graph_dir).mkdir(parents=True, exist_ok=True)
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.stats_file_path = os.path.join(self.log_dir, "stats.txt")
        self.bkp_raw_vcf_file_path = os.path.join(self.results_dir, "groovi_bkps_raw.vcf")
        self.bkp_cluster_vcf_file_path = os.path.join(self.results_dir, "groovi_bkps_cluster.vcf")
        self.bkp_vcf_file_path = os.path.join(self.results_dir, "groovi_bkps_classified.vcf")
        self.complex_sv_vcf_file_path = os.path.join(self.results_dir, "groovi.vcf")
        # TODO: add other outputs
        self.bkp_unmatched_file_path = os.path.join(self.results_dir, "groovi_bkps_unmatched.vcf")
        self.complex_sv_unfiltered_file_path = os.path.join(self.results_dir, "groovi_unfiltered.vcf")

        # -------- setup compute
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            limit_cores(self.n_cpus)

        # -------- setup env
        seed_everything(self.seed)

        self.validate_params()
        logging.info(self)


class TrainEdgeConfig(Config):
    def __init__(self, config_yaml):
        self.set_defaults(TRAIN_DEFAULTS)
        self.required_params = ["graphs", "graphs_info"]
        super().__init__(config_yaml)

        # -------- setup the output directory structure
        self.img_dir = os.path.join(self.experiment_dir, "vis")  # plots, graph visualizations
        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_checkpoints_dir = os.path.join(self.experiment_dir, "models")  # model checkpoints
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_checkpoints_dir).mkdir(parents=True, exist_ok=True)
        self.model_path = os.path.join(self.experiment_dir, "best_model.pt")
        self.stats_file_path = os.path.join(self.log_dir, "stats.txt")

        # -------- setup compute
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            limit_cores(self.n_cpus)

        # -------- setup env
        seed_everything(self.seed)

        # -------- training-related setup
        self.label_weights = torch.ones(self.n_classes)
        for c in self.classes:
            self.label_weights[self.class2label[c]] = 1.0 #if c == SVBpType.NEG else 10.0

        self.validate_params()
        logging.info(self)

class TrainGraphConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # ... Fill
        self.enforced_params += [
            # additional model specifications
            "pooling",
            "label_key",
        ]
        self._validate_params()

