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
from groovi.utils.utils import seed_everything, limit_cores
from groovi.graph.constants import *

SHARED_DEFAULTS = {
    'logging_level': "INFO",
    'class_set': "CPX10",
    "seed": 12345,
    "n_cpus": 20,
    "gpu_ids": [],
    'candidate_support_threshold': 2,
}

GENERATE_DEFAULTS = {
    'save_graphs': True,
    'visualize_graphs': False,
    'cluster_radius': 20,
    'cluster_support_threshold': 1,
    'node_feature_radius': 20
}

MODEL_DEFAULTS = {
    "ht_emb_dim": 2,
    "rgcn_input_dim": 100,
    "rgcn_hidden_dim": [100, 200],
    "mlp_hidden_dim": [200, 50, 50],
    "mlp_dropout": 0.3,
    "update": "WConcat",
    "normalize": None,
    "transforms": [] #['NORMALIZE_NODES_01', 'NORMALIZE_EDGES_01'],
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

GENERATE_DEFAULTS = {**SHARED_DEFAULTS, **GENERATE_DEFAULTS}
TRAIN_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **TRAIN_ENGINE_DEFAULTS}
DISCOVER_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **GENERATE_DEFAULTS}
DISCOVER_DEFAULTS.update({'save_graphs': False})

class Config:
    def __init__(self, config_yaml):
        with open(config_yaml, "r") as file:
            config_data = yaml.safe_load(file)
        self.__dict__.update(config_data)
        self.validate_params()

        # -------- setup output directory structure
        self.experiment_dir = str(Path(config_yaml).parent.resolve())
        self.log_dir = self.experiment_dir + "/logs/"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # -------- setup logging
        self.log_file = self.log_dir + 'main.log'
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName(self.logging_level),
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])

        # -------- setup shared data/model configs
        self.classes = groovi.seq.sv.TYPE_SET_TO_TYPES[groovi.seq.sv.SVTypeSet[self.class_set]]
        self.n_classes = len(self.classes)
        self.class2label = {sv_class: i for i, sv_class in enumerate(self.classes)}
        self.n_node_features = len(NodeFeature)
        self.n_edge_types = len(EdgeFeature)

    def set_defaults(self, default_values_dict):
        for k, v, in default_values_dict.items():
            if not hasattr(self, k): self.__setattr__(k, v)

    def validate_params(self):
        for param_name in set(self.required_params):
            assert hasattr(self, param_name), f"Parameter '{param_name}' is missing in the configuration"

    def __str__(self):
        s = "==== Config ====\n\t"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class GenerateConfig(Config):
    def __init__(self, config_yaml):
        self.set_defaults(GENERATE_DEFAULTS)
        self.required_params = ["bam", "vcf", "fa"]
        super().__init__(config_yaml)

        # -------- setup data inputs
        self.fai = "%s.fai" % self.fa
        self.fai_index = ChrFAIndex.load(self.fai)

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
        self.fai_index = ChrFAIndex.load(self.fai)

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
        self.required_params = ["graphs"]
        super().__init__(config_yaml)

        # -------- setup the output directory structure
        self.img_dir = os.path.join(self.experiment_dir, "vis")  # plots, graph visualizations
        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_checkpoints_dir = os.path.join(self.experiment_dir, "modes")  # model checkpoints
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_checkpoints_dir).mkdir(parents=True, exist_ok=True)
        self.model_path = os.path.join(self.experiment_dir, "final_model.pt")
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
            self.label_weights[self.class2label[c]] = 1.0 if c == SVBpType.NEG else 10.0

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

