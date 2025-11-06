from groovi.seq.io import *

import datetime
import logging
import os
from pathlib import Path
import shutil
import sys
import torch
import yaml

SHARED_DEFAULTS = {
    'logging_level': "INFO",
    "seed": None,
    "num_dgl_cores": None
}

GENERATE_DEFAULTS = {
    'save_graphs': True,
    'visualize_graphs': False,
    'cluster_radius': 10,
    'cluster_support_threshold': 2
}

MODEL_SPECIFICATIONS = [
    "ht_emb_dim",
    "hidden_dim",
    "num_convolutions",
    "num_hidden_layers",
    "update",
    "normalize",
]

GENERATE_DEFAULTS = {**SHARED_DEFAULTS, **GENERATE_DEFAULTS}

class Config:
    def __init__(self, config_yaml):
        with open(config_yaml, "r") as file:
            config_data = yaml.load(file, Loader=yaml.FullLoader)
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
        #self.n_classes = len(self.classes)

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

        self.validate_params()
        logging.info(self)


class TrainConfig(Config):
    def __init__(self, config_yaml):
        super().__init__(config_yaml)
        # ... Fill
        self.enforced_params = [
            # data specifications
            "graph_paths",
            "transforms",
            # training specifications
            "validation_ratio",
            "batch_size",
            "num_epochs",
            "learning_rate",
            "device",
            # misc
            "graph_visualize_every",  # post transformation visualization frequency
        ] + MODEL_SPECIFICATIONS + INIT_SPECIFICATIONS
        self._validate_params()

        if self.device == "cuda":
            assert torch.cuda.is_available()

        # data directories
        self.data_dir = os.path.join(
            self.experiment_dir, "data"
        )  # misc data, ie labeling statistics
        self.img_dir = os.path.join(
            self.experiment_dir, "visuals"
        )  # plots, graph visualizations
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(
            self.experiment_dir, "runs", timestamp
        )  # tensorboard logs
        self.model_save_dir = os.path.join(
            self.experiment_dir, "model"
        )  # model checkpoints
        [
            os.makedirs(dir_path, exist_ok=True)
            for dir_path in [
                self.data_dir,
                self.img_dir,
                self.log_dir,
                self.model_save_dir,
            ]
        ]
        self.stats_path = os.path.join(self.experiment_dir, "stats.txt")


class TrainGraphConfig(TrainConfig):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # ... Fill
        self.enforced_params += [
            # additional model specifications
            "pooling",
            "label_key",
        ]
        self._validate_params()


class CallConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        self.enforced_params = [
            # data specifications
            "graph_paths",
            "transforms",
        ]  + MODEL_SPECIFICATIONS + INIT_SPECIFICATIONS
        self._validate_params()
        self.stats_path = os.path.join(self.experiment_dir, "stats.txt")
        self.img_dir = os.path.join(
            self.experiment_dir, "visuals"
        )  # plots, graph visualizations
        [os.makedirs(dir_path, exist_ok=True) for dir_path in [self.img_dir]]
