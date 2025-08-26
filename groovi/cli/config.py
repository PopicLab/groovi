import datetime
import os
from pathlib import Path
import shutil
import yaml

class Config:
    def __init__(self, config_yaml: str, **kwargs):
        self.dataset_dir = str(Path(config_yaml).parent.resolve())
        with open(config_yaml, 'r') as file:
            config_data = yaml.safe_load(file)
        self.__dict__.update(config_data)
        self.enforced_params = [] # Changed to a list

    def _validate_params(self):
        for param_name in set(self.enforced_params): 
            assert hasattr(self, param_name), f"Parameter '{param_name}' is missing in the configuration"
    
    def __repr__(self):
        return '\n'.join(f"{key}: {value}" for key, value in self.__dict__.items() if not key.startswith('_'))

class GenerateConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # Changed enforced_params to a list of strings
        self.enforced_params = [
            # data files
            'bam',
            'vcf',
            'fa',
            # groovi parameters
            'cluster_radius',
            'evidence_threshold',
            # ... (add other params as needed)
        ]
        self._validate_params()
        # directory setup
        self.data_dir = os.path.join(self.dataset_dir, 'data')
        self.img_dir = os.path.join(self.dataset_dir, 'visuals')
        [shutil.rmtree(d, ignore_errors=True) for d in [self.data_dir, self.img_dir]]
        [os.makedirs(d, exist_ok=True) for d in [self.data_dir, self.img_dir]]
        self.stats_path = os.path.join(self.dataset_dir, 'stats.txt')

# Edges
class TrainEdgeClassificationConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # ... Fill

class CallConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # ... Fill
        
# Motifs
class TrainMotifConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        self.enforced_params = [
            # data specifications
            'train_graph_paths',
            'validation_graph_paths',
            'zygosity',
            'transforms',
            # extraction settings
            'train_extraction',
            'validation_extraction',
            'train_extraction_params',
            'validation_extraction_params',
            # model specifications
            'n_in_dim',
            'm_in_dim',
            'h_dim',
            'num_rels',
            'num_hidden_layers',
            'normalize',
            'pooling',
            # training specifications
            'num_epochs',
            'device',
            'batch_size',
            'lr',
            # validation specifications
            'validate_interval'
            # misc
            'logging_level',
            'visualize_every',
        ]
        self._validate_params()
        # data directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.chk_dir = os.path.join(self.dataset_dir, 'checkpoints')
        self.log_dir = os.path.join(self.dataset_dir, 'runs', timestamp)
        self.data_dir = os.path.join(self.dataset_dir, 'data')
        # visualization directories
        self.img_dir = os.path.join(self.dataset_dir, 'visuals')
        self.img_dir_graph = os.path.join(self.img_dir, 'graphs')
        self.img_dir_subgraph = os.path.join(self.img_dir, 'subgraphs')
        self.img_dir_plots = os.path.join(self.img_dir, 'plots')
        if self.visualize_every > 0:
            [shutil.rmtree(d, ignore_errors=True) for d in [self.img_dir_graph, self.img_dir_subgraph]]
        [shutil.rmtree(d, ignore_errors=True) for d in [self.chk_dir, self.img_dir_plots]]
        [os.makedirs(d, exist_ok=True) for d in [self.chk_dir, self.log_dir,  self.data_dir, self.img_dir, self.img_dir_graph, self.img_dir_subgraph, self.img_dir_plots]]

class InferenceMotifConfig(Config):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        # ... Fill