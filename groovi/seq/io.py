from groovi.seq.constants import *
from groovi.seq.io import *

from collections import namedtuple, OrderedDict
from pysam import VariantFile
import yaml

Chr = namedtuple('Chr', 'tid name len')
class ChrFAIndex:
    def __init__(self):
        self.tid2chr = OrderedDict()
        self.name2chr = OrderedDict()

    @staticmethod
    def load(fai_fname, all=False):
        chr_index = ChrFAIndex()
        with open(fai_fname, "r") as faidx:
            for tid, line in enumerate(faidx):
                tid = tid + 1
                name, length, _, _, _ = line[:-1].split()
                if not all and ("_" in name or "chrM" in name or "chrEBV" in name or "hs" in name): continue
                chr_index.add(Chr(tid, name, int(length)))
        return chr_index

    def add(self, chr):
        self.tid2chr[chr.tid] = chr
        self.name2chr[chr.name] = chr

    def get_tid_by_name(self, chr_name):
        return self.name2chr[chr_name].tid

    def get_chr_by_tid(self, tid):
        return self.tid2chr[tid]

    def get_chr_by_name(self, chr_name):
        return self.name2chr[chr_name]

    def contigs(self):
        return self.tid2chr.values()

    def get_chr_names(self) -> list[str]:
        return list(self.name2chr.keys())

    def get_chr_ids(self) -> list[int]:
        return list(self.tid2chr.keys())

    def get_chr_lengths(self) -> dict[int, int]:
        return {tid: self.get_chr_by_tid(tid).len for tid in self.tid2chr}
    
    def get_chr_prefix_sums(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping chromosome names to their starting positions in the genome.
        '''
        chr_lengths = self.get_chr_lengths()
        chr_prefix_sums = OrderedDict({1: 0})  # start with first chromosome at position 0
        for tid, chr in self.tid2chr.items():
            if tid == 1: continue
            chr_prefix_sums[tid] = chr_prefix_sums[tid-1] + chr_lengths[tid-1]
        return chr_prefix_sums
    
    def get_tid_to_str_conversion(self) -> dict[int, str]:
        return {tid: chr.name for tid, chr in self.tid2chr.items()}


class GenomeInfo:
    def __init__(self, config_yaml: str):
        required_keys = ['class', 'num_classes']  # add other required keys as needed
        with open(config_yaml, 'r') as file:
            self.config_data = yaml.safe_load(file)

            for key, info in self.config_data.items():
                for req_key in required_keys:
                    assert req_key in info, f"Key '{req_key}' is missing in the genome info for '{key}'"
    
    def get_label(self, key: str) -> str:
        assert key in self.config_data
        return self.config_data[key]['class']
    
    def get_num_classes(self, key: str) -> int:
        assert key in self.config_data
        return self.config_data[key]['num_classes']
