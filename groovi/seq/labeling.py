from groovi.seq.constants import *
from groovi.seq.io import *

import numpy as np
from scipy.spatial import KDTree

'''
Groovi specific VCF processing, adaptable to future labeling needs
'''
class AdjRecord:
    def __init__(self, vcf_record: VCFRecord):
        self.vcf_record = vcf_record
        self.start = vcf_record.start
        self.stop = vcf_record.stop

        # TODO: change depending on label
        self.label = SV_CLASS_TO_ID_FN(vcf_record.get_sv_type())

    def get_start_pos(self):
        return self.start
    
    def get_target_pos(self):
        return self.stop

    def get_start_chrom(self) -> str:
        return self.vcf_record.chrom

    def get_target_chrom(self) -> str:
        return self.vcf_record.target_chrom

    def get_label(self):
        return self.label

    def __repr__(self):
        return f'AdjRecord({self.vcf_record.sv_id}, {self.vcf_record.chrom}:{self.start}-{self.stop}, {self.vcf_record.target_chrom}, {self.vcf_record.sv_type})'

class AdjRecordContainer:
    def __init__(self, vcf_record_container: VCFRecordContainer):
        self.edges = []
        for record in vcf_record_container.get_records():
            edge_record = AdjRecord(record)
            self.edges.append(edge_record)
        
    def get_adj_records(self):
        return self.edges    
    
'''
Structures for querying Groovi specific VCF Records
'''
class VCF_KDTree:
    def __init__(self, vcf_path: str):
        edge_record_container = AdjRecordContainer(VCFRecordContainer(vcf_path))
        self.edges = edge_record_container.get_adj_records()
        coordinates = np.array([[rec.start, rec.stop] for rec in self.edges])
        self.kdtree = KDTree(coordinates)

    def query_top_label(self, chrom: str, chrom_target: str, query: np.ndarray, k: int = 5) -> list[VCFRecord]:
        # Query the KDTree for the top k records within a certain distance
        distances, indices = self.kdtree.query(query, k=k)
        # TODO ... other filters (OP type, head/tail, ...)
        # ie indices = [i for i in indices if self.edges[i].get_start_chrom() == chrom and self.edges[i].get_target_chrom() == chrom_target]
        filtered_records = [self.edges[i] for i in indices]

        # TODO: handle no matches
        if len(filtered_records) == 0:
            raise ValueError("No matching records found in VCF for the given query.")
        
        label = filtered_records[0].get_label()

        return label