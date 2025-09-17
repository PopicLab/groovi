from groovi.seq.constants import *
from groovi.seq.io import *

import numpy as np
from pysam import VariantFile
from scipy.spatial import KDTree

'''
VCF Parsing, Method agnostic
'''
class VCFRecord:
    def __init__(self, start: int, stop: int, target: str, chrom: str, target_chrom: str, sv_id: str, op_type, sv_type: str, sv_grammar: str, sv_zygosity: str):
        self.start = start
        self.stop = stop
        self.target = target
        self.chrom = chrom
        self.target_chrom = target_chrom
        self.sv_id = sv_id
        self.op_type = op_type
        self.sv_type = sv_type
        self.sv_grammar = sv_grammar
        self.sv_zygosity = sv_zygosity

    def get_sv_type(self):
        return self.sv_type

class VCFRecordContainer:
    def __init__(self, vcf_path: str):
        self.vcf_path = vcf_path
        vcf_file: VariantFile = VariantFile(vcf_path)
        self.records = []

        for vcf_rec in vcf_file.fetch():
            vcf_info = dict(vcf_rec.info)
                
            start = vcf_rec.pos
            stop = vcf_rec.stop
            target = vcf_rec.info.get('TARGET', None)
            chrom = vcf_rec.chrom
            target_chrom = vcf_rec.info.get('TARGET_CHROM', None)
            sv_id = vcf_info['SVID']
            op_type = vcf_info['OP_TYPE']

            sv_type = vcf_info['SVTYPE']
            sv_grammar = vcf_info['GRAMMAR']
            sv_zygosity = "HOM" if vcf_rec.samples[0]['GT'] == (1, 1) else "HET"

            record = VCFRecord(start, stop, target, chrom, target_chrom, sv_id, op_type, sv_type, sv_grammar, sv_zygosity)
            self.records.append(record)
    
    def get_records(self):
        return self.records
    
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