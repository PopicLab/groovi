from groovi.seq.constants import *
from groovi.seq.io import *

from collections import namedtuple, OrderedDict
from pysam import VariantFile

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
    