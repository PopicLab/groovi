import groovi.utils.types
import groovi.seq.io as io
import numpy as np
from scipy.spatial import KDTree
from typing import Union
from collections import defaultdict
import logging


BP_TYPES = ["DEL", "INV", "DUP",
            "DUP_INV_A", "DUP_INV_B",
            "INVdel_A", "INVdel_B",
            "delINV_A", "delINV_B",
            "INVdup_A", "INVdup_B",
            "dupINV_A", "dupINV_B",
            "dupINVdel_A", "dupINVdel_B",
            "delINVdup_A", "delINVdup_B",
            "delINVdel_A",
            "dupINVdup_A"]

SVBpType = groovi.utils.types.make_enum("SVBpType", ["NEG"] + BP_TYPES, __name__)
SVTypeSet = groovi.utils.types.make_enum("SVBpTypeSet",["CPX12"], __name__)
SV_CLASS_SETS_GT = {SVTypeSet.CPX12}
TYPE_SET_TO_TYPES = {SVTypeSet.CPX12: [SVBpType.NEG, SVBpType.DEL, SVBpType.INV, SVBpType.DUP,
                                       SVBpType.DUP_INV_A, SVBpType.DUP_INV_B,
                                       SVBpType.INVdel_A, SVBpType.INVdel_B, SVBpType.delINV_A, SVBpType.delINV_B,
                                       SVBpType.INVdup_A, SVBpType.INVdup_B, SVBpType.dupINV_A, SVBpType.dupINV_B,
                                       SVBpType.dupINVdel_A, SVBpType.dupINVdel_B,
                                       SVBpType.delINVdup_A, SVBpType.delINVdup_B,
                                       SVBpType.delINVdel_A,
                                       SVBpType.dupINVdup_A,
                                       ]}

class SVBreakpoint:
    def __init__(self, sv_type, chr1, pos1, chr2, pos2, qual, gt):
        self.type = sv_type
        self.chr1 = chr1
        self.pos1 = pos1
        self.chr2 = chr2
        self.pos2 = pos2
        self.qual = qual
        self.gt = gt
        self.len = abs(self.pos1 - self.pos2) if self.chr1 == self.chr2 else None
        self.orientation = None
        self.evidence = None
        self.evidence_fuzzy = None
        self.annotation = None

    @classmethod
    def from_vcf(cls, rec):
        gt = (None, None)
        if 'GT' in rec.samples[rec.samples[0].name]: gt = rec.samples[rec.samples[0].name]['GT']
        assert 'SVTYPE' in rec.info
        chr1 = rec.contig
        chr2 = chr1
        if 'END_CHROM' in rec.info and rec.info['END_CHROM'] is not None:
            chr2 = rec.info['END_CHROM']
        sv = SVBreakpoint(rec.info['SVTYPE'], chr1, rec.start - 1, chr2, rec.stop - 1, rec.qual, gt)
        if 'ORIENTATION' in rec.info and rec.info['ORIENTATION'] is not None:
            sv.orientation = rec.info['ORIENTATION']
        if 'ReadSupport' in rec.info and rec.info['ReadSupport'] is not None:
            sv.evidence = int(rec.info['ReadSupport'])
        if 'ReadSupportFuzzy' in rec.info and rec.info['ReadSupportFuzzy'] is not None:
            sv.evidence_fuzzy = int(rec.info['ReadSupportFuzzy'])
        return sv

    def __str__(self):
        return '\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())

    @staticmethod
    def compare(sv1, sv2):
        return sv1.start - sv2.start

    @staticmethod
    def compare_by_score(sv1, sv2):
        return float(sv1.qual) - float(sv2.qual)

    @staticmethod
    def compare_by_support(sv1, sv2):
        return sv1.evidence - sv2.evidence

    def __lt__(self, sv):
        return self.start < sv.start

    @staticmethod
    def get_sv_type_labels(sv_classes):
        return {sv_class: i for i, sv_class in enumerate(sv_classes)}


class SVKDTree:
    """
    Efficient querying of SV breakpoint records by L2 distance using KDTrees (stratified by chromosome pair).
    Supports filtering by: distance and breakpoint type/orientation
    """
    def __init__(self, vcf_path: str):
        logging.info('Loading SV breakpoint from {}'.format(vcf_path))
        self.sv_bp_records = [sv for sv in io.vcf_iter(vcf_path)]
        self.chr2kdtrees = {} # maps (chr1, chr2) pairs -> KDTree
        self.chr2idx = {}
        ref_pos_by_chr = defaultdict(list)
        for idx, sv in enumerate(self.sv_bp_records):
            ref_pos_by_chr[tuple(sorted([sv.chr1, sv.chr2]))].append(idx)
        for chr_key, sv_bkp_idx in ref_pos_by_chr.items():
            self.chr2idx[chr_key] = sv_bkp_idx
            self.chr2kdtrees[chr_key] = KDTree( np.array([[self.sv_bp_records[idx].pos1, self.sv_bp_records[idx].pos2]
                                                          for idx in ref_pos_by_chr[chr_key]]))

    def find_best_label(self, query: np.ndarray, chr1: str, chr2: str, k: int = 5,
                        orientation = None, max_distance = 100):
        if (chr1, chr2) not in self.chr2kdtrees: return SVBpType.NEG, None
        distances, sv_bkp_idx = self.chr2kdtrees[(chr1, chr2)].query(query, k=k)

        matched_sv_breakpoints = []
        for sv_idx, distance in zip(sv_bkp_idx, distances):
            if distance > max_distance: continue
            sv_match = self.sv_bp_records[self.chr2idx[(chr1, chr2)][sv_idx]]
            if orientation is not None and sv_match.orientation != orientation: continue
            matched_sv_breakpoints.append((sv_match.type, distance))
        return (SVBpType.NEG, None) if not matched_sv_breakpoints else matched_sv_breakpoints[0]