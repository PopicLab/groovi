import groovi.utils.types
import groovi.seq.io as io
import numpy as np
from scipy.spatial import KDTree
from intervaltree import IntervalTree
from typing import Union
from collections import defaultdict
import logging
from pycues import *


BP_TYPES = ["DEL", "DUP", "INV", "INV_A", "INV_B",
            "DUP_INV_A", "DUP_INV_B", "DUP_INV_C",
            "INVdel_A", "INVdel_B",
            "delINV_A", "delINV_B",
            "INVdup_A", "INVdup_B",
            "dupINV_A", "dupINV_B",
            "dupINVdel_A", "dupINVdel_B",
            "delINVdup_A", "delINVdup_B",
            "delINVdel_A", "delINVdel_B",
            "dDUP_A", "dDUP_B",
            "dDUP_INV_A", "dDUP_INV_B", "dDUP_INV_C", "dDUP_INV_D",
            "nrTRA_A", "nrTRA_B",
            "INV_nrTRA_A", "INV_nrTRA_B", "INV_nrTRA_C",
            "dupINV3_A", "dupINV3_B",
            "dupINVdup_A", "dupINVdup_B"]

SVBpType = groovi.utils.types.make_enum("SVBpType", ["NEG", "BND"] + BP_TYPES, __name__)
SVTypeSet = groovi.utils.types.make_enum("SVBpTypeSet",["CPX12", "CPX10", "CPX15"], __name__)
SV_CLASS_SETS_GT = {SVTypeSet.CPX12}
TYPE_SET_TO_TYPES = {SVTypeSet.CPX12: [SVBpType.NEG, SVBpType.DEL, SVBpType.DUP, SVBpType.INV_A, SVBpType.INV_B,
                                       SVBpType.DUP_INV_A, SVBpType.DUP_INV_B,
                                       SVBpType.INVdel_A, SVBpType.INVdel_B, SVBpType.delINV_A, SVBpType.delINV_B,
                                       SVBpType.INVdup_A, SVBpType.INVdup_B, SVBpType.dupINV_A, SVBpType.dupINV_B,
                                       SVBpType.dupINVdel_A, SVBpType.dupINVdel_B,
                                       SVBpType.delINVdup_A, SVBpType.delINVdup_B,
                                       SVBpType.delINVdel_A,
                                       SVBpType.dupINVdup_A,
                                       ],
                     SVTypeSet.CPX10: [SVBpType.NEG, SVBpType.DEL, SVBpType.DUP, SVBpType.INV,
                                       SVBpType.INV_A, SVBpType.INV_B,
                                       SVBpType.DUP_INV_A, SVBpType.DUP_INV_B, SVBpType.DUP_INV_C,
                                       SVBpType.INVdel_A, SVBpType.INVdel_B, SVBpType.delINV_A, SVBpType.delINV_B,
                                       SVBpType.dupINVdel_A, SVBpType.dupINVdel_B,
                                       SVBpType.delINVdup_A, SVBpType.delINVdup_B,
                                       SVBpType.delINVdel_A, SVBpType.delINVdel_B,
                                       SVBpType.dupINVdup_A, SVBpType.dupINVdup_B,
                                       ],
                     SVTypeSet.CPX15: [SVBpType.NEG, SVBpType.DEL, SVBpType.DUP, SVBpType.INV_A, SVBpType.INV_B,
                                       SVBpType.DUP_INV_A, SVBpType.DUP_INV_B,
                                       SVBpType.INVdel_A, SVBpType.INVdel_B, SVBpType.delINV_A, SVBpType.delINV_B,
                                       SVBpType.dupINVdel_A, SVBpType.dupINVdel_B,
                                       SVBpType.delINVdup_A, SVBpType.delINVdup_B,
                                       SVBpType.delINVdel_A,
                                       SVBpType.dupINVdup_A,
                                       SVBpType.dDUP_A, SVBpType.dDUP_B,
                                       SVBpType.dDUP_INV_A, SVBpType.dDUP_INV_B, SVBpType.dDUP_INV_C, SVBpType.dDUP_INV_D,
                                       SVBpType.nrTRA_A, SVBpType.nrTRA_B,
                                       SVBpType.INV_nrTRA_A, SVBpType.INV_nrTRA_B, SVBpType.INV_nrTRA_C,
                                       SVBpType.dupINV3_A, SVBpType.dupINV3_B,
                                       ]
                     }

class SVBreakpoint:
    def __init__(self, sv_type, chr1, pos1, chr2, pos2, qual, gt=(None, None), evidence=None):
        self.type = sv_type
        self.chr1 = chr1
        self.pos1 = pos1
        self.chr2 = chr2
        self.pos2 = pos2
        self.qual = qual
        self.gt = gt
        self.len = abs(self.pos1 - self.pos2) if self.chr1 == self.chr2 else None
        self.orientation = None
        self.evidence = evidence
        self.evidence_fuzzy = None
        self.source = None
        self.annotation = None

    @classmethod
    def from_vcf(cls, rec):
        gt = (None, None)
        if len(rec.samples) > 0 and 'GT' in rec.samples[rec.samples[0].name]: gt = rec.samples[rec.samples[0].name]['GT']
        chr1 = rec.contig
        chr2 = chr1
        if 'END_CHROM' in rec.info and rec.info['END_CHROM'] is not None:
            chr2 = rec.info['END_CHROM']
        if 'SVTYPE' in rec.info: sv_type = rec.info['SVTYPE']
        elif 'BKP_TYPE' in rec.info: sv_type = rec.info['BKP_TYPE']
        else: sv_type = SVBpType.BND
        sv = SVBreakpoint(sv_type, chr1, rec.start, chr2, rec.stop, rec.qual, gt)
        if 'ORIENTATION' in rec.info and rec.info['ORIENTATION'] is not None:
            sv.orientation = rec.info['ORIENTATION']
        if 'SUPPORT' in rec.info and rec.info['SUPPORT'] is not None:
            sv.evidence = int(rec.info['SUPPORT'])
        if 'ReadSupportFuzzy' in rec.info and rec.info['ReadSupportFuzzy'] is not None:
            sv.evidence_fuzzy = int(rec.info['ReadSupportFuzzy'])
        if 'SOURCE' in rec.info and rec.info['SOURCE'] is not None:
            sv.source = rec.info['SOURCE']
        return sv

    @classmethod
    def from_junction(cls, junction):
        sv = SVBreakpoint("BND", junction.chr1, junction.pos1, junction.chr2, junction.pos2, 0,
                            (None, None), junction.support)
        sv.orientation = junction.orientation.name
        sv.source = junction.source.name
        return sv

    def to_junction(self):
        return Junction(chr1=self.chr1, chr2=self.chr2, pos1=self.pos1, pos2=self.pos2,
                        orientation=getattr(JOrientation, self.orientation),
                        evidence_type=getattr(JSource, self.source), support=self.evidence)

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
        return self.pos1 < sv.pos1

    @staticmethod
    def get_sv_type_labels(sv_classes):
        print({sv_class: i for i, sv_class in enumerate(sv_classes)})
        return {sv_class: i for i, sv_class in enumerate(sv_classes)}


class SVDB:
    """
    Efficient querying of SV breakpoint records:
     * using KDTrees (stratified by chromosome pair) with L2 distance
     * using Interval trees with overlap and size constraints
    Supports filtering by: distance and breakpoint type/orientation
    """
    def __init__(self, vcf_path, sv_classes_to_label, store_type="KDTree"):
        logging.info('Loading SV breakpoint records from {}'.format(vcf_path))
        self.sv_classes_to_label = sv_classes_to_label
        self.sv_bp_records = [sv for sv in io.vcf_iter(vcf_path)]
        self.chr2idx = defaultdict(list)
        for idx, sv in enumerate(self.sv_bp_records):
            self.chr2idx[self._get_key(sv.chr1, sv.chr2)].append(idx)

        if store_type == "KDTree":
            self.chr2kdtrees = {}  # maps (chr1, chr2) pairs -> KDTree
            for chr_key, sv_bkp_idx in self.chr2idx.items():
                self.chr2kdtrees[chr_key] = KDTree(np.array([[self.sv_bp_records[idx].pos1,
                                                              self.sv_bp_records[idx].pos2] for idx in sv_bkp_idx]))
        else:
            self.chr2intvtrees = {}
            for chr_key, sv_bkp_idx in self.chr2idx.items():
                self.chr2intvtrees[chr_key] = IntervalTree()
                for idx in sv_bkp_idx:
                    self.chr2intvtrees[chr_key].addi(self.sv_bp_records[idx].pos1, self.sv_bp_records[idx].pos2, idx)

    def _get_key(self, chr1, chr2):
        return tuple(sorted([chr1, chr2]))


    def find_best_label(self, query: np.ndarray, chr1: str, chr2: str, k: int = 5,
                        orientation = None, max_distance = 100):
        chr_key = self._get_key(chr1, chr2)
        candidates = []
        if hasattr(self, 'chr2kdtrees'):
            if chr_key not in self.chr2kdtrees: return SVBpType.NEG, None
            distances, sv_bkp_idx = self.chr2kdtrees[chr_key].query(query, k=k)
            for sv_idx, distance in zip(sv_bkp_idx, distances):
                if distance < max_distance: candidates.append((distance, sv_idx))
        else:
            if chr_key not in self.chr2intvtrees: return SVBpType.NEG, None
            size_sim = 0.5
            for c in self.chr2intvtrees[(chr1, chr2)].overlap(query[0], query[1]):
                if abs(c.begin - query[0]) > max_distance or abs(c.end - query[1]) > max_distance: continue
                size_frac = (min(abs(c.end - c.begin), abs(query[0] - query[1])) /
                             max(abs(c.end - c.begin), abs(query[0] - query[1])))
                if size_frac < size_sim: continue
                candidates.append((1-size_frac, c.data))

        for dist, sv_idx in sorted(candidates, key=lambda item: item[0]):
            sv_match = self.sv_bp_records[self.chr2idx[chr_key][sv_idx]]
            if sv_match.type not in self.sv_classes_to_label: continue
            if orientation is not None and sv_match.orientation != orientation: continue
            return sv_match.type, dist
        return SVBpType.NEG, None
