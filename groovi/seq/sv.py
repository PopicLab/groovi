import groovi.utils.types
import groovi.seq.io as io

import numpy as np
from scipy.spatial import KDTree
from intervaltree import IntervalTree
from collections import defaultdict
from dataclasses import dataclass
import logging
from pycues import *
from copy import copy

JUNCTION_TYPES = [
    # simple
    "DEL", "DUP",
    "INV", "INV_A", "INV_B",
    # 2 bkp
    "DUP_INV_A", "DUP_INV_B", "DUP_INV_C",
    # 3 bkp
    "INVdel_A", "INVdel_B",
    "delINV_A", "delINV_B",
    "INVdup_A", "INVdup_B",
    "dupINV_A", "dupINV_B",
    # 4 bkp
    "dupINVdel_A", "dupINVdel_B",
    "delINVdup_A", "delINVdup_B",
    "delINVdel_A", "delINVdel_B",
    "dupINVdup_A", "dupINVdup_B",
    # dispersed
    "dDUP_A", "dDUP_B", "dDUP_C", "dDUP_D",
    "nrTRA_A", "nrTRA_B",  # NOTE Currently edges [A,B,A] subject to change
    "INV_nrTRA_A", "INV_nrTRA_B", "INV_nrTRA_C",
    "INV_nrTRA_D", "INV_nrTRA_E", "INV_nrTRA_F",
    "INV_dDUP_A", "INV_dDUP_B", "INV_dDUP_C", "INV_dDUP_D",
    # non class types
    "NEG",
]

JunctionType = groovi.utils.types.make_enum("JunctionType", JUNCTION_TYPES, __name__)
JunctionSet = groovi.utils.types.make_enum("JunctionSet", ["CPX10", "CPX14"], __name__)

TYPE_SET_TO_TYPES = {
    JunctionSet.CPX10: [JunctionType.NEG, JunctionType.DEL, JunctionType.DUP, JunctionType.INV_A, JunctionType.INV_B,
                        JunctionType.DUP_INV_A, JunctionType.DUP_INV_B, JunctionType.DUP_INV_C,
                        JunctionType.INVdel_A, JunctionType.INVdel_B, JunctionType.delINV_A, JunctionType.delINV_B,
                        JunctionType.dupINVdel_A, JunctionType.dupINVdel_B,
                        JunctionType.delINVdup_A, JunctionType.delINVdup_B,
                        JunctionType.delINVdel_A, JunctionType.delINVdel_B,
                        JunctionType.dupINVdup_A, JunctionType.dupINVdup_B,
                        ],
    JunctionSet.CPX14: [JunctionType.NEG, JunctionType.DEL, JunctionType.DUP, JunctionType.INV_A, JunctionType.INV_B,
                        JunctionType.DUP_INV_A, JunctionType.DUP_INV_B, JunctionType.DUP_INV_C,
                        JunctionType.INVdel_A, JunctionType.INVdel_B, JunctionType.delINV_A, JunctionType.delINV_B,
                        JunctionType.dupINVdel_A, JunctionType.dupINVdel_B,
                        JunctionType.delINVdup_A, JunctionType.delINVdup_B,
                        JunctionType.delINVdel_A, JunctionType.delINVdel_B,
                        JunctionType.dupINVdup_A, JunctionType.dupINVdup_B,
                        JunctionType.nrTRA_A, JunctionType.nrTRA_B,
                        JunctionType.dDUP_A, JunctionType.dDUP_B,
                        JunctionType.INV_nrTRA_A, JunctionType.INV_nrTRA_B, JunctionType.INV_nrTRA_C,
                        JunctionType.INV_dDUP_A, JunctionType.INV_dDUP_B,
                        ]
}

class SVJunction:
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
        else: sv_type = JunctionType.BND
        sv = SVJunction(sv_type, chr1, rec.start, chr2, rec.stop, rec.qual, gt)
        if 'ORIENTATION' in rec.info and rec.info['ORIENTATION'] is not None:
            sv.orientation = rec.info['ORIENTATION']
        if 'SUPPORT' in rec.info and rec.info['SUPPORT'] is not None:
            sv.evidence = int(rec.info['SUPPORT'])
        if 'ReadSupportFuzzy' in rec.info and rec.info['ReadSupportFuzzy'] is not None:
            sv.evidence_fuzzy = int(rec.info['ReadSupportFuzzy'])
        if 'SOURCE' in rec.info and rec.info['SOURCE'] is not None:
            sv.source = rec.info['SOURCE']
        return sv

    def __repr__(self):
        return f"SVBreakpoint({self.chr1}:{self.pos1}-{self.chr2}:{self.pos2} | type={self.type})"

    @classmethod
    def from_junction(cls, junction):
        sv = SVJunction("BND", junction.chr1, junction.pos1, junction.chr2, junction.pos2, 0,
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


class JunctionDB:
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
            if chr_key not in self.chr2kdtrees: return JunctionType.NEG, None
            distances, sv_bkp_idx = self.chr2kdtrees[chr_key].query(query, k=k)
            for sv_idx, distance in zip(sv_bkp_idx, distances):
                if distance < max_distance: candidates.append((distance, sv_idx))
        else:
            if chr_key not in self.chr2intvtrees: return JunctionType.NEG, None
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
        return JunctionType.NEG, None


class ComplexSV:
    def __init__(self, spec, chrom, qual=None, records=None):
        self.spec = spec
        self.sv_type = self.spec.get('sv_type')
        self.direction = self.spec.get('direction')
        self.chrom = chrom
        self.qual = qual
        self.records = records if isinstance(records, dict) else dict(zip(self.spec['candidates'], records))
        self.fragments = self.spec['fragment_fn'](self)
        self.validated = all(fn(self) for fn in self.spec['validation_fns'])
        self.grammar = self.spec['grammar']
        self.target, self.target_chrom = None, None
        if self.spec.get('dispersion', False):
            moved = next(frag for frag in self.fragments if frag.target_start is not None)
            self.target, self.target_chrom = moved.target_start, moved.target_chrom
            if isinstance(self.grammar, dict):
                self.direction = 'left' if self.target < moved.start else 'right'
                self.grammar = self.grammar[self.direction]
        self.grammars = [(self.grammar, self.fragments)]
        for grammar, fragments in self.spec.get('alternative_grammars_fn', lambda csv: [])(self):
            alternative = copy(self)
            alternative.fragments = fragments
            if all(fn(alternative) for fn in self.spec['validation_fns']):
                self.grammars.append((grammar, fragments))

    def __str__(self):
        return repr((self.sv_type, self.chrom, self.qual, self.records))

    def get_vcf_info(self):
        start = min(frag.start for frag in self.fragments)
        stop = max(frag.end for frag in self.fragments)
        vcf_info = {'chrom': self.chrom,
                    'start': start,
                    'stop': stop,
                    'sv_type': self.sv_type,
                    'qual': self.qual,
                    'sv_len': stop - start,
                    'gt': (None, None), 'op_type': self.spec.get('op_name', self.sv_type)}
        if self.target:
            vcf_info['target'] = self.target
        return vcf_info

class JunctionStitchIndex:
    def __init__(self, buffer):
        self.buffer = buffer
        self.junctions = []
        self.ends = defaultdict(IntervalTree)
        self.spans = defaultdict(IntervalTree)

    def add(self, rec):
        self.junctions.append(rec)
        for chrom, pos in ((rec.chr1, rec.pos1), (rec.chr2, rec.pos2)):
            self.ends[chrom].addi(pos - self.buffer, pos + self.buffer + 1, rec)
        if rec.chr1 == rec.chr2:
            self.spans[rec.chr1].addi(rec.pos1, rec.pos2, rec)

    def near(self, chrom, pos):
        tree = self.ends.get(chrom)
        return {i.data for i in tree.at(pos)} if tree else set()

    def overlapping(self, chrom, begin, end):
        tree = self.spans.get(chrom)
        return {i.data for i in tree.overlap(begin, end)} if tree else set()

    def enveloped(self, chrom, begin, end):
        tree = self.spans.get(chrom)
        return {i.data for i in tree.envelop(begin, end)} if tree else set()

    def __iter__(self):
        return iter(self.junctions)

@dataclass
class FragmentOp:
    chrom: str
    start: int
    end: int
    operation: str
    target_chrom: str=None
    target_start: int=None
    target_end: int=None
    insord: int=None

    def __post_init__(self):
        # force ends to be at least one position after start
        self.end = max(self.end, self.start + 1)
        if self.target_start is not None and self.target_end is not None:
            self.target_end = max(self.target_end, self.target_start + 1)
