from groovi.seq.constants import *
from groovi.seq.constants import *
from groovi.seq.intervals import GenomeInterval
from groovi.seq.io import *

import numpy as np
from scipy.spatial import KDTree
from pysam import VariantFile

def get_intervals_from_VCF(vcf_path: str) -> list[GenomeInterval]:
    '''
    Convert a VCF file to a list of GenomeIntervals

    Expects the following INFO fields:
    - END: int, end position of the variant
    - END_CHROM: str, target chromosome of the variant
    - ORIENTATION: str, one of 'LR', 'RL', 'LL', '
    '''
    vcf_in = VariantFile(vcf_path)
    intervals = []
    for record in vcf_in.fetch():
        source_chrom = record.chrom
        start = record.start
        sv_class = record.id

        info = record.info
        end = info['END']
        target_chrom = info['END_CHROM']
        orientation = Laterality(info['ORIENTATION'])

        interval = GenomeInterval(source_chrom, target_chrom, start, end, sv_class=sv_class, orientation=orientation)
        intervals.append(interval)

    return intervals
    
'''
Structures for querying Groovi specific VCF Records
'''
class VCF_KDTree:
    def __init__(self, vcf_path: str):
        self.intervals = get_intervals_from_VCF(vcf_path)
        self.positions = np.array([interval.get_positions() for interval in self.intervals])
        self.kdtree = KDTree(self.positions)

    def query_top_label(self, chrom: str, chrom_target: str, query: np.ndarray, k: int = 5, expected_orientation = None, return_distance = False) -> list[VCFRecord]:
        # Query the KDTree for the top k records within a certain distance
        distances, indices = self.kdtree.query(query, k=k)
        # TODO ... other filters (OP type, head/tail, ...)
        # Filter by chromosome
        indices = [i for i in indices if self.intervals[i].source_chrom == chrom and self.intervals[i].target_chrom == chrom_target]

        # Filter by orientation
        if expected_orientation is not None:
            indices = [i for i in indices if self.intervals[i].orientation == expected_orientation]

        filtered_intervals = [self.intervals[i] for i in indices]

        top_sv_label = 'NEG'
        lowest_distance = float('inf')

        if len(filtered_intervals) > 0:
            top_sv_label = best_interval.sv_class
            best_interval = filtered_intervals[0]
            lowest_distance = distances[0]
        
        return (top_sv_label, lowest_distance) if return_distance else top_sv_label