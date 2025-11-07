from groovi.seq.constants import *
from groovi.seq.constants import *
from groovi.seq.intervals import GenomeInterval
from groovi.seq.io import *

import numpy as np
from pysam import VariantFile
from scipy.spatial import KDTree
from typing import Union

###############################################################
# Structures for record keeping
###############################################################

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
        end = record.stop
        target_chrom = info.get('END_CHROM', source_chrom) if 'END_CHROM' in info else source_chrom
        orientation = ReadAlignmentOrientation(info['ORIENTATION'].value) if 'ORIENTATION' in info else None

        interval = GenomeInterval(source_chrom, target_chrom, start, end, sv_class=sv_class, orientation=orientation)
        intervals.append(interval)

    return intervals
    
###############################################################
# Structures for querying
###############################################################

class VCF_KDTree:
    """
    Datastructure for efficient top-1 ofg top-k interval querying by L2 distance of (start, end) points.

    Includes ability to filter by split alignment orientation by
    - maximum distance
    - chromosome
    - split alignment orientation
    """
    def __init__(self, vcf_path: str):
        self.intervals: list[GenomeInterval] = get_intervals_from_VCF(vcf_path)
        self.positions = np.array([[interval.get_start_pos(), interval.get_end_pos()] for interval in self.intervals])
        self.kdtree = KDTree(self.positions)

    def query_top_label(self, query: np.ndarray, start_chrom: str, end_chrom: str, k: int = 5, orientation = None,
                        return_distance = False, max_distance = 100) -> Union[tuple[str, int], str]:
        # Query the KDTree for the top k records within a certain distance
        distances, indices = self.kdtree.query(query, k=k)
        # TODO ... other filters (OP type, head/tail, ...)
        # Filter by chromosome
        indices = [i for i in indices if self.intervals[i].start_chrom == start_chrom and self.intervals[i].end_chrom == end_chrom]

        # filter by max distance
        indices = [i for i, d in zip(indices, distances) if d <= max_distance]

        # Filter by orientation
        if orientation is not None:
            indices = [i for i in indices if self.intervals[i].orientation == orientation]

        filtered_intervals = [self.intervals[i] for i in indices]

        top_sv_label = 'NEG'
        lowest_distance = float('inf')

        if len(filtered_intervals) > 0:
            top_sv_label = best_interval.sv_class
            best_interval = filtered_intervals[0]
            lowest_distance = distances[0]
        
        return (top_sv_label, lowest_distance) if return_distance else top_sv_label