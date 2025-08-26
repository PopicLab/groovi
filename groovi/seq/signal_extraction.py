# Copyright (c) 2023-2025 Victoria Popic
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

from groovi.seq.constants import *

from collections import OrderedDict
from pyfaidx import Fasta

class Edge:
    def __init__(self, chrom1: str, chrom2: str, pos1: int, pos2: int,
                 source: JunctionSource, edge_type: EdgeRelation, orientation, evidence: int):
        self.chrom1 = chrom1
        self.chrom2 = chrom2
        self.pos1 = pos1
        self.pos2 = pos2
        self.source = source
        self.edge_type = edge_type
        self.orientation = orientation
        self.evidence = evidence
    
    def __repr__(self):
        return f"Edge({self.chrom1}, {self.chrom2}, {self.pos1}, {self.pos2}, {self.source}, {self.edge_type}, {self.orientation}, {self.evidence})"

class SignalExtractor:
    def __init__(self, config):
        self.config = config
        self.chrom_lengths = self._populate_chrom_lengths()
        # TODO: change to bam
        if hasattr(config, 'adj'):
            pass

    def get_adj(self) -> list[Edge]:
        if hasattr(self.config, 'adj') and self.config.adj is not None:
            return self._get_adj_from_file()
        elif hasattr(self.config, 'bam') and self.config.bam is not None:
            return self._get_adj_from_pycue()
        else:
            raise ValueError("No adjacency source provided in the configuration.")     

    def range_query(self, chrom: str, start: int, end: int):
        # Placeholder for range query implementation
        raise NotImplementedError("Range query is not implemented yet.") 

    def get_chr_names(self) -> list[str]:
        return list(self.chrom_lengths.keys())
    
    def get_chr_lengths(self) -> OrderedDict[str, int]:
        return self.chrom_lengths
    
    def _chr_str_to_chr_id(self, chrom: str) -> int:
        """
        Converts a chromosome string (e.g., 'chr1', 'X') to an integer without using try-except.
        """
        # Remove 'chr' prefix if present
        if chrom.startswith('chr'):
            chrom = chrom[3:]

        # Use a dictionary for special cases
        special_chroms = {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}
        
        if chrom in special_chroms:
            return special_chroms[chrom]
        elif chrom.isdigit():
            return int(chrom)
        else:
            raise ValueError(f"Unrecognized chromosome: {chrom}")


    def _populate_chrom_lengths(self) -> OrderedDict[str, int]:
        chr_lengths = OrderedDict()
        genes = Fasta(self.config.fa)
        for record_name in genes.keys():
            record = genes[record_name]
            chr_id = self._chr_str_to_chr_id(record.name)
            chr_lengths[chr_id] = len(record)
        return chr_lengths
       
    def _get_adj_from_file(self) -> list[Edge]:
        edges = []
        with open(self.config.adj, 'r') as f:
            for line in f:
                if line.startswith("Junction:"):
                    parts = line.split()
                    data_parts = parts[1:]

                    chrom1 = self._chr_str_to_chr_id(data_parts[0])
                    chrom2 = self._chr_str_to_chr_id(data_parts[1])
                    pos1 = int(data_parts[2])
                    pos2 = int(data_parts[3])

                    source_str = data_parts[4].split('.')[-1]
                    orientation_str = data_parts[5].split('.')[-1]
                    order_str = data_parts[6].split('.')[-1]

                    source = JunctionSource(source_str)
                    segment_orientation = SegmentOrientation(orientation_str)
                    order = SegmentOrder(order_str)

                    edge_type, orientation = None, None
                    if order == SegmentOrder.kOrd12 and segment_orientation == SegmentOrientation.kStrandMatch:
                        edge_type = EdgeRelation.ORD12
                        orientation = Laterality.LR
                    elif order == SegmentOrder.kOrd21 and segment_orientation == SegmentOrientation.kStrandMatch:
                        edge_type = EdgeRelation.ORD21
                        orientation = Laterality.RL
                    elif segment_orientation == SegmentOrientation.kStrandFlipRR:
                        edge_type = EdgeRelation.STRAND_FLIP
                        orientation = Laterality.RR
                    elif segment_orientation == SegmentOrientation.kStrandFlipLL:
                        edge_type = EdgeRelation.STRAND_FLIP
                        orientation = Laterality.LL

                    evidence = int(data_parts[7])

                    edges.append(Edge(chrom1, chrom2, pos1, pos2, source, edge_type, orientation, evidence))
                   

        return edges

    def _get_adj_from_pycue(self) -> list[Edge]:
        raise NotImplementedError("Signal extraction from BAM is not implemented yet.")
