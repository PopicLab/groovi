from groovi.seq.constants import *
from groovi.seq.io import ChrFAIndex

from collections import OrderedDict

class Laterality(Enum):
    LR = 'LR'
    RL = 'RL'
    LL = 'LL'
    RR = 'RR'

class Edge:
    def __init__(self, chrom1: str, chrom2: str, pos1: int, pos2: int,
                 source: JunctionSource, edge_type: EdgeFeature, orientation: Laterality, evidence: int):
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
    
    def get_edge_type(self) -> EdgeFeature:
        return self.edge_type

class SignalExtractor:
    def __init__(self, config):
        self.config = config
        self.fai_index: ChrFAIndex = ChrFAIndex.load(config.fai, all=True)

    def get_edges(self) -> list[Edge]:
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
        return self.fai_index.get_chr_names()

    def get_chr_lengths(self) -> OrderedDict[int, int]:
        return self.fai_index.get_chr_lengths()

    def _get_adj_from_file(self) -> list[Edge]:
        edges = []
        with open(self.config.adj, 'r') as f:
            for line in f:
                if line.startswith("Junction:"):
                    parts = line.split()
                    data_parts = parts[1:]

                    chrom1 = self.fai_index.get_tid_by_name(data_parts[0])
                    chrom2 = self.fai_index.get_tid_by_name(data_parts[1])
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
                        edge_type = EdgeFeature.ORD12
                        orientation = Laterality.LR
                    elif order == SegmentOrder.kOrd21 and segment_orientation == SegmentOrientation.kStrandMatch:
                        edge_type = EdgeFeature.ORD21
                        orientation = Laterality.RL
                    elif segment_orientation == SegmentOrientation.kStrandFlipRR:
                        edge_type = EdgeFeature.STRAND_FLIP
                        orientation = Laterality.RR
                    elif segment_orientation == SegmentOrientation.kStrandFlipLL:
                        edge_type = EdgeFeature.STRAND_FLIP
                        orientation = Laterality.LL

                    evidence = int(data_parts[7])

                    edges.append(Edge(chrom1, chrom2, pos1, pos2, source, edge_type, orientation, evidence))
                   
        return edges

    def _get_adj_from_pycue(self) -> list[Edge]:
        raise NotImplementedError("Signal extraction from BAM is not implemented yet.")
