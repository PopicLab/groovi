import groovi.utils.types
from pycues import *
from enum import Enum

FEAT_TYPE = "feat_type"
FEAT = "feat"

################################################
# Node Type Enums
################################################

class NodeType(Enum):
    HEAD = 'HEAD'
    TAIL = 'TAIL'

################################################
# Relational Graph Enums
################################################

# Edges
# TODO: consider feature subsets
class EdgeFeature(Enum):
    LR = 'LR'
    RL = 'RL'
    LL = 'LL'
    RR = 'RR'
    REF = 'REF'
    SEG = 'SEG'
    IS_TINY_SEG = 'IS_TINY_SEG'
    RD = 'PD'
    RD_POS = 'RD_POS'
    RD_NEG = 'RD_NEG'

CANDIDATE_EDGE_FEATURE_TYPES = [EdgeFeature.LL, EdgeFeature.RR, EdgeFeature.LR, EdgeFeature.RL]
BINARY_EDGE_FEATURE_TYPES = [EdgeFeature.REF, EdgeFeature.SEG, EdgeFeature.IS_TINY_SEG]

# Nodes
class NodeFeature(Enum):
    RD_LOW = 'RD_LOW'
    RD = 'RD'
    RD_CLIPPED_L = 'RD_CLIPPED_L'
    RD_CLIPPED_R = 'RD_CLIPPED_R'
    INS = 'INS'
    DIV = 'DIV'
    HT = 'HT'

AUX_NODE_FEATURES = [NodeFeature.HT]  # auxiliary features not derived from alignment
BINARY_NODE_FEATURE_TYPES = [NodeFeature.HT]


ORIENTATION_TO_NODE_TYPES = {JOrientation.kLL: [NodeType.TAIL, NodeType.TAIL],
                             JOrientation.kRR: [NodeType.HEAD, NodeType.HEAD],
                             JOrientation.kLR: [NodeType.TAIL, NodeType.HEAD],
                             JOrientation.kRL: [NodeType.HEAD, NodeType.TAIL],}

ORIENTATION_TO_EDGE_TYPE =  {JOrientation.kLL: EdgeFeature.LL,
                             JOrientation.kRR: EdgeFeature.RR,
                             JOrientation.kLR: EdgeFeature.LR,
                             JOrientation.kRL: EdgeFeature.RL}

NODE_TYPES_TO_ORIENTATION = {(NodeType.TAIL, NodeType.TAIL): "LL",
                             (NodeType.HEAD, NodeType.HEAD): "RR",
                             (NodeType.TAIL, NodeType.HEAD): "LR",
                             (NodeType.HEAD, NodeType.TAIL): "RL"}

ALNSTAT_TO_NODE_FEAT =  {AlnStatistic.kRD: NodeFeature.RD,
                         AlnStatistic.kRDAll: NodeFeature.RD_LOW,
                         AlnStatistic.kClipLeft: NodeFeature.RD_CLIPPED_L,
                         AlnStatistic.kClipRight: NodeFeature.RD_CLIPPED_R,
                         AlnStatistic.kIns: NodeFeature.INS,
                         AlnStatistic.kDiv: NodeFeature.DIV}
NODE_FEAT_TO_ALNSTAT = {v: k for k, v in ALNSTAT_TO_NODE_FEAT.items()}

NodeFeatureSet = groovi.utils.types.make_enum("NodeFeatureSet",["A", "B"], __name__)
EdgeFeatureSet = groovi.utils.types.make_enum("EdgeFeatureSet",["A"], __name__)
NODE_FEATURE_SETS = {
    NodeFeatureSet.A:  [NodeFeature.RD_LOW, NodeFeature.RD,
                        NodeFeature.RD_CLIPPED_L, NodeFeature.RD_CLIPPED_R,
                        NodeFeature.INS],
    NodeFeatureSet.B:  [NodeFeature.RD_LOW, NodeFeature.RD,
                        NodeFeature.RD_CLIPPED_L, NodeFeature.RD_CLIPPED_R,
                        NodeFeature.INS, NodeFeature.HT],
}

EDGE_FEATURE_SETS = {
    EdgeFeatureSet.A:  [EdgeFeature.LR, EdgeFeature.RL, EdgeFeature.LL, EdgeFeature.RR,
                        EdgeFeature.REF, EdgeFeature.SEG, EdgeFeature.IS_TINY_SEG],
}
