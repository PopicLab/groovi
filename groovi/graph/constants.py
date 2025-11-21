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
    HINT = 'HINT'

CANDIDATE_EDGE_FEATURE_TYPES = [EdgeFeature.LL, EdgeFeature.RR, EdgeFeature.LR, EdgeFeature.RL]
NORM_EDGE_FEATURE_TYPES = [EdgeFeature.LL, EdgeFeature.RR, EdgeFeature.LR, EdgeFeature.RL,
                           EdgeFeature.RD, EdgeFeature.RD_POS, EdgeFeature.RD_NEG]

# Nodes
class NodeFeature(Enum):
    RD_LOW = 'RD_LOW'
    RD = 'RD'
    RD_CLIPPED_L = 'RD_CLIPPED_L'
    RD_CLIPPED_R = 'RD_CLIPPED_R'
    INS = 'INS'


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
                         AlnStatistic.kIns: NodeFeature.INS}

EDGE_FEATURE_TO_ID = {e_type: i for i, e_type in enumerate(EdgeFeature)}
ID_TO_EDGE_FEATURE = {i: e_type for e_type, i in EDGE_FEATURE_TO_ID.items()}

NODE_FEATURE_TO_ID = {feature: i for i, feature in enumerate(NodeFeature)}

NODE_TYPE_TO_ID = {n_type: i for i, n_type in enumerate(NodeType)}
ID_TO_NODE_TYPE = {i: n_type for n_type, i in NODE_TYPE_TO_ID.items()}

# Graphs
GRAPH_CLASSES = ["CASE", "CONTROL"]

# -------- Node and edge features --------
# NodeFeature = groovi.utils.types.make_enum("NodeFeature",
#                                            [signal_name for signal_name in SVSignal.__members__] +
#                                            [signal_combo_name for signal_combo_name in SIGNAL_COMBINATIONS], __name__)
# EdgeFeature = groovi.utils.types.make_enum("EdgeFeature",
#                                            [signal_name for signal_name in SVSignal.__members__] +
#                                            [signal_combo_name for signal_combo_name in SIGNAL_COMBINATIONS], __name__)
# NODE_FEATURE_TO_SIGNAL = {NodeFeature[signal]: SVSignal[signal] for signal in SVSignal.__members__}
# EDGE_FEATURE_TO_SIGNAL = {EdgeFeature[signal]: SVSignal[signal] for signal in SVSignal.__members__}
# CHANNEL_TO_SIGNAL_COMBO = {Channel[channel_name]: SIGNAL_COMBINATIONS[channel_name]
#                            for channel_name in Channel.__members__ if channel_name not in SVSignal.__members__}
#
# # -------- Node/edge feature sets (e.g. features corresponding to a specific platform/regime) --------
# NodeFeatureSet = groovi.utils.types.make_enum("EdgeFeatureSet",["LONG"], __name__)
# EdgeFeatureSet = groovi.utils.types.make_enum("EdgeFeatureSet",["LONG"], __name__)
#
# NODE_FEATURE_SETS = {
#     EdgeFeatureSet.LONG:  [NodeFeature.RD, NodeFeature.RD_CLIPPED_L, NodeFeature.RD_CLIPPED_R, NodeFeature.CINS],
# }
#
# EDGE_FEATURE_SETS = {
#     EdgeFeatureSet.LONG:  [EdgeFeature.SRORD12_CDEL, EdgeFeature.SRORD21_SFORD21, EdgeFeature.SRFLIP_SFFLIP],
# }
#
