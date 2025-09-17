from enum import Enum

################################################
# SV Classes
################################################

SV_CLASSES = [
    'NEG',
    'DEL',
    'DUP',
    'INS',
    'INV',
    'dDUP-A', 'dDUP-B',
    'nrTRA-A', 'nrTRA-B',
]

SV_CLASS_TO_ID_HELPER = {sv_class: i for i, sv_class in enumerate(SV_CLASSES)}
SV_CLASS_TO_ID_FN = lambda sv_class: SV_CLASS_TO_ID_HELPER.get(sv_class, 0)  # default to 'NEG' if not found, used for background custom CSVs 

################################################
# Graph Construction Enums
################################################

class Laterality(Enum):
    LR = 'LR'
    RL = 'RL'
    LL = 'LL'
    RR = 'RR'

class SegmentOrientation(Enum):
    kStrandMatch = "kStrandMatch"
    kStrandFlipLL = "kStrandFlipLL"
    kStrandFlipRR = "kStrandFlipRR"

class SegmentOrder(Enum):
    kOrd12 = "kOrd12"
    kOrd21 = "kOrd21"

class JunctionSource(Enum):
    """
    Represents the source of the junction.
    """
    kSplitRead = "kSplitRead"
    kCigar = "kCigar"

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
    ORD12 = 'ORD12',
    ORD21 = 'ORD21',
    STRAND_FLIP = 'STRAND_FLIP',
    REF = 'REF',
    SEG = 'SEG',
    PD = 'PD',

# Nodes
class NodeFeature(Enum):
    RD_LOW = 'RD_LOW',
    RD = 'RD',
    RD_CLIPPED_L = 'RD_CLIPPED_L',
    RD_CLIPPED_R = 'RD_CLIPPED_R',
    IN = 'IN'  # fixed target mode

################################################
# Global Mapping Exports
################################################

EDGE_FEATURE_TO_ID = {e_type: i for i, e_type in enumerate(EdgeFeature)}
NODE_FEATURE_TO_ID = {feature: i for i, feature in enumerate(NodeFeature)}

################################################
# Visualization Constants
################################################
EDGE_COLOR_MAP = {i: color for i, color in enumerate(['mediumpurple', 'pink', 'darkseagreen', 'lightblue', 'lightgray', 'yellow', 'pink'])}
