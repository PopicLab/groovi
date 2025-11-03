from enum import Enum

################################################
# SV Classes
################################################

# SV Class Edge Types, one SV class per line. Comment out lines to remove classes
SV_CLASSES = [ 
    'NEG'
    # Simple SVs
    'DEL',
    'INV',
    'DUP',
    # Non dispersed CSVs, 2 Segments
    'dupINV-A', 'dupINV-B',
    'delINV-A', 'delINV-B',
    'INVdel-A', 'INVdel-B',
    'INVdup-A', 'INVdup-B',
    'DUP_INV-A', 'DUP_INV-B',
    # Non dispersed CSVs, 3 segments
    'delINVdel-A',
    'dupINVdup-A',
    'dupINVdel-A', 'dupINVdel-B',
    'delINVdup-A', 'delINVdup-B',
    # Non-Reciprocal dispersions
    'dDUP-A', 'dDUP-B',
    'nrTRA-A', 'nrTRA-B',
    'INV_dDUP-A', 'INV_dDUP-B',
    'rINV_DUP-A', 'rINV_DUP-B',
    'lINV_DUP-A', 'lINV_DUP-B',
    'INV_rTRA-A', 'INV_rTRA-B',
    'ldDUP_iDEL-A', 'ldDUP_iDEL-B',
    'rdDUP_iDEL-A', 'rdDUP_iDEL-B',
    'lINV_nrTRA-A', 'lINV_nrTRA-B', 'lINV_nrTRA-C',
    'rINV_nrTRA-A', 'rINV_nrTRA-B', 'rINV_nrTRA-C',
    'lINS_iDEL-A', 'lINS_iDEL-B', 'lINS_iDEL-C',
    'rINS_iDEL-A', 'rINS_iDEL-B', 'rINS_iDEL-C',
    'rdDUP_INV-A', 'rdDUP_INV-B', 'rdDUP_INV-C', 'rdDUP_INV-D',
    'ldDUP_INV-A', 'ldDUP_INV-B', 'ldDUP_INV-C', 'ldDUP_INV-D',
    'dDUP_INV-A', 'dDUP_INV-B', 'dDUP_INV-C', 'dDUP_INV-D',
    'INV_nrTRA-A', 'INV_nrTRA-B', 'INV_nrTRA-C',
    # Recipricoal dispersions
    'rTRA-A', 'rTRA-B',
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

NODE_TYPE_TO_ID = {n_type: i for i, n_type in enumerate(NodeType)}
ID_TO_NODE_TYPE = {i: n_type for n_type, i in NODE_TYPE_TO_ID.items()}

################################################
# Visualization Constants
################################################
EDGE_COLOR_MAP = {i: color for i, color in enumerate(['mediumpurple', 'pink', 'darkseagreen', 'lightblue', 'lightgray', 'yellow', 'pink'])}
