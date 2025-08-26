from enum import Enum

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
# Relational Graph Enums
################################################

# Edges
class EdgeRelation(Enum):
    ORD12 = 'ORD12',
    ORD21 = 'ORD21',
    STRAND_FLIP = 'STRAND_FLIP',
    REF = 'REF',
    SEG = 'SEG',
    PD = 'PD',
    SUPER = 'SUPER'

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

EDGE_RELATION_MAP = {e_type: i for i, e_type in enumerate(EdgeRelation)}
NODE_FEATURE_MAP = {feature: i for i, feature in enumerate(NodeFeature)}

################################################
# Visualization Constants
################################################
EDGE_COLOR_MAP = {i: color for i, color in enumerate(['mediumpurple', 'pink', 'darkseagreen', 'lightblue', 'lightgray', 'yellow', 'pink'])}
