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

GRAPH_CLASSES = ["CASE", "CONTROL"]

SV_CLASS_TO_ID_HELPER = {sv_class: i for i, sv_class in enumerate(SV_CLASSES)}
SV_EDGE_CLASS_TO_ID_FN = lambda sv_class: SV_CLASS_TO_ID_HELPER.get(sv_class, 0)  # default to 'NEG' if not found, used for background custom CSVs
SV_GRAPH_CLASS_TO_ID_FN =  lambda sv_class: SV_CLASS_TO_ID_HELPER.get(sv_class, 0)

################################################
# Global Mapping Exports
################################################
#
# EDGE_FEATURE_TO_ID = {e_type: i for i, e_type in enumerate(EdgeFeature)}
# NODE_FEATURE_TO_ID = {feature: i for i, feature in enumerate(NodeFeature)}
#
# NODE_TYPE_TO_ID = {n_type: i for i, n_type in enumerate(NodeType)}
# ID_TO_NODE_TYPE = {i: n_type for n_type, i in NODE_TYPE_TO_ID.items()}

################################################
# Visualization Constants
################################################
EDGE_COLOR_MAP = {i: color for i, color in enumerate(['mediumpurple', 'pink', 'darkseagreen', 'lightblue', 'lightgray', 'yellow', 'pink'])}
