from intervaltree import IntervalTree

from groovi.seq.sv import Fragment

def approx_equal(a, b, threshold=0):
    return abs(a - b) <= threshold


def identity_match(trees, tolerance=0):
    """
    Default function to extract data from first IntervalTree
    :param trees: list of trees. For this function, should only be one tree
    :param tolerance: Unused
    :return: list of records stored in tree values
    """
    return [[i.data] for i in trees[0]]


def interlocking(left_tree, right_tree, tolerance=0, margin=0):
    """
    Finds the set of pairs where the left and right intervals interlock in order
    """
    interlocking_pairs = list()
    for i_left in left_tree:
        search_begin = i_left.begin - tolerance
        search_end = i_left.end + tolerance

        candidates = right_tree.overlap(search_begin, search_end)

        for i_right in candidates:
            if i_right is i_left:  # skip self-matches
                continue
            # Check if the left interval starts before the right, plus a margin of error.
            if i_left.begin + margin <= i_right.begin + tolerance and i_left.end + margin <= i_right.end + tolerance:
                interlocking_pairs.append((i_left.data, i_right.data))

    return interlocking_pairs


def left_right_adjacent(pairs, tolerance=0, side='either'):
    """
    Filters a set of interval pairs to entries where the left and/or right edges are within 1 position (within tolerance)
    :param pairs: iterable of interval pairs
    :param tolerance: Maximum distance positions can be to be considered adjacent
    :param side: 'start', 'end', 'both', or 'either'. 'start' and 'end' return cases where the start or ends are adjacent,
                'both' returns cases where both start and end are adjacent. 'either' returns when either start or end are adjacent
    :return: list of interval pairs fitting adjacency criteria
    """
    adjacent_pairs = []

    start = side in ['start', 'either']
    end = side in ['end', 'either']
    both = side == 'both'

    for i_left, i_right in pairs:
        left_adj = approx_equal(abs(i_right.pos1 - i_left.pos1), 1, tolerance)
        right_adj = approx_equal(abs(i_right.pos2 - i_left.pos2), 1, tolerance)
        if both and left_adj and right_adj or \
            start and left_adj or \
            end and right_adj:
            adjacent_pairs.append((i_left, i_right))
    return adjacent_pairs


def interlocking_adjacent(left_tree, right_tree, tolerance=0, side='either'):
    """
    Finds the set of pairs where left and right intervals interlock and the starts or ends are adjacent
    """
    interlocking_pairs = interlocking(left_tree, right_tree, tolerance)
    adjacent_pairs = left_right_adjacent(interlocking_pairs, tolerance, side)
    return adjacent_pairs


def match_ddup_inv(interval_trees, tolerance=0):
    """
    dDUP_INV has four record types A, B, C, D with A and B interlocking adjacent on both sides and C enveloping D,
    adjacent at the target. B and C share the same start, and A's end and D's start are shared. The CD adjacency
    (the target) can be before or after the AB region.
    """
    matches = []

    ab_pairs = interlocking_adjacent(interval_trees[0], interval_trees[1], tolerance=tolerance, side='both')
    cd_pairs = envelop_adjacent(interval_trees[2], interval_trees[3], tolerance)

    cd_tree = IntervalTree.from_tuples([(min(c.pos1, d.pos1), max(c.pos2, d.pos2), (c, d)) for c, d in cd_pairs])

    for a, b in ab_pairs:
        candidates = cd_tree.overlap(min(a.pos1, b.pos1) - tolerance, max(a.pos2, b.pos2) + tolerance)
        for cd in candidates:
            c, d = cd.data
            if approx_equal(b.pos1, c.pos1, tolerance) and c.pos2 > b.pos2 or \
                approx_equal(a.pos2, c.pos2) and c.pos1 < a.pos1:
                matches.append([a, b, c, d])

    return matches


def match_nrtra(a_tree, b_tree, c_tree, tolerance=0):
    """
    Matches the nrTRA structure, which has two interlocking nrTRA-A intervals that overlap by one, and a nrTRA-B
    interval that is enclosed by and adjacent to the outer boundaries of the two nrTRA-A intervals.
    """

    interlocking_pairs = interlocking(a_tree, c_tree, tolerance)
    relevant_pairs = [(left, right) for left, right in interlocking_pairs
                      if approx_equal(left.pos2 - right.pos1, 1, tolerance)]

    matches = []

    for left, right in relevant_pairs:
        candidates = b_tree.envelop(left.pos1 - tolerance, right.pos2 + tolerance)

        for b in candidates:
            if approx_equal(left.pos1 + 1, b.data.pos1, tolerance) and approx_equal(right.pos2, b.data.pos2 + 1, tolerance):
                matches.append([left, b.data, right])

    return matches


def interlocking_intersection(left_tree, middle_tree, right_tree, tolerance=0):
    """
    Finds the set of intervals where the left and right interval interlock, and the middle interval is the intersection
    """

    interlocking_pairs = interlocking(left_tree, right_tree, tolerance)

    valid_triples = []

    for i_left, i_right in interlocking_pairs:

        candidates = middle_tree.overlap(i_right.pos1 - tolerance, i_left.pos2 + tolerance)

        for i_middle in candidates:
            if approx_equal(i_middle.begin, i_right.pos1, tolerance) and approx_equal(i_middle.end, i_left.pos2, tolerance):
                valid_triples.append((i_left, i_middle.data, i_right))
    return valid_triples


def envelop(left_tree, right_tree, tolerance=0):
    """
    Finds the pairs where intervals from the right tree are enveloped by intervals in the left tree
    """
    valid_pairs = []
    for i_left in left_tree:
        matches = right_tree.envelop(i_left.begin - tolerance, i_left.end + tolerance)
        valid_pairs += [(i_left.data, i_right.data) for i_right in matches]

    return valid_pairs


def envelop_adjacent(left_tree, right_tree, tolerance=0, side='either'):
    """
    Finds the pairs of intervals where the left interval envelops the right interval and
    the one set of endpoints are adjacent to each other
    """
    valid_pairs = envelop(left_tree, right_tree, tolerance)
    return left_right_adjacent(valid_pairs, tolerance, side)


def match_inv_nrtra(trees, tolerance=0):
    """
    Seeks triples where B and C intervals are envelop adjacent, and A surrounds the non-adjacent endpoints of B and C
    """
    matches = []

    bc_pairs = envelop_adjacent(trees[1], trees[2], tolerance)

    for b, c in bc_pairs:
        end_dist = abs(b.pos2 - c.pos2)
        start_dist = abs(b.pos1 - c.pos1)
        if start_dist <= end_dist:
            target_interval = (c.pos2, b.pos2)
        else:
            target_interval = (b.pos1, c.pos1)

        candidates = trees[0].overlap(target_interval[0], target_interval[1])

        for a_interval in candidates:
            a = a_interval.data
            if approx_equal(target_interval[0] - 1, a.pos1, tolerance) and approx_equal(target_interval[1] + 1, a.pos2, tolerance):
                matches.append([a, b, c])
    return matches

def dispersion_source_target(csv):
    """
    Find the source and target of dispersion CSVs.
    """

    if csv.sv_type in ['dDUP', 'INV_dDUP']:
        # look for two endpoints that are close together (off by exactly one if exact). The earlier is the target
        start_dist = abs(csv.records[0].pos1 - csv.records[1].pos1)
        stop_dist = abs(csv.records[0].pos2 - csv.records[1].pos2)

        if start_dist < stop_dist:
            source_start, source_end = sorted([csv.records[0].pos2, csv.records[1].pos2])
            target = min(csv.records[0].pos1, csv.records[1].pos1)
            target_chrom = csv.records[0].chr1
        else:
            source_start, source_end = sorted([csv.records[0].pos1, csv.records[1].pos1])
            target = min(csv.records[0].pos2, csv.records[1].pos2)
            target_chrom = csv.records[0].chr2
    elif csv.sv_type in ['dDUP_INV']:
        # ABCD edges. Source is always defined by edge B. Target is the endpoint of C not matching the end of B
        source_start = csv.records[1].pos1
        source_end = csv.records[1].pos2 - 1
        if abs(csv.records[1].pos1 - csv.records[2].pos1) < abs(csv.records[0].pos2 - csv.records[2].pos2):
            # dispersion goes to the right
            target = csv.records[3].pos2
            target_chrom = csv.records[3].chr2
        else:
            target = csv.records[2].pos1
            target_chrom = csv.records[2].chr1
    elif csv.sv_type in ['nrTRA']:
        # nrTRA graphs are completely symmetrical in structure, so we can use the heuristic that the source is the
        # shorter of the possible sources
        candidates = [(csv.records[1].pos1, csv.records[2].pos1, csv.records[1].pos2, csv.records[1].chr2),
                      (csv.records[0].pos2, csv.records[1].pos2, csv.records[0].pos1, csv.records[0].chr1)]
        source_start, source_end, target, target_chrom = min(candidates, key=lambda t: t[1] - t[0])
    elif csv.sv_type in ['INV_nrTRA']:
        # Since INV_nrTRA has an explicit C edge, the source will always be the portion of the B edge
        # not covered by C, and the target is on the other side of C
        if abs(csv.records[0].pos1 - csv.records[1].pos1) < abs(csv.records[0].pos2 - csv.records[1].pos2):
            # target is to the left
            source_start = csv.records[1].pos1
            source_end = csv.records[2].pos1
            target = csv.records[2].pos2
            target_chrom = csv.records[2].chr2
        else:
            source_start = csv.records[2].pos2
            source_end = csv.records[1].pos2
            target = csv.records[1].pos1
            target_chrom = csv.records[1].chr1

    return source_start, source_end, target, target_chrom

"""
    Fields and default values:
    'op_name': Replacement SV name for BED output (SV name)
    'fragment_fn': function to compute fragments (None)
    'start_stop_fn': function to compute start and stop of VCF entry (None)
"""
SV_REGISTRY = {
    'DEL': {
        'grammar': 'A->',
        'candidates': ['DEL'],
        'op_name': 'CUT',
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[0].pos2 - 1),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records[0].chr1, start=csv.records[0].pos1 + 1, end=csv.records[0].pos2 - 1, operation='CUT'),
        ]
    },
    'INV': {
        'grammar': 'A->a',
        'candidates': ['INV_A', 'INV_B'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'both'),
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[0].pos2),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records[0].chr1, start=csv.records[0].pos1 + 1, end=csv.records[0].pos2, operation='INV'),
        ]
    },
    'DUP': {
        'grammar': 'A->AA',
        'candidates': ['DUP'],
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
        'start_stop_fn': lambda csv: (csv.records[0].pos1, csv.records[0].pos2),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records[0].chr1, start=csv.records[0].pos1, end=csv.records[0].pos2, operation='DUP'),
        ]
    },
    'DUP_INV': {
        'grammar': 'A->aa',
        'candidates': ['DUP_INV_A', 'DUP_INV_B', 'DUP_INV_C'],
        'match_fn': lambda x, tolerance: interlocking_intersection(x[0], x[1], x[2], tolerance),
        'start_stop_fn': lambda csv: (csv.records[1].pos1, csv.records[1].pos2),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records[1].chr1, start=csv.records[1].pos1, end=csv.records[1].pos2, operation='INV'),
            Fragment(chrom=csv.records[1].chr1, start=csv.records[1].pos1, end=csv.records[1].pos2,
                     target_chrom=csv.records[1].chr2, target_start=csv.records[1].pos2, target_end=csv.records[1].pos2 + 1,
                     operation='COPYinv-PASTE'),
        ],
    },
    'dDUP': {
        'grammar': ('A_->A_A', '_A->A_A'),
        'op_name': 'DUP',
        'dispersion': True,
        'candidates': [('dDUP_A', 'dDUP_C'), ('dDUP_B', 'dDUP_D')],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance) + interlocking_adjacent(x[1], x[0], tolerance),
        'start_stop_fn': lambda csv: (csv.source_start, csv.source_end),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end,
                     target_chrom=csv.target_chrom, target_start=csv.target, target_end=csv.target + 1,
                     operation='COPY-PASTE')
        ]
    },
    'nrTRA': {
        'grammar': ('A_->_A', '_A->A_'),
        'candidates': ['nrTRA_A', 'nrTRA_B', 'nrTRA_C'],
        'match_fn': lambda x, tolerance: match_nrtra(x[0], x[1], x[2], tolerance),
        'dispersion': True,
        'start_stop_fn': lambda csv: (csv.source_start, csv.source_end),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end,
                     target_chrom=csv.target_chrom, target_start=csv.target, target_end=csv.target + 1,
                     operation='CUT-PASTE')
        ]
    },
    'INV_nrTRA': {
        'grammar': ('A_->_a',  '_A->a_'),
        'candidates': [
            ('INV_nrTRA_A', 'INV_nrTRA_F'),
            ('INV_nrTRA_B', 'INV_nrTRA_D'),
            ('INV_nrTRA_C', 'INV_nrTRA_E'),
        ],
        'match_fn': match_inv_nrtra,
        'dispersion': True,
        'start_stop_fn': lambda csv: (csv.source_start, csv.source_end),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end,
                     target_chrom=csv.target_chrom, target_start=csv.target, target_end=csv.target + 1,
                     operation='CUTinv-PASTE')
        ]
    },
    'INV_dDUP': {
        'grammar': ('A_->A_a', '_A->a_A'),
        'candidates': ['INV_dDUP_A', 'INV_dDUP_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance),
        'dispersion': True,
        'start_stop_fn': lambda csv: (csv.source_start, csv.source_end),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end,
                     target_chrom=csv.target_chrom, target_start=csv.target, target_end=csv.target + 1,
                     operation='COPYinv-PASTE')
        ]
    },
    'dDUP_INV': {
        'grammar': ('A_->a_a', '_A->a_a'),
        'candidates': ['dDUP_INV_A', 'dDUP_INV_B', 'dDUP_INV_C', 'dDUP_INV_D'],
        'match_fn': match_ddup_inv,
        'dispersion': True,
        'start_stop_fn': lambda csv: (csv.source_start, csv.source_end),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end,
                     target_chrom=csv.target_chrom, target_start=csv.target, target_end=csv.target + 1,
                     operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.source_start, end=csv.source_end, operation='INV')
        ]
    },
    'dupINVdel': {
        'grammar': 'ABC->Aba',
        'candidates': ['dupINVdel_A', 'dupINVdel_B'],
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'start_stop_fn': lambda csv: (csv.records[0].pos1, csv.records[0].pos2 - 1),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1, end=csv.records[1].pos1,
                     target_chrom=csv.records[0].chr2, target_start=csv.records[1].pos2, target_end=csv.records[1].pos2 + 1,
                     operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1 + 1, end=csv.records[1].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos2 + 1, end=csv.records[0].pos2 - 1, operation='CUT'),
        ],
    },
    'delINVdup': {
        'grammar': 'ABC->cbC',
        'candidates': ['delINVdup_A', 'delINVdup_B'],
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[0].pos2),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1 + 1, end=csv.records[1].pos1 - 1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1, end=csv.records[1].pos2 - 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos2, end=csv.records[0].pos2,
                     target_chrom=csv.records[0].chr2, target_start=csv.records[0].pos1,
                     target_end=csv.records[0].pos1 + 1, operation='COPYinv-PASTE'),
        ]
    },
    'delINVdel': {
        'grammar': 'ABC->b',
        'candidates': ['delINVdel_A', 'delINVdel_B'],
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance, margin=1),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1 + 1, end=csv.records[1].pos1 - 1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1, end=csv.records[0].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[0].pos2 + 1, end=csv.records[1].pos2 - 1, operation='CUT')
        ],
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[1].pos2 - 1),
        'is_self_matching': True,
    },
    'dupINVdup': {
        'grammar': 'ABC->AcbaC',
        'candidates': ['dupINVdup_A', 'dupINVdup_B'],
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1, end=csv.records[1].pos1,
                     target_chrom=csv.records[0].chr2, target_start=csv.records[0].pos2 - 1,
                     target_end=csv.records[0].pos2, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1 + 1, end=csv.records[0].pos2 - 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[0].pos2, end=csv.records[1].pos2,
                     target_chrom=csv.records[1].chr1, target_start=csv.records[1].pos1, target_end=csv.records[1].pos1 + 1,
                     operation='COPYinv-PASTE')
        ],
        'start_stop_fn': lambda csv: (csv.records[0].pos1, csv.records[1].pos2),
        'is_self_matching': True,
    },
    'delINV': {
        'grammar': 'AB->b',
        'candidates': ['delINV_A', 'delINV_B'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'end'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1 + 1, end=csv.records[1].pos1 - 1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1, end=csv.records[0].pos2, operation='INV')
        ],
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[1].pos2 - 1),
    },
    'INVdel': {
        'grammar': 'AB->a',
        'candidates': ['INVdel_A', 'INVdel_B'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'start'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[1].pos1, end=csv.records[0].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[0].pos2 + 1, end=csv.records[1].pos2 - 1, operation='CUT')
        ],
        'start_stop_fn': lambda csv: (csv.records[0].pos1 + 1, csv.records[1].pos2 - 1),
    },
    'dupINV': {
        'grammar': 'AB->Aba',
        'candidates': ['dupINV_A', 'dupINV_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='end'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records[0].pos1, end=csv.records[1].pos1,
                     target_chrom=csv.records[1].chr2, target_start=csv.records[1].pos2,
                     target_end=csv.records[1].pos2 + 1, operation='COPYinv-PASTE'),
            # Fragment(chrom=csv.chrom, start=csv.records[1].pos1 + 1, end=csv.records[1].pos2, operation='INV'),
        ],
        'start_stop_fn': lambda csv: (csv.records[0].pos1, csv.records[1].pos2),
    },
    'INVdup': {
        'grammar': 'AB->baB',
        'candidates': ['INVdup_A', 'INVdup_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='start'),
        'fragment_fn': lambda csv: [
            # Fragment(chrom=csv.chrom, start=csv.records[1].pos1, end=csv.records[1].pos2 - 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records[1].pos2, end=csv.records[0].pos2,
                     target_chrom=csv.chrom, target_start=csv.records[1].pos1 - 1, target_end=csv.records[1].pos1,
                     operation='COPYinv-PASTE'),
        ],
        'start_stop_fn': lambda csv: (csv.records[1].pos1, csv.records[0].pos2),
    }
}
