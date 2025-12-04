from intervaltree import IntervalTree


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


def left_right_adjacent(pairs, tolerance=0, side='both'):
    adjacent_pairs = []

    start = side in ['start', 'both']
    end = side in ['end', 'both']

    for i_left, i_right in pairs:
        if start and approx_equal(abs(i_right.pos1 - i_left.pos1), 1, tolerance) or end and approx_equal(abs(i_right.pos2 - i_left.pos2), 1, tolerance):
            adjacent_pairs.append((i_left, i_right))
    return adjacent_pairs


def interlocking_adjacent(left_tree, right_tree, tolerance=0, side='both'):
    """
    Finds the set of pairs where left and right intervals interlock and the starts or ends are adjacent
    """
    interlocking_pairs = interlocking(left_tree, right_tree, tolerance)
    interlocking_pairs += [(x[1], x[0]) for x in interlocking(right_tree, left_tree, tolerance)]
    adjacent_pairs = left_right_adjacent(interlocking_pairs, tolerance, side)
    return adjacent_pairs


def match_ddup_inv(interval_trees, tolerance=0):
    """
    dDUP_INV has four record types A, B, C, D with A and B interlocking adjacent and C enveloping D,
    adjacent at the target. B and C share the same start, and A's end and D's start are shared.
    Symmetric version swaps order (so start/end relationships also swap)
    """
    matches = []

    ab_pairs = interlocking_adjacent(interval_trees[0], interval_trees[1], tolerance)
    cd_pairs = envelop_adjacent(interval_trees[2], interval_trees[3], tolerance)

    cd_tree = IntervalTree.from_tuples([(min(c.pos1, d.pos1), max(c.pos2, d.pos2), (c, d)) for c, d in cd_pairs])

    for a, b in ab_pairs:
        candidates = cd_tree.overlap(min(a.pos1, b.pos1) - tolerance, max(a.pos2, b.pos2) + tolerance)
        if not candidates:
            continue
        for cd in candidates:
            c, d = cd.data
            if a.pos1 - tolerance < b.pos1 and approx_equal(b.pos1, c.pos1, tolerance) and approx_equal(a.pos2, d.pos1, tolerance):
                matches.append([a, b, c, d])

            if a.pos2 + tolerance > b.pos2 and approx_equal(b.pos2,  c.pos2, tolerance) and approx_equal(a.pos1, d.pos2, tolerance):
                matches.append([a, b, c, d])

    return matches


def match_nrtra(a_tree, b_tree, tolerance=0):
    """
    Matches the nrTRA structure, which has two interlocking nrTRA-A intervals that overlap by one, and a nrTRA-B
    interval that is enclosed by and adjacent to the outer boundaries of the two nrTRA-A intervals.
    """

    interlocking_pairs = interlocking(b_tree, b_tree, tolerance)
    relevant_pairs = [(left, right) for left, right in interlocking_pairs
                      if approx_equal(abs(left.pos2 - right.pos1), 1, tolerance)]

    matches = []

    for left, right in relevant_pairs:
        candidates = a_tree.envelop(left.pos1 - tolerance, right.pos2 + tolerance)

        larger_b = left if left.pos2 - left.pos1 >= right.pos2 - right.pos1 else right

        for a_interval in candidates:
            a = a_interval.data
            if approx_equal(abs(a.pos1 - left.pos1), 1, tolerance) and approx_equal(abs(right.pos2 - a.pos2), 1, tolerance):
                matches.append([larger_b, a])

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
    Finds the pairs where intervals from the right tree are enveloped by intervals in the leff tree
    """
    valid_pairs = []
    for i_left in left_tree:
        matches = right_tree.envelop(i_left.begin - tolerance, i_left.end + tolerance)
        valid_pairs += [(i_left.data, i_right.data) for i_right in matches]

    return valid_pairs


def envelop_adjacent(left_tree, right_tree, tolerance=0, side='both'):
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
    },
    'INV': {
        'grammar': 'A->a',
        'candidates': ['INV'],
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
    },
    'DUP': {
        'grammar': 'A->AA',
        'candidates': ['DUP'],
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
    },
    'DUP_INV': {
        'grammar': 'A->aa',
        'candidates': ['DUP_INV_A', 'DUP_INV_B', 'DUP_INV_A'],
        'match_fn': lambda x, tolerance: interlocking_intersection(x[0], x[1], x[2], tolerance),
        'start_stop_fn': lambda csv: (csv.records[1].pos1, csv.records[1].pos2),
        'fragment_fn': lambda csv: [
            (csv.second_interval[0], csv.second_interval[1], -1, -1, 'INV'),
            (csv.second_interval[0], csv.second_interval[1], csv.second_interval[1], csv.second_interval[1] + 1, 'COPYinv-PASTE'),
        ],
    },
    'dDUP': {
        'grammar': 'A_->A_A',  # TODO also _A->A_A
        'op_name': 'DUP',
        'candidates': ['dDUP_A', 'dDUP_B'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance),
        'dispersion': True,
    },
    'nrTRA': {
        'grammar': 'A_->_A',  # TODO also _A -> A_
        'candidates': ['nrTRA_B', 'nrTRA_A'],
        'match_fn': lambda x, tolerance: match_nrtra(x[0], x[1], tolerance),
        'dispersion': True,
    },
    'INV_nrTRA': {
        'grammar': 'A_->_a',  # TODO also '_A->a_'
        'candidates': ['INV_nrTRA_A', 'INV_nrTRA_B', 'INV_nrTRA_C'],
        'match_fn': match_inv_nrtra,
        'dispersion': True,
        'key_records': ['INV_nrTRA_B', 'INV_nrTRA_C'],
    },
    'INV_dDUP': {
        'grammar': 'A_->A_a',  # TODO also '_A->a_A'
        'candidates': ['INV_dDUP_A', 'INV_dDUP_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance),
        'dispersion': True,
    },
    'dDUP_INV': {
        'grammar': '_A->a_a',  # TODO also 'A_->a_a'
        'candidates': ['dDUP_INV_A', 'dDUP_INV_B', 'dDUP_INV_C', 'dDUP_INV_D'],
        'match_fn': match_ddup_inv,
        'key_records': ['dDUP_INV_C', 'dDUP_INV_D'],
        'dispersion': True,
    },
    'dupINVdel': {
        'grammar': 'ABC->Aba',
        'candidates': ['dupINVdel_A', 'dupINVdel_B'],
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'start_stop_fn': lambda csv: (csv.first_interval[0], csv.first_interval[1] - 1),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0], csv.second_interval[0], csv.second_interval[1] + 1, csv.second_interval[1] + 2, 'COPYinv-PASTE'),
            (csv.second_interval[0] + 1, csv.second_interval[1], -1, -1, 'INV'),
            (csv.second_interval[1] + 1, csv.first_interval[1] - 1, -1, -1, 'CUT'),
        ],
    },
    'delINVdup': {
        'grammar': 'ABC->cbC',
        'candidates': ['delINVdup_A', 'delINVdup_B'],
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'start_stop_fn': lambda csv: (csv.first_interval[0] + 1, csv.first_interval[1]),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0] + 1, csv.second_interval[0] - 1, -1, -1, 'CUT'),
            (csv.second_interval[0], csv.second_interval[1] - 1, -1, -1, 'INV'),
            (csv.second_interval[1], csv.first_interval[1], csv.first_interval[0], csv.first_interval[1] + 1, 'COPYinv-PASTE'),
        ]
    },
    'delINVdel': {
        'grammar': 'ABC->b',
        'candidates': ['delINVdel_A', 'delINVdel_A'],
        'key_records': ['delINVdel_A', 'delINVdel_A'],
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance, margin=1),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0] + 1, csv.second_interval[0] - 1, -1, -1, 'CUT'),
            (csv.second_interval[0], csv.first_interval[1], -1, -1, 'INV'),
            (csv.first_interval[1] + 1, csv.second_interval[1] - 1, -1, -1, 'CUT')
        ],
        'start_stop_fn': lambda csv: (csv.first_interval[0] + 1, csv.second_interval[1] - 1),
        'is_self_matching': True,
    },
    'dupINVdup': {
        'grammar': 'ABC->AcbaA',
        'candidates': ['dupINVdup_A', 'dupINVdup_A'],
        'key_records': ['dupINVdup_A', 'dupINVdup_A'],
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0], csv.second_interval[0], csv.first_interval[1] - 1, csv.first_interval[1] + 1, 'COPYinv-PASTE'),
            (csv.second_interval[0] + 1, csv.first_interval[1] - 1, -1, -1, 'INV'),
            (csv.first_interval[1], csv.second_interval[1], csv.second_interval[0] + 1, csv.second_interval[0] + 2, 'COPYinv-PASTE')
        ],
        'start_stop_fn': lambda csv: (csv.first_interval[0], csv.second_interval[1]),
        'is_self_matching': True,
    },
    'delINV': {
        'grammar': 'AB->b',
        'candidates': ['delINV_A', 'delINV_B'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'end'),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0] + 1, csv.second_interval[0] - 1, -1, -1, 'CUT'),
            (csv.second_interval[0], csv.first_interval[1], -1, -1, 'INV')
        ],
        'start_stop_fn': lambda csv: (csv.first_interval[0] + 1, csv.second_interval[1] - 1),
    },
    'INVdel': {
        'grammar': 'AB->a',
        'candidates': ['INVdel_B', 'INVdel_A'],
        'key_records': ['INVdel_B', 'INVdel_A'],
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'start'),
        'fragment_fn': lambda csv: [
            (csv.second_interval[0], csv.first_interval[1], -1, -1, 'INV'),
            (csv.first_interval[1] + 1, csv.second_interval[1] - 1, -1, -1, 'CUT')
        ],
        'start_stop_fn': lambda csv: (csv.first_interval[0] + 1, csv.second_interval[1] - 1),
    },
    'dupINV': {
        'grammar': 'AB->Aba',
        'candidates': ['dupINV_A', 'dupINV_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='end'),
        'fragment_fn': lambda csv: [
            (csv.first_interval[0], csv.second_interval[0], csv.second_interval[1] + 1, csv.second_interval[1] + 2, 'COPYinv-PASTE'),
            (csv.second_interval[0] + 1, csv.second_interval[1], -1, -1, 'INV'),
        ],
        'start_stop_fn': lambda csv: (csv.first_interval[0], csv.second_interval[1]),
    },
    'INVdup': {
        'grammar': 'AB->baB',
        'candidates': ['INVdup_A', 'INVdup_B'],
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='start'),
        'fragment_fn': lambda csv: [
            (csv.second_interval[0], csv.second_interval[1] - 1, -1, -1, 'INV'),
            (csv.second_interval[1], csv.first_interval[1], csv.second_interval[0] - 1, csv.second_interval[0], 'COPYinv-PASTE'),
        ],
        'start_stop_fn': lambda csv: (csv.second_interval[0], csv.first_interval[1]),
    }
}
