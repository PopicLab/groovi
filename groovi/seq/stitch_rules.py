from collections import defaultdict
from intervaltree import IntervalTree

from groovi.seq.sv import Fragment

def approx_equal(a, b, threshold=0):
    return abs(a - b) <= threshold

# Stitching functions

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


def match_ddup_inv(interval_trees, tolerance=0, side='end'):
    """
    dDUP_INV has four record types A, B, C, D with A and B interlocking adjacent on both sides and C enveloping D,
    adjacent at the target on `side`: 'end' when the target is downstream of the AB region (B and C share a start),
    'start' when it is upstream (A and C share an end).
    """
    matches = []

    ab_pairs = interlocking_adjacent(interval_trees[0], interval_trees[1], tolerance=tolerance, side='both')
    cd_pairs = envelop_adjacent(interval_trees[2], interval_trees[3], tolerance, side=side)

    cd_tree = IntervalTree.from_tuples([(min(c.pos1, d.pos1), max(c.pos2, d.pos2), (c, d)) for c, d in cd_pairs])

    for a, b in ab_pairs:
        candidates = cd_tree.overlap(min(a.pos1, b.pos1) - tolerance, max(a.pos2, b.pos2) + tolerance)
        for cd in candidates:
            c, d = cd.data
            if side == 'end' and approx_equal(b.pos1, c.pos1, tolerance) and c.pos2 > b.pos2 or \
                side == 'start' and approx_equal(a.pos2, c.pos2, tolerance) and c.pos1 < a.pos1:
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


def match_inv_nrtra(trees, tolerance=0, side='end'):
    """
    Seeks triples where B and C intervals are envelop adjacent at the target on `side` ('end' when the target is
    downstream of the source, 'start' when upstream), and A surrounds the non-adjacent endpoints of B and C
    """
    matches = []

    bc_pairs = envelop_adjacent(trees[1], trees[2], tolerance, side=side)

    for b, c in bc_pairs:
        if side == 'end':
            target_interval = (b.pos1, c.pos1)
        else:
            target_interval = (c.pos2, b.pos2)

        candidates = trees[0].overlap(target_interval[0], target_interval[1])

        for a_interval in candidates:
            a = a_interval.data
            if approx_equal(target_interval[0] - 1, a.pos1, tolerance) and approx_equal(target_interval[1] + 1, a.pos2, tolerance):
                matches.append([a, b, c])
    return matches

def match_ddup_inv_left(trees, tolerance=0):
    """
    dDUP_INV with the target upstream of the source has edges E, F (dispersion) and G, H (inversion): the same
    structure as A, B, C, D with the pairs swapped. Returns [E, F, G, H].
    """
    return [[e, f, g, h] for g, h, e, f in match_ddup_inv([trees[2], trees[3], trees[0], trees[1]], tolerance, side='start')]


def match_inv_nrtra_left(trees, tolerance=0):
    """
    INV_nrTRA with the target upstream of the source has edges D, E (dispersion) and F (deletion): the same
    structure as A, B, C with the deletion edge last. Returns [D, E, F].
    """
    return [[d, e, f] for f, d, e in match_inv_nrtra([trees[2], trees[0], trees[1]], tolerance, side='start')]


def nrtra_fragments(csv):
    """
    nrTRA edges are not direction-specific (A, B, A), so the graph is symmetric and the source could be either side
    of the B edge. Use the heuristic that the source is the shorter of the two possible segments.
    """
    a1, b, a2 = csv.records['A1'], csv.records['B'], csv.records['A2']
    candidates = [(b.pos1, a2.pos1, b.pos2 + 1, b.chr2),
                  (a1.pos2, b.pos2, a1.pos1 + 1, a1.chr1)]
    source_start, source_end, target, target_chrom = min(candidates, key=lambda t: t[1] - t[0])
    # TODO: Resolve source and target ambiguity by evidence
    return [Fragment(chrom=csv.chrom, start=source_start, end=source_end + 1,
                     target_chrom=target_chrom, target_start=target, target_end=target + 1,
                     operation='CUT-PASTE')]

# Validation methods
def validate_non_inter_chromosomal(csv):
    """
    Checks every fragment with a target is on the SV's chromosome
    """
    if not csv.fragments:
        return True
    chrom = csv.fragments[0].chrom
    return all(chrom == frag.chrom == frag.target_chrom for frag in csv.fragments if frag.target_chrom)

def validate_positive_lengths(csv):
    """
    Checks every fragment interval is strictly positive in length
    """
    return all(frag.end > frag.start for frag in csv.fragments)

def validate_contiguous_segments(csv):
    """
    Checks fragment intervals form a contiguous sequence of segments
    """
    intervals = sorted(set((frag.start, frag.end) for frag in csv.fragments))
    return all(end == next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:]))

def validate_targets_on_breakpoints(csv):
    """
    Checks every target start lands on a fragment interval boundary, for non dispersed SVs
    """
    breakpoints = {coord for frag in csv.fragments for coord in (frag.start, frag.end)}
    return all(frag.target_start in breakpoints for frag in csv.fragments if frag.target_start)

def validate_no_split_segments(csv):
    """
    Checks no target lands strictly inside a fragment's interval, implying a split segment
    """
    for frag in csv.fragments:
        if frag.target_start is None:
            continue
        for source in csv.fragments:
            if not frag.target_chrom or frag.target_chrom != source.chrom:
                continue  # a target on another chromosome cannot split this segment
            if source.start < frag.target_start < source.end:
                return False
    return True

# SV Registry

"""
    One entry per edge pattern, keyed by pattern. SV types with direction-specific edges have one pattern per
    direction ('<type>_right': target downstream of the source, '<type>_left': upstream).
    Fields and default values:
    'sv_type': SV type the pattern is reported as (the pattern)
    'direction': 'right' or 'left' for directional entries (None). nrTRA's grammar is a dict {'right', 'left'}
                 resolved from its fragments, since its edges are not direction-specific.
    'candidates': ordered dict role -> edge type; fragment_fn reads records by role (csv.records[role])
    'match_fn': (trees, tolerance) -> record lists, one tree per candidate and each list in candidate order
    'op_name': Replacement SV name for BED output (SV name)
    'fragment_fn': function to compute fragments (None)
    'start_stop_fn': function to compute start and stop of VCF entry (None)
"""
SV_REGISTRY = {
    'DEL': {
        'grammar': 'A->',
        'candidates': {'A': 'DEL'},
        'op_name': 'CUT',
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2, operation='CUT'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints] 
    },
    'INV': {
        'grammar': 'A->a',
        'candidates': {'A': 'INV_A', 'B': 'INV_B'},
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'both'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2 + 1,
                     operation='INV'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'DUP': {
        'grammar': 'A->AA',
        'candidates': {'A': 'DUP'},
        'match_fn': lambda x, tolerance: identity_match(x, tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1, end=csv.records['A'].pos2 + 1,
                     operation='DUP'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'DUP_INV': {
        'grammar': 'A->aa',
        'candidates': {'A': 'DUP_INV_A', 'B': 'DUP_INV_B', 'C': 'DUP_INV_C'},
        'match_fn': lambda x, tolerance: interlocking_intersection(x[0], x[1], x[2], tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.records['B'].chr1, start=csv.records['B'].pos1, end=csv.records['B'].pos2 + 1, operation='INV'),
            Fragment(chrom=csv.records['B'].chr1, start=csv.records['B'].pos1, end=csv.records['B'].pos2 + 1,
                     target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1, target_end=csv.records['B'].pos2 + 2,
                     operation='COPYinv-PASTE'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'dDUP_right': {
        'sv_type': 'dDUP',
        'direction': 'right',
        'grammar': 'A_->A_A',
        'op_name': 'DUP',
        'dispersion': True,
        'candidates': {'A': 'dDUP_A', 'B': 'dDUP_B'},
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, side='end'),
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['A'].chr2, target_start=csv.records['A'].pos2 + 1,
                     target_end=csv.records['A'].pos2 + 2, operation='COPY-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'dDUP_left': {
        'sv_type': 'dDUP',
        'direction': 'left',
        'grammar': '_A->A_A',
        'op_name': 'DUP',
        'dispersion': True,
        # 'candidates': {'C': 'dDUP_C', 'D': 'dDUP_D'},
        'candidates': {'B': 'dDUP_B', 'A': 'dDUP_A'},
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, side='start'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                                 target_chrom=csv.records['B'].chr1, target_start=csv.records['B'].pos1 + 1,
                                 target_end=csv.records['B'].pos1 + 2, operation='COPY-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'nrTRA': {
        'grammar': {'right': 'A_->_A', 'left': '_A->A_'},
        'candidates': {'A1': 'nrTRA_A', 'B': 'nrTRA_B', 'A2': 'nrTRA_A'},
        'match_fn': lambda x, tolerance: match_nrtra(x[0], x[1], x[2], tolerance),
        'dispersion': True,
        'fragment_fn': nrtra_fragments,
        'validation_fns': [validate_no_split_segments]
    },
    'INV_nrTRA_right': {
        'sv_type': 'INV_nrTRA',
        'direction': 'right',
        'grammar': 'A_->_a',
        'candidates': {'A': 'INV_nrTRA_A', 'B': 'INV_nrTRA_B', 'C': 'INV_nrTRA_C'},
        'match_fn': lambda x, tolerance: match_inv_nrtra(x, tolerance, side='end'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['C'].pos1 + 1,
                     target_chrom=csv.records['C'].chr2, target_start=csv.records['C'].pos2 + 1,
                     target_end=csv.records['C'].pos2 + 2, operation='CUTinv-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'INV_nrTRA_left': {
        'sv_type': 'INV_nrTRA',
        'direction': 'left',
        'grammar': '_A->a_',
        # 'candidates': {'D': 'INV_nrTRA_D', 'E': 'INV_nrTRA_E', 'F': 'INV_nrTRA_F'},
        'candidates': {'B': 'INV_nrTRA_B', 'C': 'INV_nrTRA_C', 'A': 'INV_nrTRA_A'},
        'match_fn': match_inv_nrtra_left,
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['C'].pos2, end=csv.records['B'].pos2 + 1,
                     target_chrom=csv.records['B'].chr1, target_start=csv.records['B'].pos1 + 1,
                     target_end=csv.records['B'].pos1 + 2, operation='CUTinv-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'INV_dDUP_right': {
        'sv_type': 'INV_dDUP',
        'direction': 'right',
        'grammar': 'A_->A_a',
        'candidates': {'A': 'INV_dDUP_A', 'B': 'INV_dDUP_B'},
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='end'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1,
                     target_end=csv.records['B'].pos2 + 2, operation='COPYinv-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'INV_dDUP_left': {
        'sv_type': 'INV_dDUP',
        'direction': 'left',
        'grammar': '_A->a_A',
        'candidates': {'A': 'INV_dDUP_A', 'B': 'INV_dDUP_B'},
        # 'candidates': {'C': 'INV_dDUP_C', 'D': 'INV_dDUP_D'},
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='start'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                                 target_chrom=csv.records['A'].chr1, target_start=csv.records['A'].pos1 + 1,
                                 target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'dDUP_INV_right': {
        'sv_type': 'dDUP_INV',
        'direction': 'right',
        'grammar': 'A_->a_a',
        'candidates': {'A': 'dDUP_INV_A', 'B': 'dDUP_INV_B', 'C': 'dDUP_INV_C', 'D': 'dDUP_INV_D'},
        'match_fn': lambda x, tolerance: match_ddup_inv(x, tolerance, side='end'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2,
                     target_chrom=csv.records['D'].chr2, target_start=csv.records['D'].pos2 + 1,
                     target_end=csv.records['D'].pos2 + 2, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'dDUP_INV_left': {
        'sv_type': 'dDUP_INV',
        'direction': 'left',
        'grammar': '_A->a_a',
        'candidates': {'E': 'dDUP_INV_E', 'F': 'dDUP_INV_F', 'G': 'dDUP_INV_G', 'H': 'dDUP_INV_H'},
        'match_fn': match_ddup_inv_left,
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            Fragment(chrom=csv.chrom, start=csv.records['H'].pos1, end=csv.records['H'].pos2,
                     target_chrom=csv.records['E'].chr1, target_start=csv.records['E'].pos1 + 1,
                     target_end=csv.records['E'].pos1 + 2, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['H'].pos1, end=csv.records['H'].pos2, operation='INV')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'dupINVdel': {
        'grammar': 'ABC->Aba', #Acba
        'candidates': {'A': 'dupINVdel_A', 'B': 'dupINVdel_B'},
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['A'].chr2, target_start=csv.records['B'].pos2 + 1, target_end=csv.records['B'].pos2 + 2,
                     operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['B'].pos2 + 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2 + 1, end=csv.records['A'].pos2, operation='CUT'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINVdup': {
        'grammar': 'ABC->cbC',
        'candidates': {'A': 'delINVdup_A', 'B': 'delINVdup_B'},
        'match_fn': lambda x, tolerance: envelop(x[0], x[1], tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                     target_chrom=csv.records['A'].chr2, target_start=csv.records['A'].pos1 + 1,
                     target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINVdel': {
        'grammar': 'ABC->b',
        'candidates': {'A': 'delINVdel_A', 'B': 'delINVdel_B'},
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance, margin=1),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ],
        'is_self_matching': True,
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'dupINVdup': {
        'grammar': 'ABC->AcbaC',
        'candidates': {'A': 'dupINVdup_A', 'B': 'dupINVdup_B'},
        'match_fn': lambda x, tolerance: interlocking(x[0], x[1], tolerance),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['A'].chr2, target_start=csv.records['A'].pos2,
                     target_end=csv.records['A'].pos2 + 1, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['A'].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos2, end=csv.records['B'].pos2 + 1,
                     target_chrom=csv.records['B'].chr1, target_start=csv.records['B'].pos1 + 1, target_end=csv.records['B'].pos1 + 2,
                     operation='COPYinv-PASTE')
        ],
        'is_self_matching': True,
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINV': {
        'grammar': 'AB->b',
        'candidates': {'A': 'delINV_A', 'B': 'delINV_B'},
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'end'),
        'fragment_fn': lambda csv:
        [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV')
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV')
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'INVdel': {
        'grammar': 'AB->a',
        'candidates': {'A': 'INVdel_A', 'B': 'INVdel_B'},
        'match_fn': lambda x, tolerance: interlocking_adjacent(x[0], x[1], tolerance, 'start'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2 + 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'dupINV': {
        'grammar': 'AB->Aba',
        'candidates': {'A': 'dupINV_A', 'B': 'dupINV_B'},
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='end'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['B'].chr2, target_start=csv.records['A'].pos2,
                     target_end=csv.records['A'].pos2 + 1, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['A'].pos2, operation='INV'),
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                     target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1,
                     target_end=csv.records['B'].pos2 + 2, operation='COPYinv-PASTE'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['B'].pos2 + 1, operation='INV'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'INVdup': {
        'grammar': 'AB->baB',
        'candidates': {'A': 'INVdup_A', 'B': 'INVdup_B'},
        'match_fn': lambda x, tolerance: envelop_adjacent(x[0], x[1], tolerance, side='start'),
        'fragment_fn': lambda csv: [
            Fragment(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                     target_chrom=csv.chrom, target_start=csv.records['A'].pos1 + 1,
                     target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE'),
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV'),
            Fragment(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                     target_chrom=csv.chrom, target_start=csv.records['B'].pos1,
                     target_end=csv.records['B'].pos1 + 1, operation='COPYinv-PASTE'),
        ],
        'validation_fns': [validate_non_inter_chromosomal, validate_positive_lengths,
                           validate_contiguous_segments, validate_targets_on_breakpoints]
    }
}

# EDGE_TYPE_TO_PATTERN = {edge_type: pattern
#                         for pattern, entry in SV_REGISTRY.items()
#                         for edge_type in entry['candidates'].values()}

EDGE_TYPE_TO_STITCH_CLASSES = defaultdict(list)
for pattern, entry in SV_REGISTRY.items():
    for edge_type in entry['candidates'].values():
        EDGE_TYPE_TO_STITCH_CLASSES[edge_type].append(pattern)
EDGE_TYPE_TO_STITCH_CLASSES = dict(EDGE_TYPE_TO_STITCH_CLASSES)