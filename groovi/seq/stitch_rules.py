from groovi.seq.sv import ComplexSV, JunctionStitchIndex
from collections import defaultdict, Counter
from functools import partial
from dataclasses import replace
from groovi.seq.sv import FragmentOp
import logging
import warnings

def assemble_csvs(recs, tolerance=0):
    junction_db = defaultdict(lambda: JunctionStitchIndex(buffer=tolerance + 1))
    for rec in recs:
        if rec.type not in KNOWN_EDGE_TYPES:
            warnings.warn(f"Unknown SV type: {rec.type}")
            continue
        junction_db[rec.type].add(rec)
    csvs = []
    stitched_counts = Counter()
    for stitch_class, spec in SV_REGISTRY.items():
        edge_types = list(spec['candidates'].values())
        if not all(t in junction_db for t in edge_types): continue
        matches = spec['match_fn'](*(junction_db[t] for t in edge_types), tolerance=tolerance)
        stitched_counts[stitch_class] = len(matches)
        for match in matches:
            chrom = match[0].chr1
            qual = round(sum(rec.qual for rec in match) / len(match))
            csvs.append(ComplexSV(spec, chrom, qual, match))
    logging.debug("Edge stitching diagnostics")
    edge_counts = Counter(rec.type for rec in recs)
    for stitch_class, spec in SV_REGISTRY.items():
        edge_types = list(spec['candidates'].values())
        logging.debug(f'-- {stitch_class} --')
        logging.debug(f'Num edges: {", ".join(f"{t}: {edge_counts[t]}" for t in edge_types)}')
        logging.debug(f'Num stitched: {stitched_counts[stitch_class]}')
    return csvs

def approx_equal(a, b, threshold=0):
    return abs(a - b) <= threshold

# Stitching functions

def identity_match(index, tolerance=0):
    return [[rec] for rec in index]


def interlocks(first, second, tolerance=0, margin=0):
    return (second.chr1, second.pos1) < (first.chr2, first.pos2 + tolerance) and \
        (second.chr2, second.pos2) > (first.chr1, first.pos1 - tolerance) and \
        (first.chr1, first.pos1 + margin) <= (second.chr1, second.pos1 + tolerance) and \
        (first.chr2, first.pos2 + margin) <= (second.chr2, second.pos2 + tolerance)


def envelops(outer, inner, tolerance=0):
    return (inner.chr1, inner.pos1) >= (outer.chr1, outer.pos1 - tolerance) and \
        (inner.chr2, inner.pos2) <= (outer.chr2, outer.pos2 + tolerance)


def interlock(first_index, second_index, tolerance=0, margin=0):
    return [(first, second)
            for first in first_index
            for second in second_index.overlapping(first.chr1, first.pos1 - tolerance, first.pos2 + tolerance)
            if second is not first and interlocks(first, second, tolerance, margin)]


def envelop(outer_index, inner_index, tolerance=0):
    return [(outer, inner)
            for outer in outer_index
            for inner in inner_index.enveloped(outer.chr1, outer.pos1 - tolerance, outer.pos2 + tolerance)]


def adjacent_pairs(first_index, second_index, related, tolerance=0, side='either'):
    pairs = []
    for first in first_index:
        candidates = set()
        if side in ('start', 'both', 'either'):
            candidates |= second_index.near(first.chr1, first.pos1)
        if side in ('end', 'either'):
            candidates |= second_index.near(first.chr2, first.pos2)

        for second in candidates:
            if second is first or not related(first, second, tolerance):
                continue
            starts_adjacent = first.chr1 == second.chr1 and approx_equal(abs(second.pos1 - first.pos1), 1, tolerance)
            ends_adjacent = first.chr2 == second.chr2 and approx_equal(abs(second.pos2 - first.pos2), 1, tolerance)
            if side == 'start' and starts_adjacent or \
                    side == 'end' and ends_adjacent or \
                    side == 'both' and starts_adjacent and ends_adjacent or \
                    side == 'either' and (starts_adjacent or ends_adjacent):
                pairs.append((first, second))
    return pairs


def interlocking_adjacent(first_index, second_index, tolerance=0, side='either'):
    return adjacent_pairs(first_index, second_index, interlocks, tolerance, side)


def envelop_adjacent(outer_index, inner_index, tolerance=0, side='either'):
    return adjacent_pairs(outer_index, inner_index, envelops, tolerance, side)


def interlocking_intersection(first_index, middle_index, second_index, tolerance=0):
    valid_triples = []
    for first, second in interlock(first_index, second_index, tolerance):
        for middle in middle_index.near(second.chr1, second.pos1):
            if middle.chr1 == second.chr1 and approx_equal(middle.pos1, second.pos1, tolerance) and \
                    middle.chr2 == first.chr2 and approx_equal(middle.pos2, first.pos2, tolerance):
                valid_triples.append((first, middle, second))
    return valid_triples


def match_ddup_inv(a_index, b_index, c_index, d_index, tolerance=0, side='end'):
    ds_by_c = defaultdict(list)
    for c, d in envelop_adjacent(c_index, d_index, tolerance, side=side):
        ds_by_c[c].append(d)

    matches = []
    for a, b in interlocking_adjacent(a_index, b_index, tolerance, side='both'):
        if side == 'end':
            cs = [c for c in c_index.near(b.chr1, b.pos1)
                  if c.chr1 == b.chr1 and approx_equal(c.pos1, b.pos1, tolerance) and (c.chr2, c.pos2) > (b.chr2, b.pos2)]
        else:
            cs = [c for c in c_index.near(a.chr2, a.pos2)
                  if c.chr2 == a.chr2 and approx_equal(c.pos2, a.pos2, tolerance) and (c.chr1, c.pos1) < (a.chr1, a.pos1)]
        matches += [[a, b, c, d] for c in cs for d in ds_by_c.get(c, ())]
    return matches


def match_nrtra(a1_index, b_index, a2_index, tolerance=0):
    matches = []
    for a1 in a1_index:
        for a2 in a2_index.near(a1.chr2, a1.pos2):
            if a2 is a1 or not interlocks(a1, a2, tolerance) or \
                    a2.chr1 != a1.chr2 or not approx_equal(a1.pos2 - a2.pos1, 1, tolerance):
                continue
            for b in b_index.near(a1.chr1, a1.pos1):
                if b.chr1 == a1.chr1 and approx_equal(b.pos1 - a1.pos1, 1, tolerance) and \
                        b.chr2 == a2.chr2 and approx_equal(a2.pos2 - b.pos2, 1, tolerance):
                    matches.append([a1, b, a2])
    return matches


def match_inv_nrtra(a_index, b_index, c_index, tolerance=0, side='end'):
    matches = []
    for b, c in envelop_adjacent(b_index, c_index, tolerance, side=side):
        if side == 'end':
            (target_chrom, target_start), (target_end_chrom, target_end) = (b.chr1, b.pos1), (c.chr1, c.pos1)
        else:
            (target_chrom, target_start), (target_end_chrom, target_end) = (c.chr2, c.pos2), (b.chr2, b.pos2)
        if target_chrom != target_end_chrom or target_start >= target_end:
            continue

        for a in a_index.near(target_chrom, target_start):
            if a.chr1 == a.chr2 == target_chrom and a.pos1 < target_end and a.pos2 > target_start and \
                    approx_equal(target_start - a.pos1, 1, tolerance) and approx_equal(a.pos2 - target_end, 1, tolerance):
                matches.append([a, b, c])
    return matches

def match_inv_nrtra_left(b_index, c_index, a_index, tolerance=0):
    return [[b, c, a] for a, b, c in match_inv_nrtra(a_index, b_index, c_index, tolerance, side='start')]


def nrtra_source_choices(csv):
    a1, b, a2 = csv.records['A1'], csv.records['B'], csv.records['A2']
    candidates = [(b.pos1, a2.pos1, b.pos2 + 1, b.chr2),
                  (a1.pos2, b.pos2, a1.pos1 + 1, a1.chr1)]
    return [FragmentOp(chrom=csv.chrom, start=source_start, end=source_end + 1,
                       target_chrom=target_chrom, target_start=target, target_end=target + 1,
                       operation='CUT-PASTE')
            for source_start, source_end, target, target_chrom in sorted(candidates, key=lambda t: t[1] - t[0])]


def nrtra_fragments(csv):
    return nrtra_source_choices(csv)[:1]


def nrtra_alternative_grammars(csv):
    return [(SV_REGISTRY['nrTRA']['grammar']['left' if (frag.target_chrom, frag.target_start) < (frag.chrom, frag.start)
                                             else 'right'], [frag])
            for frag in nrtra_source_choices(csv)[1:]]


def inv_ddup_alternative_grammars(csv):
    """
    The same inverted copy with the spacer between source and target inverted: AB->Aba for a target downstream
    of the source, AB->baB upstream. Not defined for an interchromosomal or adjacent target.
    """
    source = csv.fragments[0]
    if source.target_chrom != source.chrom:
        return []
    if source.target_start > source.end:
        spacer_start, spacer_end, grammar = source.end, source.target_start, 'AB->Aba'
    elif source.target_start < source.start:
        spacer_start, spacer_end, grammar = source.target_start, source.start, 'AB->baB'
    else:
        return []
    return [(grammar, [source, FragmentOp(chrom=csv.chrom, start=spacer_start, end=spacer_end, operation='INV')])]

def dupinvdup_alternative_grammars(csv):
    """
    The same junctions with the middle segment B kept in place: ABC->ABCaC, the inverted copy of A then a copy of C
    inserted after C, and ABC->AcABC, the inverted copy of C then a copy of A inserted after A.
    """
    a, b, c = csv.fragments  # COPYinv-PASTE of A, INV of B, COPYinv-PASTE of C
    keep = replace(b, operation='KEEP')
    after_c = dict(target_chrom=c.chrom, target_start=c.end, target_end=c.end + 1)
    after_a = dict(target_chrom=a.chrom, target_start=a.end, target_end=a.end + 1)
    return [('ABC->ABCaC', [replace(a, **after_c, insord=0), keep,
                            replace(c, operation='COPY-PASTE', **after_c, insord=1)]),
            ('ABC->AcABC', [replace(c, **after_a, insord=0), keep,
                            replace(a, operation='COPY-PASTE', **after_a, insord=1)])]

# Validation methods

def validate_positive_lengths(csv):
    return all(frag.end > frag.start for frag in csv.fragments)

def validate_contiguous_segments(csv):
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
    'match_fn': (*indexes, tolerance) -> record lists, called with one JunctionIndex per candidate in candidate order;
                each list is in candidate order
    'op_name': Replacement SV name for BED output (SV name)
    'fragment_fn': function to compute fragments (None)
    'alternative_grammars_fn': csv -> [(grammar, fragments)], the other grammars the same breakpoints admit (None)
"""
SV_REGISTRY = {
    'DEL': {
        'grammar': 'A->',
        'candidates': {'A': 'DEL'},
        'op_name': 'CUT',
        'match_fn': identity_match,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2,
                       operation='CUT'),
        ],
        'validation_fns': [validate_positive_lengths, validate_targets_on_breakpoints]
    },
    'INV': {
        'grammar': 'A->a',
        'candidates': {'A': 'INV_A', 'B': 'INV_B'},
        'match_fn': partial(interlocking_adjacent, side='both'),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2 + 1,
                       operation='INV'),
        ],
        'validation_fns': [validate_positive_lengths, validate_targets_on_breakpoints]
    },
    'DUP': {
        'grammar': 'A->AA',
        'candidates': {'A': 'DUP'},
        'match_fn': identity_match,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.records['A'].chr1, start=csv.records['A'].pos1, end=csv.records['A'].pos2 + 1,
                       operation='DUP'),
        ],
        'validation_fns': [validate_positive_lengths, validate_targets_on_breakpoints]
    },
    'DUP_INV': {
        'grammar': 'A->aa',
        'candidates': {'A': 'DUP_INV_A', 'B': 'DUP_INV_B', 'C': 'DUP_INV_C'},
        'match_fn': interlocking_intersection,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.records['B'].chr1, start=csv.records['B'].pos1, end=csv.records['B'].pos2 + 1,
                       operation='INV'),
            FragmentOp(chrom=csv.records['B'].chr1, start=csv.records['B'].pos1, end=csv.records['B'].pos2 + 1,
                       target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1,
                       target_end=csv.records['B'].pos2 + 2,
                       operation='COPYinv-PASTE'),
        ],
        'validation_fns': [validate_positive_lengths, validate_targets_on_breakpoints]
    },
    'dDUP_right': {
        'sv_type': 'dDUP',
        'direction': 'right',
        'grammar': 'A_->A_A',
        'op_name': 'DUP',
        'dispersion': True,
        'candidates': {'A': 'dDUP_A', 'B': 'dDUP_B'},
        'match_fn': partial(interlocking_adjacent, side='end'),
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
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
        'candidates': {'B': 'dDUP_B', 'A': 'dDUP_A'},
        'match_fn': partial(interlocking_adjacent, side='start'),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                       target_chrom=csv.records['B'].chr1, target_start=csv.records['B'].pos1 + 1,
                       target_end=csv.records['B'].pos1 + 2, operation='COPY-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'nrTRA': {
        'grammar': {'right': 'A_->_A', 'left': '_A->A_'},
        'candidates': {'A1': 'nrTRA_A', 'B': 'nrTRA_B', 'A2': 'nrTRA_A'},
        'match_fn': match_nrtra,
        'dispersion': True,
        'fragment_fn': nrtra_fragments,
        'alternative_grammars_fn': nrtra_alternative_grammars,
        'validation_fns': [validate_no_split_segments]
    },
    'INV_nrTRA_right': {
        'sv_type': 'INV_nrTRA',
        'direction': 'right',
        'grammar': 'A_->_a',
        'candidates': {'A': 'INV_nrTRA_A', 'B': 'INV_nrTRA_B', 'C': 'INV_nrTRA_C'},
        'match_fn': partial(match_inv_nrtra, side='end'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['C'].pos1 + 1,
                       target_chrom=csv.records['C'].chr2, target_start=csv.records['C'].pos2 + 1,
                       target_end=csv.records['C'].pos2 + 2, operation='CUTinv-PASTE')
        ],
        'validation_fns': [validate_no_split_segments]
    },
    'INV_nrTRA_left': {
        'sv_type': 'INV_nrTRA',
        'direction': 'left',
        'grammar': '_A->a_',
        'candidates': {'B': 'INV_nrTRA_B', 'C': 'INV_nrTRA_C', 'A': 'INV_nrTRA_A'},
        'match_fn': match_inv_nrtra_left,
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve source and target ambiguity by evidence
            FragmentOp(chrom=csv.chrom, start=csv.records['C'].pos2, end=csv.records['B'].pos2 + 1,
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
        'match_fn': partial(envelop_adjacent, side='end'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                       target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1,
                       target_end=csv.records['B'].pos2 + 2, operation='COPYinv-PASTE')
        ],
        'alternative_grammars_fn': inv_ddup_alternative_grammars,
        'validation_fns': [validate_no_split_segments]
    },
    'INV_dDUP_left': {
        'sv_type': 'INV_dDUP',
        'direction': 'left',
        'grammar': '_A->a_A',
        'candidates': {'A': 'INV_dDUP_A', 'B': 'INV_dDUP_B'},
        'match_fn': partial(envelop_adjacent, side='start'),
        'dispersion': True,
        'fragment_fn': lambda csv: [
            # TODO: Resolve target ambiguity by evidence
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                       target_chrom=csv.records['A'].chr1, target_start=csv.records['A'].pos1 + 1,
                       target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE')
        ],
        'alternative_grammars_fn': inv_ddup_alternative_grammars,
        'validation_fns': [validate_no_split_segments]
    },
    'dupINVdel': {
        'grammar': 'ABC->Aba', #Acba
        'candidates': {'A': 'dupINVdel_A', 'B': 'dupINVdel_B'},
        'match_fn': envelop,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                       target_chrom=csv.records['A'].chr2, target_start=csv.records['B'].pos2 + 1, target_end=csv.records['B'].pos2 + 2,
                       operation='COPYinv-PASTE'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['B'].pos2 + 1, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2 + 1, end=csv.records['A'].pos2, operation='CUT'),
        ],
        'alternative_grammars_fn': lambda csv: [
            ('ABC->ABa', [replace(frag, operation='KEEP') if frag.operation == 'INV' else frag
                          for frag in csv.fragments])
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINVdup': {
        'grammar': 'ABC->cbC',
        'candidates': {'A': 'delINVdup_A', 'B': 'delINVdup_B'},
        'match_fn': envelop,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                       target_chrom=csv.records['A'].chr2, target_start=csv.records['A'].pos1 + 1,
                       target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE'),
        ],
        'alternative_grammars_fn': lambda csv: [
            ('ABC->cBC', [replace(frag, operation='KEEP') if frag.operation == 'INV' else frag
                          for frag in csv.fragments])
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINVdel': {
        'grammar': 'ABC->b',
        'candidates': {'A': 'delINVdel_A', 'B': 'delINVdel_B'},
        'match_fn': partial(interlock, margin=1),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ],
        'is_self_matching': True,
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'dupINVdup': {
        'grammar': 'ABC->AcbaC',
        'candidates': {'A': 'dupINVdup_A', 'B': 'dupINVdup_B'},
        'match_fn': interlock,
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                       target_chrom=csv.records['A'].chr2, target_start=csv.records['A'].pos2,
                       target_end=csv.records['A'].pos2 + 1, operation='COPYinv-PASTE'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['A'].pos2, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos2, end=csv.records['B'].pos2 + 1,
                       target_chrom=csv.records['B'].chr1, target_start=csv.records['B'].pos1 + 1, target_end=csv.records['B'].pos1 + 2,
                       operation='COPYinv-PASTE')
        ],
        'alternative_grammars_fn': dupinvdup_alternative_grammars,
        'is_self_matching': True,
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'delINV': {
        'grammar': 'AB->b',
        'candidates': {'A': 'delINV_A', 'B': 'delINV_B'},
        'match_fn': partial(interlocking_adjacent, side='end'),
        'fragment_fn': lambda csv:
        [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV')
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos1, operation='CUT'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV')
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'INVdel': {
        'grammar': 'AB->a',
        'candidates': {'A': 'INVdel_A', 'B': 'INVdel_B'},
        'match_fn': partial(interlocking_adjacent, side='start'),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['A'].pos2 + 1, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['A'].pos2 + 1, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos2 + 1, end=csv.records['B'].pos2, operation='CUT')
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'dupINV': {
        'grammar': 'AB->Aba',
        'candidates': {'A': 'dupINV_A', 'B': 'dupINV_B'},
        'match_fn': partial(envelop_adjacent, side='end'),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                       target_chrom=csv.records['B'].chr2, target_start=csv.records['A'].pos2,
                       target_end=csv.records['A'].pos2 + 1, operation='COPYinv-PASTE'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['A'].pos2, operation='INV'),
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1, end=csv.records['B'].pos1 + 1,
                       target_chrom=csv.records['B'].chr2, target_start=csv.records['B'].pos2 + 1,
                       target_end=csv.records['B'].pos2 + 2, operation='COPYinv-PASTE'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1 + 1, end=csv.records['B'].pos2 + 1, operation='INV'),
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    },
    'INVdup': {
        'grammar': 'AB->baB',
        'candidates': {'A': 'INVdup_A', 'B': 'INVdup_B'},
        'match_fn': partial(envelop_adjacent, side='start'),
        'fragment_fn': lambda csv: [
            FragmentOp(chrom=csv.chrom, start=csv.records['A'].pos1 + 1, end=csv.records['B'].pos2, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                       target_chrom=csv.chrom, target_start=csv.records['A'].pos1 + 1,
                       target_end=csv.records['A'].pos1 + 2, operation='COPYinv-PASTE'),
        ] if csv.records['A'].evidence >= csv.records['B'].evidence else [
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos1, end=csv.records['B'].pos2, operation='INV'),
            FragmentOp(chrom=csv.chrom, start=csv.records['B'].pos2, end=csv.records['A'].pos2 + 1,
                       target_chrom=csv.chrom, target_start=csv.records['B'].pos1,
                       target_end=csv.records['B'].pos1 + 1, operation='COPYinv-PASTE'),
        ],
        'validation_fns': [validate_positive_lengths, validate_contiguous_segments, validate_targets_on_breakpoints]
    }
}

for pattern, entry in SV_REGISTRY.items():
    entry.setdefault('sv_type', pattern)
KNOWN_EDGE_TYPES = {t for spec in SV_REGISTRY.values() for t in spec['candidates'].values()}