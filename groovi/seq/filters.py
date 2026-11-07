from intervaltree import IntervalTree
from collections import defaultdict, Counter
import logging

def check_length_limits(fragments, config):
    lengths = [f.end - f.start for f in fragments]
    if sum(lengths) < config.minimum_length:
        return f"total length < {config.minimum_length}"
    if sum(lengths) > config.maximum_length:
        return f"total length > {config.maximum_length}"
    if min(lengths) < config.minimum_fragment_length:
        return f"smallest fragment < {config.minimum_fragment_length}"
    return None

def filter_csvs(svs, config):
    kept, dropped = [], []
    for sv in svs:
        failure = check_length_limits(sv.fragments, config) or (None if sv.validated else 'failed validation')
        if failure:
            dropped.append((sv, failure))
            continue
        sv.grammars = [(g, frags) for g, frags in sv.grammars if not check_length_limits(frags, config)]
        kept.append(sv)

    if config.deduplicate_calls:
        kept, duplicates = deduplicate_svs(kept, config.duplication_threshold)
        dropped += [(sv, 'near-duplicate') for sv in duplicates]

    filtered_by_type, filtered_by_failure = Counter(), Counter()
    with open(config.filter_log_file, "a") as log:
        for sv, failure in dropped:
            log.write(f"{sv}\t{failure}\n")
            filtered_by_type[sv.sv_type] += 1
            filtered_by_failure[failure] += 1

    logging.debug(f"Filtered (class): {dict(sorted(filtered_by_type.items()))}")
    logging.debug(f"Filtered (reason): {dict(sorted(filtered_by_failure.items()))}")
    return kept

def deduplicate_svs(svs, overlap_frac = 0.7, size_sim = 0.7):
    """
    Remove duplicates if the full set of fragments meets the fractional overlap
    and size similarity thresholds against an existing SV.
    """

    def get_evidence(csv):
        return sum([rec.evidence if rec.evidence else 0 for rec in csv.records.values()]) / (len(csv.records) or 1)

    svs = sorted(svs, key=get_evidence, reverse=True)
    chrom_intervals = defaultdict(IntervalTree)
    kept_svs  = []
    dropped_svs = []
    for sv in svs:
        keep = True
        # Sort fragments canonically to guarantee 1-to-1 comparison aligns correctly
        sorted_frags = sorted(sv.fragments, key=lambda f: (f.chrom, f.start, f.end))
        anchor_frag = sorted_frags[0]
        tree = chrom_intervals[(anchor_frag.chrom, sv.sv_type)]

        # Query the tree using the anchor fragment's actual interval
        # (Using max(start + 1, end) ensures a valid interval even for 0-length insertions)
        anchor_end_padded = max(anchor_frag.start + 1, anchor_frag.end)
        candidates = tree.overlap(anchor_frag.start, anchor_end_padded)

        for match in candidates:
            accepted_sv, accepted_sorted_frags = match.data
            # Verify soft overlap for every single pair
            all_match = True
            for f_new, f_accepted in zip(sorted_frags, accepted_sorted_frags):

                # Calculate lengths
                len_new = max(1, f_new.end - f_new.start)
                len_accepted = max(1, f_accepted.end - f_accepted.start)

                # Calculate the intersection distance
                intersection = max(0, min(f_new.end, f_accepted.end) - max(f_new.start, f_accepted.start))

                # Calculate similarity metrics
                current_size_frac = min(len_new, len_accepted) / max(len_new, len_accepted)
                current_rec_overlap = intersection / max(len_new, len_accepted)

                # If a pair fails BOTH the size similarity and overlap thresholds, it's not a duplicate
                if current_size_frac < size_sim and current_rec_overlap < overlap_frac:
                    all_match = False
                    break

            if all_match:
                keep = False
                break

        if keep:
            tree.addi(anchor_frag.start, anchor_end_padded, data=(sv, sorted_frags))
            kept_svs.append(sv)
        else:
            dropped_svs.append(sv)
    return kept_svs, dropped_svs
