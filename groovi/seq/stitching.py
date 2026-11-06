from pysam import VariantFile, VariantRecord
from collections import defaultdict, Counter
from copy import copy
from intervaltree import Interval, IntervalTree
import pysam

from groovi.cli.config import DiscoverConfig
from groovi.seq.sv import SVBreakpoint, Fragment
import groovi.seq.io as io
import logging

import warnings

from groovi.seq.stitch_rules import SV_REGISTRY, EDGE_TYPE_TO_STITCH_CLASSES


class ComplexSV:
    def __init__(self, stitch_class, chrom, qual=None, records=None):
        """
        :param pattern: SV_REGISTRY entry the records were stitched with, e.g. 'DEL' or 'dDUP_left'
        :param records: dict role -> SVBreakpoint, or a list in the entry's candidate order
        """
        sv_registry_info = SV_REGISTRY[stitch_class]

        self.stitch_class = stitch_class
        self.sv_type = sv_registry_info.get('sv_type', stitch_class)
        self.direction = sv_registry_info.get('direction')
        self.chrom = chrom
        self.qual = qual
        self.records = records if isinstance(records, dict) else dict(zip(sv_registry_info['candidates'], records))

        self.fragments = sv_registry_info['fragment_fn'](self)

        # dispersion logic
        if sv_registry_info.get('dispersion', False):
            moved = next(frag for frag in self.fragments if frag.target_start is not None)
            self.target, self.target_chrom = moved.target_start, moved.target_chrom
        else:
            self.target, self.target_chrom = None, None

        self.grammar = sv_registry_info['grammar']
        if isinstance(self.grammar, dict):
            # nrTRA: the fragment function decided the direction
            self.direction = 'left' if self.target < moved.start else 'right'
            self.grammar = self.grammar[self.direction]

        self.validated = all(fn(self) for fn in sv_registry_info['validation_fns'])

        # every grammar the stitched breakpoints admit, as (grammar, fragments); alternatives must pass validation
        self.grammars = [(self.grammar, self.fragments)]
        for grammar, fragments in sv_registry_info.get('alternative_grammars_fn', lambda csv: [])(self):
            alternative = copy(self)
            alternative.fragments = fragments
            if all(fn(alternative) for fn in sv_registry_info['validation_fns']):
                self.grammars.append((grammar, fragments))

    def __str__(self):
        return repr((self.sv_type, self.chrom, self.qual, self.records))

    @staticmethod
    def from_records(recs, stitch_class):
        # pattern = EDGE_TYPE_TO_SV_CLASSES[recs[0].type]
        chrom = recs[0].chr1
        qual = round(sum([rec.qual for rec in recs]) / len(recs))

        return ComplexSV(stitch_class, chrom, qual, recs)

    @staticmethod
    def check_equality(csv1, csv2):
        return all([
            csv1.sv_type == csv2.sv_type,
            csv1.chrom == csv2.chrom,
            csv1.qual == csv2.qual,
            csv1.records == csv2.records,
            csv1.fragments == csv2.fragments
        ])

    def get_bed_info(self):
        # get the bed record(s) information representing the sv. DEPRECATED
        bed_recs = []

        # add the frag-specific records
        for frag in self.fragments:
            bed_recs.append({
                'chrom': frag.chrom, 'source_s': frag.start, 'source_e': frag.end,
                'target_s': frag.target_start, 'target_e': frag.target_end,
                'transform': frag.operation, 'sv_len': frag.end - frag.start, 'gt': './.',
                'sv_type': self.sv_type
            })
        return bed_recs

    def start_stop_fn(self):
        return min(frag.start for frag in self.fragments), max(frag.end for frag in self.fragments)

    def get_vcf_info(self):
        """
        Creates a single VCF record for the SV.
        """
        start, stop = self.start_stop_fn()

        vcf_info = {'chrom': self.chrom,
                    'start': start,
                    'stop': stop,
                    'sv_type': self.sv_type,
                    'qual': self.qual,
                    'sv_len': stop - start,
                    'gt': (0, 0), 'op_type': SV_REGISTRY[self.stitch_class].get('op_name', self.sv_type)}

        if self.target:
            vcf_info['target'] = self.target

        return vcf_info

def find_matches(recs, tolerance = 0):
    """
    Finds all matches for a given SV type by running a pipelined search.
    This avoids an O(N^d) search by finding matches in a linear sequence.

    :param recs: A list of VCF records with id, start, stop, and (optional) qual attributes.
    :param tolerance: An integer tolerance for match conditions.
    :return: A list of ComplexSV objects containing the found matches.
    """
    # Build interval trees from the input VCF records

    observed_types = set()

    trees = defaultdict(IntervalTree)
    for rec in recs:
        if rec.type not in EDGE_TYPE_TO_STITCH_CLASSES:
            warnings.warn(f"Unknown SV type: {rec.type}")
            continue

        observed_types.update(EDGE_TYPE_TO_STITCH_CLASSES[rec.type])

        rec_type = rec.type

        # Storing the original record in the interval data as a list
        trees[rec_type].add(Interval(rec.pos1, rec.pos2, rec))

    all_matches = []

    for stitch_class, sv_info in SV_REGISTRY.items():
        if stitch_class not in observed_types:
            continue
        logging.info(f"Finding stitch matches for {stitch_class}")

        candidates: list[IntervalTree] = [trees[rec_type] for rec_type in sv_info['candidates'].values()] # trees for each edge type according to classes

        match_fn = sv_info.get('match_fn', None)
        if not candidates:
            continue

        matches = match_fn(candidates, tolerance)

        # remove all non-dispersions across different contigs
        if not sv_info.get('dispersion', False):
            single_chrom = []
            for match in matches:
                matched_chroms = {c for bkp in match for c in (bkp.chr1, bkp.chr2)}
                if len(matched_chroms) == 1:
                    single_chrom.append(match)

            if len(single_chrom) < len(matches):
                logging.info(f'Filtered {len(matches) - len(single_chrom)} non-dispersions matched across contigs')
            matches = single_chrom

        all_matches.extend([ComplexSV.from_records(recs, stitch_class) for recs in matches])

    return all_matches


def filter_overlapping_svs(merged_svs, prioritize_evidence = False):
    # filter the list of ComplexSV objects created by dispatch_match_functions()
    # taking the higher-confidence ComplexSV in the case of any collision
    chrom_intervals = defaultdict(IntervalTree)
    kept_svs = []
    dropped_svs = []
    dropped_reasons =[]

    if prioritize_evidence:
        key_func = lambda csv: -sum([rec.evidence if rec.evidence else 0 for rec in csv.records.values()]) / len(csv.records)
    else:
        key_func = lambda csv: -csv.qual

    for rec in sorted(merged_svs, key=key_func):
        occupied_intervals = [(frag.start, frag.end) for frag in rec.fragments]
        occupied_intervals += [(frag.target_start, frag.target_end) for frag in rec.fragments if frag.target_start]

        if rec.chrom not in chrom_intervals or \
                not any(chrom_intervals[rec.chrom].overlap(*occupied_interval) for occupied_interval in occupied_intervals):
            for occupied_interval in occupied_intervals:
                chrom_intervals[rec.chrom].addi(*sorted(occupied_interval))
            kept_svs.append(rec)
        else:
            dropped_svs.append(rec)

    dropped_reasons = ['deduplication' for _ in dropped_svs]
    return kept_svs, dropped_svs, dropped_reasons


def _fragments_match(frag1, frag2, buffer):
    """Helper to check if two individual fragments match within the buffer."""
    if frag1.chrom != frag2.chrom: return False
    if abs(frag1.start - frag2.start) > buffer: return False
    if abs(frag1.end - frag2.end) > buffer: return False

    # Handle optional target/translocation coordinates
    t_chrom1 = getattr(frag1, 'target_chrom', None)
    t_chrom2 = getattr(frag2, 'target_chrom', None)
    if t_chrom1 != t_chrom2: return False

    t_start1 = getattr(frag1, 'target_start', None)
    t_start2 = getattr(frag2, 'target_start', None)
    if (t_start1 is None) != (t_start2 is None): return False
    if t_start1 is not None and abs(t_start1 - t_start2) > buffer: return False

    t_end1 = getattr(frag1, 'target_end', None)
    t_end2 = getattr(frag2, 'target_end', None)
    if (t_end1 is None) != (t_end2 is None): return False
    if t_end1 is not None and abs(t_end1 - t_end2) > buffer: return False

    return True


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
        if not sv.fragments:
            continue

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

            if len(sorted_frags) != len(accepted_sorted_frags):
                continue

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

    dropped_reasons = ['deduplication' for _ in dropped_svs]
    return kept_svs, dropped_svs, dropped_reasons


def export_callset_to_vcf(filtered_svs, output_path, contigs, sep_frag_recs=False):
    """
    Export function to write list of post-(merge+filter) ComplexSV objects to VCF
    """
    header = pysam.VariantHeader()
    for ctg in contigs: header.add_line("##contig=<ID=%s>\n" % ctg.name)
    header.info.add('END', number=1, type='Integer',
                    description="End position of the variant described in this record")
    header.info.add('SVID', number=1, type='String', description="ID of parent SV of which this record is one part")
    header.info.add('SVTYPE', number=1, type='String', description="type of parent SV of which this record is one part")
    header.info.add('OP_TYPE', number=1, type='String', description="Type of operation performed for the corresponding record")
    header.info.add('INSORD', number=1, type='Integer', description='Insertion order for insertions at given position')
    header.info.add('SVLEN', number=1, type='Integer', description="Length of structural variant")
    header.info.add('TARGET', number=1, type='Integer', description="Target location for dispersion or translocation")
    header.info.add('TARGET_CHROM', number=1, type='String', description="Target chromosome for a dispersed duplication or translocation")
    header.info.add('FRAGMENT_INTERVALS', number=1, type='String',
                    description="Fragment intervals for SV components; per grammar, in GRAMMAR order and separated by |")
    header.info.add('GRAMMAR', number=1, type='String',
                    description="Grammar of the structural variant; in single-line records, alternative grammars separated by |")
    header.info.add('GRAMMAR_ID', number=1, type='Integer',
                    description="Grammar this record belongs to, for an SV with alternative grammars of the same breakpoints")
    header.formats.add('GT', number=1, type='String', description="Genotype")
    header.add_sample("SAMPLE")

    vcf_out_file = VariantFile(output_path, 'w', header=header)
    for parent_id, sv in enumerate(filtered_svs):
        vcf_info_dict = sv.get_vcf_info()
        ambiguous = len(sv.grammars) > 1
        if sep_frag_recs:
            # Multi-record SV broken into fragments; every grammar of an SV with alternatives, tagged GRAMMAR_ID
            for grammar_id, (grammar, fragments) in enumerate(sv.grammars):
                for frag_id, frag in enumerate(fragments):
                    # Reorder fragment start and end if tolerance allowed stitching of misaligned edges
                    frag_start, frag_end = sorted([frag.start, frag.end])
                    info_field = {'OP_TYPE': frag.operation,
                                  'SVLEN': frag_end - frag_start, # NOTE: induces pysam bug where end position is pushed by +1 when a predefined ALT is recognized
                                  'GRAMMAR': grammar, 'SVID': f"sv{parent_id}",
                                  'SVTYPE': vcf_info_dict['sv_type']}
                    if frag.target_start:
                        info_field.update({'TARGET': frag.target_start, 'TARGET_CHROM': frag.target_chrom})
                    if frag.insord is not None:
                        info_field['INSORD'] = frag.insord
                    if ambiguous:
                        info_field['GRAMMAR_ID'] = grammar_id

                    vcf_record = vcf_out_file.header.new_record(contig=frag.chrom,
                                                                start=frag_start,
                                                                stop=frag_end,
                                                                alleles=['N', '<%s>' % frag.operation],
                                                                id=f"sv{parent_id}_g{grammar_id}_{frag_id}" if ambiguous
                                                                else f"sv{parent_id}_{frag_id}",
                                                                info=info_field,
                                                                qual=vcf_info_dict['qual'], filter='PASS',
                                                                samples=[{'GT': vcf_info_dict['gt']}])
                    vcf_out_file.write(vcf_record)
        else:
            # Single-line record; position spans every grammar, TARGET only when the grammars share it
            all_fragments = [frag for _, fragments in sv.grammars for frag in fragments]
            start, stop = min(frag.start for frag in all_fragments), max(frag.end for frag in all_fragments)
            info_field = {'SVTYPE': vcf_info_dict['sv_type'], 'SVLEN': stop - start,
                          'GRAMMAR': '|'.join(grammar for grammar, _ in sv.grammars),
                          'SVID': f"sv{parent_id}", 'OP_TYPE': vcf_info_dict['op_type'],}
            if SV_REGISTRY[sv.stitch_class].get("fragment_fn", None):
                info_field['FRAGMENT_INTERVALS'] = '|'.join(
                    ','.join([f'{frag.operation}: {frag.start}-{frag.end} ({frag.target_start}-{frag.target_end})'
                              for frag in fragments])
                    for _, fragments in sv.grammars)
            targets = {frag.target_start for frag in all_fragments if frag.target_start is not None}
            if 'target' in vcf_info_dict and len(targets) == 1:
                info_field['TARGET'] = vcf_info_dict['target']
                info_field['TARGET_CHROM'] = vcf_info_dict['chrom']
            vcf_record = vcf_out_file.header.new_record(contig=vcf_info_dict['chrom'], start=start,
                                                        stop=stop,
                                                        alleles=['N', '<%s>' % vcf_info_dict['sv_type']],
                                                        id=f"sv{parent_id}",
                                                        info=info_field,
                                                        qual=vcf_info_dict['qual'], filter='PASS',
                                                        samples=[{'GT': vcf_info_dict['gt']}])
            vcf_out_file.write(vcf_record)
    vcf_out_file.close()


def diagnostics(input_recs, kept_svs, dropped_svs, dropped_reasons):

    # -- filtering diagnostic by sv type and filter reason, log separately --
    counter_by_sv_type = Counter()
    counter_by_filter_reason = Counter()

    for csv, filter_reason in zip(dropped_svs, dropped_reasons):
        counter_by_sv_type[csv.sv_type] += 1
        counter_by_filter_reason[filter_reason] += 1
    logging.info("CSV filter diagnostics")
    for sv_type in sorted(counter_by_sv_type.keys()):
        filter_count = counter_by_sv_type[sv_type]
        if filter_count > 0:
            logging.info(f"Filtered {counter_by_sv_type[sv_type]} instances of class {sv_type}s")

    for filter_reason in sorted(counter_by_filter_reason.keys()):
        filter_count = counter_by_filter_reason[filter_reason]
        if filter_count > 0:
            logging.info(f"Filtered {counter_by_filter_reason[filter_reason]} SVs due to {filter_reason}")

    # -- edge counter vs sv counter --
    logging.info("Edge classification vs stitched edges diagnostics")
    edge_counter = Counter()
    sv_counter = Counter()

    for rec in input_recs:
        edge_counter[rec.type] += 1

    for csv in kept_svs:
        sv_counter[csv.stitch_class] += 1

    for sv_type in SV_REGISTRY:
        edge_types = list(SV_REGISTRY[sv_type]['candidates'].values())
        edge_string = ", ".join([f'{edge_type}: {edge_counter[edge_type]}' for edge_type in SV_REGISTRY[sv_type]['candidates'].values()])
        
        logging.info(f'-- {sv_type} --')
        logging.info(f'Expected edges: {edge_types}')
        logging.info(f'Num edges: {edge_string}')
        logging.info(f'Num stitched and called after filtering: {sv_counter[sv_type]}.' )

def load_exclude_list(gap_file):
    exclude_list = defaultdict(IntervalTree)
    with open(gap_file, 'r') as f:
        for line in f:
            row = line.strip().split()

            chrom = row[1]
            start, stop = int(row[2]), int(row[3])
            region_type = row[7]

            # print(f"{chrom}\t{start}\t{stop}\t{region_type}")

            # if region_type in ['telomere', 'centromere']:
            # Include all regions in this gap file
            exclude_list[chrom][start:stop] = region_type

    return exclude_list


def generate_merged_callset(input_recs, config):
    """
    Main stitching function.
    :param input_callset: list of SVBreakpoint objects from model predictions
    :param config: DiscoverConfig

    The config object should contain the following fields:

    :param output_filename: output file for final callset
    :param breakpoint_threshold: noise tolerance for breakpoint position motifs (number of BPs to extend thresholds)
    :param sep_frag_recs: whether to separate output records into fragments
    :param unmatched_edge_file: name of file to output unmatched edges
    :param filter_overlaps: whether to filter overlapping edges
    :param gap_file: optional gap file for ambiguous regions of reference genome
    :param minimum_length: minimum length of any fragment to retain
    """

    logging.info(f'Beginning stitching with {len(input_recs)} records')

    input_recs = [rec for rec in input_recs if rec.type != 'NEG']

    if config.gap_file:
        exclude_list = load_exclude_list(config.gap_file)

        cleaned_recs = []
        for rec in input_recs:
            coords = sorted([rec.pos1, rec.pos2])
            if exclude_list[rec.chr1].overlap(coords[0], coords[1]) or \
                    exclude_list[rec.chr2].overlap(coords[0], coords[1]):
                # record is in the excluded regions
                continue

            # record is allowed
            cleaned_recs.append(rec)

        logging.info(f'Found {len(cleaned_recs)} edges outside of excluded regions out of {len(input_recs)}')

        input_recs = cleaned_recs


    merged_svs = find_matches(input_recs, tolerance=config.breakpoint_threshold)
    logging.info(f'Found {len(merged_svs)} stitched SVs')

    # Filter by size and fragment validity
    if config.complex_sv_unfiltered_file_path:
        export_callset_to_vcf(merged_svs, output_path=config.complex_sv_unfiltered_file_path,
                              contigs=config.fai_index.contigs(), sep_frag_recs=config.sep_frag_recs)

    kept_svs = []
    dropped_svs = []
    dropped_reasons = []

    stitched_filter_logger = config.stitched_filter_logger
    for csv in merged_svs:
        sv_lengths = [frag.end - frag.start for frag in csv.fragments]
        total_length = sum(sv_lengths)
        smallest_fragment = min(sv_lengths)

        filter_reason = None
        if total_length < config.minimum_length:
            stitched_filter_logger.info( f"{csv} total sv length {total_length} <{config.minimum_length}")
            filter_reason = f"total sv length less than {config.minimum_length}"

        elif total_length > config.maximum_length:
            stitched_filter_logger.info( f"{csv} total sv length {total_length} > {config.maximum_length}")
            filter_reason = f"total sv length greater than {config.maximum_length}"

        elif smallest_fragment < config.minimum_fragment_length:
            stitched_filter_logger.info( f"{csv} smallest fragment length {smallest_fragment} < config.minimum_fragment_length {config.minimum_fragment_length}")
            filter_reason = f"smallest fragment length < config.minimum_fragment_length"

        elif not csv.validated:
            stitched_filter_logger.info(f"{csv} failed fragment validation")
            filter_reason = f"csv failed fragment validation"

        if filter_reason:
            dropped_svs.append(csv)
            dropped_reasons.append(filter_reason)
        else:
            # alternative grammars are held to the same length limits
            csv.grammars = [(grammar, fragments) for grammar, fragments in csv.grammars
                            if config.minimum_length <= sum(frag.end - frag.start for frag in fragments) <= config.maximum_length
                            and min(frag.end - frag.start for frag in fragments) >= config.minimum_fragment_length]
            kept_svs.append(csv)

    if config.deduplicate_calls:
        kept_svs, dropped_svs_deduplication, dropped_reasons_deduplication = deduplicate_svs(kept_svs, config.duplication_threshold)
        for dropped_csv in dropped_svs_deduplication:
            stitched_filter_logger.info(f"{dropped_csv} filtered from deduplication")
        dropped_svs.extend(dropped_svs_deduplication)
        dropped_reasons.extend(dropped_reasons_deduplication)

    if config.filter_overlaps:
        kept_svs, recs_dropped_overlaps, dropped_reason_overlaps = filter_overlapping_svs(kept_svs, config.prioritize_evidence)
        for dropped_csv in recs_dropped_overlaps:
            stitched_filter_logger.info(f"{dropped_csv} filtered from overlap filter")
        dropped_svs.extend(recs_dropped_overlaps)
        dropped_reasons.extend(dropped_reason_overlaps)

    export_callset_to_vcf(kept_svs, output_path=config.complex_sv_vcf_file_path,
                          contigs=config.fai_index.contigs(), sep_frag_recs=config.sep_frag_recs)

    diagnostics(input_recs, kept_svs, dropped_svs, dropped_reasons)

    if config.unmatched_edge_file:
        matched_records = set()
        for csv in merged_svs:
            for rec in csv.records.values():
                matched_records.add((rec.chr1, rec.pos1, rec.pos2, rec.type))

        unmatched_recs = []
        for rec in input_recs:
            key = (rec.chr1, rec.pos1, rec.pos2, rec.type)
            if key not in matched_records:
                unmatched_recs.append(rec)

        logging.info(f"{len(unmatched_recs)} edges were not matched by stitching rules")

        io.write_bkp_vcf(config.bkp_unmatched_file_path, unmatched_recs, config.fai_index)
