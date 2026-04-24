from pysam import VariantFile, VariantRecord
from collections import defaultdict, Counter
from intervaltree import Interval, IntervalTree
import pysam
import re
import argparse
import math

from groovi.cli.config import DiscoverConfig
from groovi.seq.sv import SVBreakpoint, Fragment
import groovi.seq.io as io
import logging

from typing import Callable, Any, List, Tuple, Dict, Iterable
import warnings

from groovi.seq.stitch_rules import SV_REGISTRY, dispersion_source_target


class ComplexSV:
    def __init__(self, sv_type, chrom, qual=None, records: List[SVBreakpoint] = None):
        self.sv_type = sv_type
        self.chrom = chrom
        self.qual = qual
        self.records = records

        sv_registry_info = SV_REGISTRY[sv_type]

        # dispersion logic
        if sv_registry_info.get('dispersion', False):
            self.source_start, self.source_end, self.target, self.target_chrom = dispersion_source_target(self)
            self.grammar = sv_registry_info['grammar'][0] if self.target < self.source_start else \
                sv_registry_info['grammar'][1]
            # For dispersions where the grammar can change based on direction of dispersion,
            # we store the left and right grammars as a pair
        else:
            self.source_start, self.source_end, self.target, self.target_chrom = None, None, None, None
            self.grammar = sv_registry_info['grammar']

        self.fragments = sv_registry_info['fragment_fn'](self)


    def __str__(self):
        return repr((self.sv_type, self.chrom, self.qual, self.records))

    @staticmethod
    def from_records(recs: Iterable[SVBreakpoint]):
        sv_type = re.sub("_\w$", "", recs[0].type)
        chrom = recs[0].chr1
        qual = round(sum([rec.qual for rec in recs]) / len(recs))

        return ComplexSV(sv_type, chrom, qual, recs)

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

    def get_vcf_info(self):
        """
        Creates a single VCF record for the SV.
        """
        start, stop = None, None
        start_stop_fn = SV_REGISTRY[self.sv_type].get('start_stop_fn', None)
        if start_stop_fn:
            start, stop = start_stop_fn(self)

        vcf_info = {'chrom': self.chrom,
                    'start': start,
                    'stop': stop,
                    'sv_type': self.sv_type,
                    'qual': self.qual,
                    'sv_len': stop - start,
                    'gt': (0, 0), 'op_type': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type)}

        if self.target:
            vcf_info['target'] = self.target

        return vcf_info

def find_matches(recs: List[Any], tolerance: int=0) -> List[Any]:
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
        # Assumes a naming convention with SV type followed by underscore identifier, e.g., 'dDUP_A'
        sv_type_key = re.sub("_\w$", "", rec.type)
        if sv_type_key not in SV_REGISTRY:
            warnings.warn(f"Unknown SV type: {sv_type_key}")
            continue

        observed_types.add(sv_type_key)

        rec_type = rec.type

        # Storing the original record in the interval data as a list
        trees[rec_type].add(Interval(rec.pos1, rec.pos2, rec))

    all_matches = []

    for sv_type, sv_info in SV_REGISTRY.items():
        if sv_type not in observed_types:
            continue

        print(f"Finding matches for {sv_type}")

        candidates = []
        for rec_type in sv_info.get('candidates', []):
            if isinstance(rec_type, str):
                candidates.append(trees[rec_type])
            elif isinstance(rec_type, tuple):
                candidate_trees = [trees[subtype] for subtype in rec_type]
                candidates.append(IntervalTree(interval for tree in candidate_trees for interval in tree))

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

        csvs = [ComplexSV.from_records(recs) for recs in matches]
        all_matches.extend(csvs)

    return all_matches


def filter_merged_recs(merged_recs: List[ComplexSV], prioritize_evidence: bool=False):
    # filter the list of ComplexSV objects created by dispatch_match_functions()
    # taking the higher-confidence ComplexSV in the case of any collision
    chrom_intervals = defaultdict(IntervalTree)
    filtered_recs = []

    if prioritize_evidence:
        key_func = lambda csv: -sum([rec.evidence if rec.evidence else 0 for rec in csv.records]) / len(csv.records)
    else:
        key_func = lambda csv: -csv.qual

    for rec in sorted(merged_recs, key=key_func):
        occupied_intervals = [(frag.start, frag.end + 1) for frag in rec.fragments]
        occupied_intervals += [(frag.target_start, frag.target_end) for frag in rec.fragments if frag.target_start]

        if rec.chrom not in chrom_intervals or \
                not any(chrom_intervals[rec.chrom].overlap(*occupied_interval) for occupied_interval in occupied_intervals):
            for occupied_interval in occupied_intervals:
                chrom_intervals[rec.chrom].addi(*sorted(occupied_interval))
            filtered_recs.append(rec)
        else:
            continue
    return filtered_recs


def export_callset_to_vcf(filtered_recs, output_path, contigs, sep_frag_recs=False):
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
                    description="Fragment intervals for SV components")
    header.info.add('GRAMMAR', number=1, type='String', description="Grammar of the structural variant")
    header.formats.add('GT', number=1, type='String', description="Genotype")
    header.add_sample("SAMPLE")

    vcf_out_file = VariantFile(output_path, 'w', header=header)
    for parent_id, rec in enumerate(filtered_recs):
        vcf_info_dict = rec.get_vcf_info()
        grammar = rec.grammar
        if sep_frag_recs:
            # Multi-record SV broken into fragments
            for frag_id, frag in enumerate(rec.fragments):
                # Reorder fragment start and end if tolerance allowed stitching of misaligned edges
                frag_start, frag_end = sorted([frag.start, frag.end])
                info_field = {'OP_TYPE': frag.operation, 'SVLEN': frag_end - frag_start,
                              'GRAMMAR': grammar, 'SVID': f"sv{parent_id}",
                              'SVTYPE': vcf_info_dict['sv_type']}
                if frag.target_start:
                    info_field.update({'TARGET': frag.target_start, 'TARGET_CHROM': frag.target_chrom})

                vcf_record = vcf_out_file.header.new_record(contig=frag.chrom,
                                                            start=frag_start,
                                                            stop=frag_end,
                                                            alleles=['N', '<%s>' % frag.operation],
                                                            id=f"sv{parent_id}_{frag_id}",
                                                            info=info_field,
                                                            qual=vcf_info_dict['qual'], filter='PASS',
                                                            samples=[{'GT': vcf_info_dict['gt']}])
                vcf_out_file.write(vcf_record)
        else:
            # Single-line record
            info_field = {'SVTYPE': vcf_info_dict['sv_type'], 'SVLEN': vcf_info_dict['sv_len'],
                          'GRAMMAR': grammar, 'SVID': f"sv{parent_id}", 'OP_TYPE': vcf_info_dict['op_type'],}
            if SV_REGISTRY[vcf_info_dict['sv_type']].get("fragment_fn", None):
                info_field['FRAGMENT_INTERVALS'] = ','.join(
                    [f'{frag.operation}: {frag.start}-{frag.end} ({frag.target_start}-{frag.target_end})' for frag in rec.fragments])
            if 'target' in vcf_info_dict:
                info_field['TARGET'] = vcf_info_dict['target']
                info_field['TARGET_CHROM'] = vcf_info_dict['chrom']
            vcf_record = vcf_out_file.header.new_record(contig=vcf_info_dict['chrom'], start=vcf_info_dict['start'],
                                                        stop=vcf_info_dict['stop'],
                                                        alleles=['N', '<%s>' % vcf_info_dict['sv_type']],
                                                        id=f"sv{parent_id}",
                                                        info=info_field,
                                                        qual=vcf_info_dict['qual'], filter='PASS',
                                                        samples=[{'GT': vcf_info_dict['gt']}])
            vcf_out_file.write(vcf_record)
    vcf_out_file.close()


def diagnostics(input_recs, filtered_recs):
    edge_counter = defaultdict(Counter)
    sv_counter = Counter()

    for rec in input_recs:
        sv_type = re.sub("_\w$", "", rec.type)
        edge_counter[sv_type][rec.type] += 1

    for csv in filtered_recs:
        sv_counter[csv.sv_type] += 1

    for sv_type in SV_REGISTRY:
        edge_types = SV_REGISTRY[sv_type]['candidates']
        edge_string = ", ".join([f'{edge}: {count}' for (edge, count) in edge_counter[sv_type].items()])

        logging.info(f'{sv_type}\nExpected edges: {edge_types}')
        logging.info(f'Num edges: {edge_string}\nNum stitched and called after filtering: {sv_counter[sv_type]}. \n')

    if filtered_recs:
        logging.info(f"Maximum SV length: {max([max([rec.len for rec in csv.records]) for csv in filtered_recs])}")


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


def generate_merged_callset(input_recs: List[SVBreakpoint], config: DiscoverConfig):
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


    # merged_recs = dispatch_match_functions(input_recs, threshold=breakpoint_threshold)
    merged_recs = find_matches(input_recs, tolerance=config.breakpoint_threshold)

    if config.complex_sv_unfiltered_file_path:
        export_callset_to_vcf(merged_recs, output_path=config.complex_sv_unfiltered_file_path,
                              contigs=config.fai_index.contigs(), sep_frag_recs=config.sep_frag_recs)

    # filter stitched SVs by minimum length

    stitched_small_recs = 0

    filtered_recs = []
    for csv in merged_recs:
        sv_lengths = [frag.end - frag.start for frag in csv.fragments]
        total_length = sum(sv_lengths)

        if total_length >= config.minimum_length:
            filtered_recs.append(csv)
        else:
            stitched_small_recs += 1

    logging.info(f'Removed {stitched_small_recs} stitched SVs with total length below minimum {config.minimum_length}.')

    logging.info(f'Found {len(merged_recs)} stitched SVs')

    if config.filter_overlaps:
        filtered_recs = filter_merged_recs(filtered_recs, config.prioritize_evidence)
        print(f'Filtered to {len(filtered_recs)} stitched SVs')

    export_callset_to_vcf(filtered_recs, output_path=config.complex_sv_vcf_file_path,
                          contigs=config.fai_index.contigs(), sep_frag_recs=config.sep_frag_recs)

    diagnostics(input_recs, filtered_recs)

    if config.unmatched_edge_file:
        matched_records = set()
        for csv in merged_recs:
            for rec in csv.records:
                matched_records.add((rec.chr1, rec.pos1, rec.pos2, rec.type))

        unmatched_recs = []
        for rec in input_recs:
            key = (rec.chr1, rec.pos1, rec.pos2, rec.type)
            if key not in matched_records:
                unmatched_recs.append(rec)

        logging.info(f"{len(unmatched_recs)} edges were not matched by stitching rules")

        io.write_bkp_vcf(config.bkp_unmatched_file_path, unmatched_recs, config.fai_index)
