from pysam import VariantFile, VariantRecord
from collections import defaultdict, Counter
from intervaltree import Interval, IntervalTree
import pysam
import re
import argparse
import math

from groovi.cli.config import DiscoverConfig
from groovi.seq.sv import SVBreakpoint
import groovi.seq.io as io
import logging

from typing import Callable, Any, List, Tuple, Dict, Iterable
import warnings

from groovi.seq.stitch_rules import SV_REGISTRY


class ComplexSV:
    def __init__(self, sv_type, chrom, qual=None, singleton_interval=None, records: List[SVBreakpoint] = None):
        self.sv_type = sv_type
        self.chrom = chrom
        self.qual = qual
        self.singleton_interval = singleton_interval
        self.records = records  # intervals is a list of VariantRecords
        self.first_interval = None
        self.second_interval = None
        if not singleton_interval:
            self._set_key_intervals()
        self.occupied_intervals = None
        self.set_occupied_intervals()

    def __str__(self):
        return repr((self.sv_type, self.chrom, self.qual, self.records))

    @staticmethod
    def from_records(recs: Iterable[SVBreakpoint]):
        sv_type = re.sub("_\w$", "", recs[0].type)
        chrom = recs[0].chr1
        qual = round(sum([rec.qual for rec in recs]) / len(recs))
        singleton_interval = (recs[0].pos1, recs[0].pos2) if len(recs) == 1 else None

        return ComplexSV(sv_type, chrom, qual, singleton_interval, recs)

    @staticmethod
    def check_equality(csv1, csv2):
        return all([
            csv1.sv_type == csv2.sv_type,
            csv1.chrom == csv2.chrom,
            csv1.qual == csv2.qual,
            csv1.singleton_interval == csv2.singleton_interval,
            csv1.records == csv2.records,
            csv1.occupied_intervals == csv2.occupied_intervals
        ])

    def _set_key_intervals(self):
        """
        For backward compatibility: set a_interval, b_interval. For SV types with multiple intervals, we use
        _set_key_intervals() to choose which two are most helpful for subsequent computation.

        The heuristic for choosing the key intervals is we take the earliest position a record and the latest b record.
        """
        a_key, b_key = SV_REGISTRY[self.sv_type].get('key_records', (self.sv_type + '_A', self.sv_type + '_B'))
        a_intervals = [rec for rec in self.records if rec.type == a_key]

        b_intervals = [rec for rec in self.records if rec.type == b_key]

        if a_intervals and b_intervals:
            a_record = sorted(a_intervals, key=lambda x: x.pos1)[0]
            b_record = sorted(b_intervals, key=lambda x: x.pos2, reverse=True)[0]

            self.first_interval = (a_record.pos1, a_record.pos2)
            self.second_interval = (b_record.pos1, b_record.pos2)

    def get_dispersion_src_trg(self):
        """
        Helper fn to get source interval and target locus of dispersion SV
        Uses the stored ab intervals, so the 'key_intervals' parameter should be set in the SV definition
        to enable correct logic here.

        :return: (source start), (source stop), (target start) locations in original genome
        """

        # look for two endpoints that are close together (off by exactly one if exact). The earlier is the target
        start_dist = abs(self.first_interval[0] - self.second_interval[0])
        stop_dist = abs(self.first_interval[1] - self.second_interval[1])

        if start_dist < stop_dist:
            source_start, source_stop = sorted([self.first_interval[1], self.second_interval[1]])
            target = min(self.first_interval[0], self.second_interval[0])
        else:
            source_start, source_stop = sorted([self.first_interval[0], self.second_interval[0]])
            target = min(self.first_interval[1], self.second_interval[1])

        return source_start, source_stop, target

    def set_occupied_intervals(self, buffer=0):
        """
        Get occupied intervals from matched _A/_B records for filtering overlapping records by quality
        """
        if self.singleton_interval:
            self.occupied_intervals = [
                (max(self.singleton_interval[0] - buffer, 0), self.singleton_interval[1] + buffer)]
        else:  # <- A/B recs given
            if SV_REGISTRY[self.sv_type].get('dispersion', False):
                # trg: the end of A that's shared with B
                # src_start: the end of A that != trg
                # src_end: the end of B that != trg
                src_start, src_stop, trg = self.get_dispersion_src_trg()
                self.occupied_intervals = [(max(src_start - buffer, 0), src_stop + buffer),
                                           (max(trg - buffer, 0), trg + 1 + buffer)]
            else:
                # No dispersion, so capture the full range of stored intervals
                self.occupied_intervals = [(max(min([rec.pos1 for rec in self.records]) - buffer, 0),
                                            max([rec.pos2 for rec in self.records]) + buffer)]

    def get_bed_info(self):
        # get the bed record(s) information representing the sv
        bed_recs = []
        # if singleton_interval, there will be one record
        if self.singleton_interval:
            bed_recs.append({
                'chrom': self.chrom, 'source_s': self.singleton_interval[0], 'source_e': self.singleton_interval[1],
                'target_s': self.singleton_interval[0], 'target_e': self.singleton_interval[1],
                'transform': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type),
                'sv_len': self.singleton_interval[1] - self.singleton_interval[0], 'gt': './.',
                'sv_type': self.sv_type
            })
        else:
            # for two-interval SVs need to extract the fragments (with logic specific to different types)
            # -> dispersion SV are given by one bed rec, flanked inversions by multiple
            if SV_REGISTRY[self.sv_type].get('dispersion', False):
                src_start, src_stop, trg = self.get_dispersion_src_trg()
                bed_recs.append({
                    'chrom': self.chrom, 'source_s': src_start, 'source_e': src_stop, 'target_s': trg, 'target_e': trg,
                    'transform': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type), 'sv_len': src_stop - src_start, 'gt': './.',
                    'sv_type': self.sv_type
                })
            else:
                frag_intervals = []

                fragment_fn = SV_REGISTRY[self.sv_type].get('fragment_fn', None)

                if fragment_fn:
                    frag_intervals.extend(fragment_fn(self))

                # add the frag-specific records
                for frag_start, frag_stop, target_start, target_stop, frag_type in frag_intervals:
                    if target_start == -1:
                        target_start = frag_start
                        target_stop = frag_stop
                    bed_recs.append({
                        'chrom': self.chrom, 'source_s': frag_start, 'source_e': frag_stop,
                        'target_s': target_start, 'target_e': target_stop,
                        'transform': frag_type, 'sv_len': frag_stop - frag_start, 'gt': './.',
                        'sv_type': self.sv_type
                    })
        return bed_recs

    def get_vcf_info(self):
        """
        Creates a VCF record for the SV. Converts zero-indexed locations to 1-indexed
        """
        if self.singleton_interval:
            return {'chrom': self.chrom,
                    'start': self.singleton_interval[0] + 1,
                    'stop': self.singleton_interval[1] + 1,
                    'sv_type': self.sv_type, 'qual': self.qual,
                    'sv_len': self.singleton_interval[1] - self.singleton_interval[0],
                    'gt': (0, 0), 'op_type': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type)}
        else:
            if SV_REGISTRY[self.sv_type].get('dispersion', False):
                src_start, src_stop, trg = self.get_dispersion_src_trg()
                return {'chrom': self.chrom,
                        'start': src_start + 1,
                        'stop': src_stop + 1,
                        'sv_type': self.sv_type,
                        'qual': self.qual,
                        'sv_len': src_stop - src_start,
                        'target': trg + 1,
                        'gt': (0, 0), 'op_type': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type)}
            else:
                start, stop = None, None
                start_stop_fn = SV_REGISTRY[self.sv_type].get('start_stop_fn', None)
                if start_stop_fn:
                    start, stop = start_stop_fn(self)

                return {'chrom': self.chrom,
                        'start': start + 1,
                        'stop': stop + 1,
                        'sv_type': self.sv_type,
                        'qual': self.qual,
                        'sv_len': stop - start,
                        'gt': (0, 0), 'op_type': SV_REGISTRY[self.sv_type].get('op_name', self.sv_type)}


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

    trees = {}
    for rec in recs:
        # Assumes a naming convention with SV type followed by underscore identifier, e.g., 'dDUP_A'
        sv_type_key = re.sub("_\w$", "", rec.type)
        if sv_type_key not in SV_REGISTRY:
            warnings.warn(f"Unknown SV type: {sv_type_key}")
            continue

        observed_types.add(sv_type_key)

        rec_type = rec.type
        if rec_type not in trees:
            trees[rec_type] = IntervalTree()

        # Storing the original record in the interval data as a list
        trees[rec_type].add(Interval(rec.pos1, rec.pos2, rec))

    all_matches = []

    for sv_type, sv_info in SV_REGISTRY.items():
        if sv_type not in observed_types:
            continue

        print(f"Finding matches for {sv_type}")

        candidates = [trees.get(rec_type, IntervalTree()) for rec_type in sv_info.get('candidates', [])]
        match_fn = sv_info.get('match_fn', None)
        if not candidates:
            continue

        matches = match_fn(candidates, tolerance)
        csvs = [ComplexSV.from_records(recs) for recs in matches]
        all_matches.extend(csvs)

    return all_matches


def filter_merged_recs(merged_recs: List[ComplexSV], prioritize_evidence: bool=False):
    # filter the list of ComplexSV objects created by dispatch_match_functions()
    # taking the higher-confidence ComplexSV in the case of any collision
    chrom_intervals = defaultdict(IntervalTree)
    filtered_recs = []

    if prioritize_evidence:
        key_func = lambda csv: -sum([rec.evidence for rec in csv.records]) / len(csv.records)
    else:
        key_func = lambda csv: -csv.qual

    for rec in sorted(merged_recs, key=key_func):
        if rec.chrom not in chrom_intervals or \
                not any(chrom_intervals[rec.chrom].overlap(*occupied_interval) for occupied_interval in rec.occupied_intervals):
            for occupied_interval in rec.occupied_intervals:
                chrom_intervals[rec.chrom].addi(*occupied_interval)
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
        frag_info_dicts = rec.get_bed_info()
        grammar = SV_REGISTRY[vcf_info_dict['sv_type']]['grammar']
        if SV_REGISTRY[vcf_info_dict['sv_type']].get("dispersion", False):
            # Single-record dispersion. Should have no extra fragments
            assert len(frag_info_dicts) == 1
            info_field = {'SVTYPE': vcf_info_dict['sv_type'], 'SVLEN': vcf_info_dict['sv_len'],
                          'GRAMMAR': grammar, 'SVID': f"sv{parent_id}", 'OP_TYPE': vcf_info_dict['op_type'],
                          'TARGET': vcf_info_dict['target'], 'TARGET_CHROM': vcf_info_dict['chrom'],}
            # TODO: Add interchromosomal target support. Currently assumes same chromosome
            vcf_record = vcf_out_file.header.new_record(contig=vcf_info_dict['chrom'],
                                                        start=vcf_info_dict['start'],
                                                        stop=vcf_info_dict['stop'],
                                                        alleles=['N', '<%s>' % vcf_info_dict['sv_type']],
                                                        id=info_field['SVID'],
                                                        info=info_field,
                                                        qual=vcf_info_dict['qual'], filter='PASS',
                                                        samples=[{'GT': vcf_info_dict['gt']}])
            vcf_out_file.write(vcf_record)

        elif SV_REGISTRY[vcf_info_dict['sv_type']].get("fragment_fn", None) and sep_frag_recs:
            # Multi-record SV broken into fragments
            assert len(frag_info_dicts) > 1
            for frag_id, frag in enumerate(frag_info_dicts):
                # Reorder fragment start and end if tolerance allowed stitching of misaligned edges
                frag_start, frag_end = sorted([frag['source_s'], frag['source_e']])
                info_field = {'OP_TYPE': frag['transform'], 'SVLEN': frag_end - frag_start,
                              'GRAMMAR': grammar, 'SVID': f"sv{parent_id}",
                              'SVTYPE': vcf_info_dict['sv_type'], 'TARGET': frag['target_s'] + 1, 'TARGET_CHROM': frag['chrom']}
                if frag['target_e'] == frag['source_e']:
                    del info_field['TARGET'], info_field['TARGET_CHROM']
                vcf_record = vcf_out_file.header.new_record(contig=vcf_info_dict['chrom'],
                                                            start=frag_start + 1,
                                                            stop=frag_end + 1,
                                                            alleles=['N', '<%s>' % frag['transform']],
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
                    ['%s_%s-%s' % (frag['transform'], frag['source_s'], frag['source_e']) for frag in frag_info_dicts])
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

    filtered_input_recs = []
    small_records = 0
    for rec in input_recs:
        if rec.len < config.minimum_length:
            small_records += 1
        else:
            filtered_input_recs.append(rec)

    logging.info(f'Found {small_records} records less than minimum length {config.minimum_length}. {len(input_recs)} records remain.')

    input_recs = filtered_input_recs

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
        bed_recs = csv.get_bed_info()
        if all([bed['sv_len'] > config.minimum_length for bed in bed_recs]):
            filtered_recs.append(csv)
        else:
            stitched_small_recs += 1
    logging.info(f'Removed {stitched_small_recs} stitched SVs with fragments below minimum length {config.minimum_length}.')

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
