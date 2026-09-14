from unittest import TestCase
from groovi.seq.stitching import *
import tempfile
from groovi.seq.sv import SVBreakpoint


def _create_sample_recs():
    data = [
        {'id': 'delINVdel_A', 'start': 1000, 'stop': 1100},
        {'id': 'delINVdel_B', 'start': 1050, 'stop': 1400},
        {'id': 'dupINVdup_A', 'start': 1000, 'stop': 1100},
        {'id': 'dupINVdup_B', 'start': 1050, 'stop': 1400},
        {'id': 'dDUP_A', 'start': 111, 'stop': 130},
        {'id': 'dDUP_B', 'start': 110, 'stop': 120},
        {'id': 'dDUP_A', 'start': 10, 'stop': 30},
        {'id': 'dDUP_B', 'start': 20, 'stop': 31},
        {'id': 'INV_dDUP_A', 'start': 10, 'stop': 50},
        {'id': 'INV_dDUP_B', 'start': 11, 'stop': 30},
        {'id': 'DEL', 'start': 10, 'stop': 20},
        {'id': 'nrTRA_A', 'start': 10, 'stop': 100},
        {'id': 'nrTRA_B', 'start': 11, 'stop': 119},
        {'id': 'nrTRA_A', 'start': 99, 'stop': 120},
    ]

    recs = []
    for rec_data in data:
        rec = SVBreakpoint(rec_data['id'],
                           chr1='chr1',
                           pos1=rec_data['start'],
                           chr2='chr1',
                           pos2=rec_data['stop'],
                           qual=1)
        recs.append(rec)
    return recs

class TestStitching(TestCase):
    """
    Test general stitching utility functions
    """

    def test_dispatch_match_functions(self):
        """
        Tests that stitching is correctly executed by the dispatch function that chooses the correct match subroutine
        based on SV type
        Tests for a handful of SV types with A-B edges, interlocking and contained
            Interlocking: delINVdel-A, dupINVdup-A
            contained: 'delINV-A', 'INVdel-A', 'dDUP-A', 'INV_dDUP-A', 'nrTRA-A'
        Tests for singleton records
        """

        recs = _create_sample_recs()

        merged_recs = find_matches(recs)

        for rec in merged_recs:
            print(f'{rec.get_vcf_info()}')

        # Intervals are half-open [start, stop), so sv_len is stop - start for every record
        correct_match_vcf = [
            {'chrom': 'chr1', 'start': 1001, 'stop': 1400, 'sv_type': 'delINVdel', 'qual': 1, 'sv_len': 399,
             'gt': (0, 0), 'op_type': 'delINVdel'},
            {'chrom': 'chr1', 'start': 1000, 'stop': 1401, 'sv_type': 'dupINVdup', 'qual': 1, 'sv_len': 401,
             'gt': (0, 0), 'op_type': 'dupINVdup'},
            {'chrom': 'chr1', 'start': 120, 'stop': 130, 'sv_type': 'dDUP', 'qual': 1, 'sv_len': 10, 'target': 110,
             'gt': (0, 0), 'op_type': 'DUP'},
            {'chrom': 'chr1', 'start': 10, 'stop': 20, 'sv_type': 'dDUP', 'qual': 1, 'sv_len': 10, 'target': 30,
             'gt': (0, 0), 'op_type': 'DUP'},
            {'chrom': 'chr1', 'start': 30, 'stop': 50, 'sv_type': 'INV_dDUP', 'qual': 1, 'sv_len': 20, 'target': 10,
             'gt': (0, 0), 'op_type': 'INV_dDUP'},
            {'chrom': 'chr1', 'start': 100, 'stop': 119, 'sv_type': 'nrTRA', 'qual': 1, 'sv_len': 19, 'target': 10,
             'gt': (0, 0), 'op_type': 'nrTRA'},
            {'chrom': 'chr1', 'start': 11, 'stop': 20, 'sv_type': 'DEL', 'qual': 1, 'sv_len': 9, 'gt': (0, 0), 'op_type': 'CUT'}
        ]

        self.assertEqual(sorted(correct_match_vcf, key=lambda x: (x['start'], x['stop'], x['sv_type'])),
                         sorted([rec.get_vcf_info() for rec in merged_recs], key=lambda x: (x['start'], x['stop'], x['sv_type'])))


    def test_export_to_vcf(self):
        """
        Tests to verify the various paths through this function that depend on SV types and fragment data
        """

        class MockContig:
            """
            Dummy contig clone for testing VCF contig naming
            """
            def __init__(self, name):
                self.name = name

        recs = _create_sample_recs()

        merged_recs = find_matches(recs)

        tmp_path = tempfile.mkstemp(suffix='.vcf')[1]
        export_callset_to_vcf(merged_recs, tmp_path, contigs=[MockContig('chr1')], sep_frag_recs=False)

    def test_filter_merged_recs(self):

        data = [
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 10, 'stop': 100, 'qual': 1000},
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 5, 'stop': 200, 'qual': 200},
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 100, 'stop': 110, 'qual': 1001},
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 108, 'stop': 120, 'qual': 999},
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 1001, 'stop': 2001, 'qual': 10},
            {'chrom': 'chr2', 'sv_type': 'DEL', 'start': 10, 'stop': 100, 'qual': 1002},
        ]

        recs = []
        for rec_data in data:
            csv = ComplexSV(
                sv_type=rec_data['sv_type'],
                chrom=rec_data['chrom'],
                records=[SVBreakpoint(sv_type=rec_data['sv_type'],
                                      chr1=rec_data['chrom'],
                                      pos1=rec_data['start'],
                                      chr2=rec_data['chrom'],
                                      pos2=rec_data['stop'],
                                      qual=rec_data['qual']
                                      )],
                qual=rec_data['qual'],
            )
            recs.append(csv)

        # Add a dispersion that conflicts with the last deletion
        tra_recs = []
        rec = SVBreakpoint(sv_type='dDUP_A', chr1='chr2', pos1=1, chr2='chr2', pos2=50, qual=2000)
        tra_recs.append(rec)

        rec = SVBreakpoint(sv_type='dDUP_B', chr1='chr2', pos1=20, chr2='chr2', pos2=51, qual=2000)

        tra_recs.append(rec)
        csv = ComplexSV.from_records(tra_recs)
        recs.append(csv)

        filtered_recs = filter_merged_recs(recs)

        filtered_quals = set([rec.qual for rec in filtered_recs])

        expected_quals = {1000, 1001, 10, 2000}

        self.assertSetEqual(expected_quals, filtered_quals)


class TestComplexSV(TestCase):
    """
    Test ComplexSV class functionality
    Tests will cover expected logic for different SV types. As we refactor, existing functionality should be preserved
    """

    def test_get_bed_info(self):
        """
        Test that bed information is correctly constructed for various SV types
        Checks for correct name conversion
        Only tests data population and correct dispersion computation for one type.
        Individual SV types are tested separately
        """
        csv = ComplexSV(
            sv_type="DUP",
            chrom='chr1', qual=60,
            records=[SVBreakpoint(sv_type='DUP', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=60)]
        )
        bed_info = csv.get_bed_info()
        assert len(bed_info) == 1
        self.assertEqual('chr1', bed_info[0]['chrom'])
        self.assertEqual(5, bed_info[0]['source_s'])
        self.assertEqual(16, bed_info[0]['source_e'])
        self.assertEqual(None, bed_info[0]['target_s'])
        self.assertEqual(None, bed_info[0]['target_e'])
        self.assertEqual('DUP', bed_info[0]['transform'])
        # [5, 16) spans the 11 duplicated bases 5..15
        self.assertEqual(11, bed_info[0]['sv_len'])
        self.assertEqual('DUP', bed_info[0]['sv_type'])

        csv = ComplexSV(
            sv_type="nrTRA",
            chrom='chr1', qual=1.0,
            records=[SVBreakpoint(sv_type='nrTRA_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=10),
                     SVBreakpoint(sv_type='nrTRA_B', chr1='chr1', pos1=6, chr2='chr1', pos2=20, qual=10),
                     SVBreakpoint(sv_type='nrTRA_C', chr1='chr1', pos1=14, chr2='chr1', pos2=21, qual=10),],
        )
        bed_info = csv.get_bed_info()
        self.assertEqual('chr1', bed_info[0]['chrom'])
        self.assertEqual(15, bed_info[0]['source_s'])
        self.assertEqual(20, bed_info[0]['source_e'])
        self.assertEqual(5, bed_info[0]['target_s'])
        self.assertEqual(6, bed_info[0]['target_e'])
        self.assertEqual('CUT-PASTE', bed_info[0]['transform'])
        self.assertEqual(5, bed_info[0]['sv_len'])
        self.assertEqual('nrTRA', bed_info[0]['sv_type'])



    def test_get_vcf_info(self):
        """
        Tests that the correct VCF records are generated for various SV types.
        Singleton SV (DEL)
        Dispersion SVs (nrTRA, dDUP, INV_dDUP)
        Contained SVs (delINVdel, dupINVdup, delINV, INVdel)
        delINV and INVdel check for high confidence intervals
        """

        csv = ComplexSV(
            sv_type='DEL',
            chrom=1, qual=40,
            records=[SVBreakpoint(sv_type='DEL', pos1=5, chr1='chr1', pos2=15, chr2='chr1', qual=40)]
        )

        vcf_info = csv.get_vcf_info()

        self.assertEqual(1, vcf_info['chrom'])
        self.assertEqual(6, vcf_info['start'])
        self.assertEqual(15, vcf_info['stop'])
        self.assertEqual('DEL', vcf_info['sv_type'])
        self.assertEqual(40, vcf_info['qual'])
        self.assertEqual(9, vcf_info['sv_len'])

        # Test dispersion SVs
        for dispersion_sv in ['dDUP', 'INV_dDUP']:

            csv = ComplexSV(
                sv_type=dispersion_sv,
                chrom=1, qual=0.9,
                records=[SVBreakpoint(sv_type=f'{dispersion_sv}_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=1),
                         SVBreakpoint(sv_type=f'{dispersion_sv}_B', chr1='chr1', pos1=6, chr2='chr2', pos2=20, qual=1)],
            )

            vcf_info = csv.get_vcf_info()
            self.assertEqual(15, vcf_info['start'])
            self.assertEqual(20, vcf_info['stop'])
            self.assertEqual(dispersion_sv, vcf_info['sv_type'])

        # Test contained SVs

        csv = ComplexSV(
            sv_type="delINVdel", chrom=1, qual=0.9,
            records=[SVBreakpoint(sv_type=f'delINVdel_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=1),
                     SVBreakpoint(sv_type=f'delINVdel_B', chr1='chr1', pos1=6, chr2='chr1', pos2=20, qual=1)],
        )
        vcf_info = csv.get_vcf_info()
        self.assertEqual(6, vcf_info['start'])
        self.assertEqual(20, vcf_info['stop'])


    def test_prioritize_evidence(self):
        """
        Test the toggle for filtering overlapping stitched SVs by .qual or by .evidence
        """
        overlapping_data = [
            [SVBreakpoint(sv_type='DEL', chr1='chr1', pos1=5, chr2='chr1', pos2=400, qual=30, gt=(None, None), evidence=20)],
            [SVBreakpoint(sv_type='INV_A', chr1='chr1', pos1=9, chr2='chr1', pos2=300, qual=10, gt=(None, None), evidence=30),
             SVBreakpoint(sv_type='INV_B', chr1='chr1', pos1=10, chr2='chr1', pos2=301, qual=10, gt=(None, None),
                          evidence=30)],
            [SVBreakpoint(sv_type='DUP', chr1='chr1', pos1=401, chr2='chr1', pos2=1000, qual=40, gt=(None, None), evidence=3)],
            [SVBreakpoint(sv_type='INVdel_A', chr1='chr1', pos1=600, chr2='chr1', pos2=2000, qual=20, evidence=5),
             SVBreakpoint(sv_type='INVdel_B', chr1='chr1', pos1=601, chr2='chr1', pos2=3000, qual=25, evidence=7)],
            [SVBreakpoint(sv_type='dupINVdel_A', chr1='chr2', pos1=10, chr2='chr2', pos2=800, qual=0, evidence=0),
             SVBreakpoint(sv_type='dupINVdel_B', chr1='chr2', pos1=20, chr2='chr2', pos2=600, qual=0, evidence=0)],
        ]

        csvs = [ComplexSV.from_records(bkps) for bkps in overlapping_data]

        filtered_csvs = sorted(filter_merged_recs(csvs, prioritize_evidence=False), key=lambda x: (x.chrom, x.sv_type))
        expected_result_qual = [csvs[0], csvs[2], csvs[4]]
        self.assertListEqual(filtered_csvs, expected_result_qual)

        filtered_csvs = sorted(filter_merged_recs(csvs, prioritize_evidence=True), key=lambda x: (x.chrom, x.sv_type))
        expected_result_evidence = [csvs[1], csvs[3], csvs[4]]
        self.assertListEqual(filtered_csvs, expected_result_evidence)

class TestSVTypes(TestCase):

    def run_pipeline(self, input_recs, expected_calls, expected_fragments, expected_grammars, tolerance=0):
        """
        Generic test to run stitching on a set of input records, checking that the correct calls are returned.

        :param input_recs: list of dicts containing fields 'start', 'stop', 'id', 'qual', and optional 'evidence'
        :param expected_calls: list of dicts containing fields 'start', 'stop', 'sv_type', 'sv_len',
                                'target', and info 'fragments'
                Expected bp locations are zero indexed for both the input_recs and expected_calls,
                and this function will convert them to 1-indexed for comparison with VCF format output
        :param tolerance: breakpoint tolerance for matching, needed to stitch edges with imprecise coordinates
        """
        recs = []
        for rec_data in input_recs:
            rec = SVBreakpoint(sv_type=rec_data['id'],
                               chr1='chr1',
                               pos1=rec_data['start'],
                               chr2='chr1',
                               pos2=rec_data['stop'],
                               qual=rec_data['qual'],
                               evidence=rec_data.get('evidence'))

            recs.append(rec)

        merged_recs = find_matches(recs, tolerance=tolerance)
        # filtered_recs = filter_merged_recs(merged_recs)
        print(merged_recs)

        keys_to_compare = {'start', 'stop', 'sv_type', 'sv_len', 'target', 'info', 'fragments'}

        merged_rec_dicts = [rec.get_vcf_info() for rec in merged_recs]
        pruned_merged_dicts = [
            {key: d[key] for key in keys_to_compare if key in d}
            for d in merged_rec_dicts
        ]

        print(f"Process returned {pruned_merged_dicts}, expected {expected_calls}")

        self.assertEqual(sorted(expected_calls, key=lambda x: (x['start'], x['stop'], x['sv_type'])),
                         sorted(pruned_merged_dicts, key=lambda x: (x['start'], x['stop'], x['sv_type'])))

        all_fragments = [frag for rec in merged_recs for frag in rec.fragments]
        check_values = [(frag.start, frag.end, frag.target_start, frag.target_end, frag.operation) for frag in all_fragments]
        self.assertEqual(frozenset(expected_fragments), frozenset(check_values))

        self.assertEqual(sorted(expected_grammars), sorted([rec.grammar for rec in merged_recs]))

    def test_del(self):
        """
        DEL is a singleton record directly linking the start of the deletion to the end
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DEL'}]
        expected_result = [{'start': 6, 'stop': 15, 'sv_type': 'DEL', 'sv_len': 9}]
        expected_fragments = [
            (6, 15, None, None, 'CUT')
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A->'])

    def test_inv(self):
        """
        INV is an interlocking pair adjacent on both sides
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'INV_A'},
                {'start': 6, 'stop': 16, 'qual': 0.9, 'id': 'INV_B'}]
        expected_result = [{'start': 6, 'stop': 16, 'sv_type': 'INV', 'sv_len': 10}]
        expected_fragments = [
            (6, 16, None, None, 'INV')
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A->a'])

    def test_dup(self):
        """
        DUP is a singleton record indicating the region that is repeated once
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DUP'}]
        expected_result = [{'start': 5, 'stop': 16, 'sv_type': 'DUP', 'sv_len': 11}]
        expected_fragments = [
            (5, 16, None, None, 'DUP')
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A->AA'])

    def test_dup_inv(self):
        """
        DUP-INV has a structure where each SV should have two interlocking A edges and a contained B edge that has
        shared start and end with each other edge. The interlocking edges should be off by one.
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DUP_INV_A'},
                {'start': 6, 'stop': 16, 'qual': 0.9, 'id': 'DUP_INV_C'},
                {'start': 6, 'stop': 15, 'qual': 0.9, 'id': 'DUP_INV_B'},
                ]
        expected_result = [{'start': 6, 'stop': 16, 'sv_type': 'DUP_INV', 'sv_len': 10}]
        expected_fragments = [
            (6, 16, None, None, 'INV'),
            (6, 16, 16, 17, 'COPYinv-PASTE')
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A->aa'])

    def test_ddup(self):
        """
        dDUP has an A type record that interlocks with a B record.
        We should handle forward and backward duplications.
        The duplication will be inserted into a location with starts or ends that are off by one
        """
        data = [
            {'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'dDUP_A'},
            {'start': 6, 'stop': 20, 'qual': 0.9, 'id': 'dDUP_B'},
            {'start': 1, 'stop': 30, 'qual': 0.9, 'id': 'dDUP_A'},
            {'start': 7, 'stop': 31, 'qual': 0.9, 'id': 'dDUP_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 7, 'sv_type': 'dDUP', 'target': 30, 'sv_len': 6},
            {'start': 15, 'stop': 20, 'sv_type': 'dDUP', 'target': 5, 'sv_len': 5},
        ]
        expected_fragments = [
            (1, 7, 30, 31, 'COPY-PASTE'),
            (15, 20, 5, 6, 'COPY-PASTE'),
        ]

        self.run_pipeline(data, expected_result, expected_fragments, ['A_->A_A', '_A->A_A'])

    def test_inv_ddup(self):
        """
        INV-dDUP has an A type record that envelops a B-type record, and either their starts or ends are adjacent
        """

        data  = [
            {'start': 1, 'stop': 10, 'qual': 100, 'id': 'INV_dDUP_A'},
            {'start': 2, 'stop': 6, 'qual': 100, 'id': 'INV_dDUP_B'},
            {'start': 15, 'stop': 40, 'qual': 100, 'id': 'INV_dDUP_A'},
            {'start': 17, 'stop': 39, 'qual': 100, 'id': 'INV_dDUP_B'},
        ]
        expected_result = [
            {'start': 6, 'stop': 10, 'sv_type': 'INV_dDUP', 'target': 1, 'sv_len': 4},
            {'start': 15, 'stop': 17, 'sv_type': 'INV_dDUP', 'target': 39, 'sv_len': 2},
        ]
        expected_fragments = [
            (6, 10, 1, 2, 'COPYinv-PASTE'),
            (15, 17, 39, 40, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['_A->a_A', 'A_->A_a'])

    def test_ddup_inv(self):
        """
        dDUP_INV has four record types A, B, C, D, with
            A and B interlocking adjacent on both sides,
            C and D enveloping adjacent on one side, and
            C and B share a start or an end
        """
        data = [
            {'start': 1, 'stop': 50, 'qual': 100, 'id': 'dDUP_INV_A'},
            {'start': 2, 'stop': 51, 'qual': 100, 'id': 'dDUP_INV_B'},
            {'start': 2, 'stop': 100, 'qual': 100, 'id': 'dDUP_INV_C'},
            {'start': 50, 'stop': 99, 'qual': 100, 'id': 'dDUP_INV_D'},
            {'start': 150, 'stop': 199, 'qual': 100, 'id': 'dDUP_INV_A'},
            {'start': 151, 'stop': 200, 'qual': 100, 'id': 'dDUP_INV_B'},
            {'start': 101, 'stop': 199, 'qual': 100, 'id': 'dDUP_INV_C'},
            {'start': 102, 'stop': 151, 'qual': 100, 'id': 'dDUP_INV_D'},
        ]
        expected_result = [
            {'start': 2, 'stop': 50, 'sv_type': 'dDUP_INV', 'target': 99, 'sv_len': 48},
            {'start': 151, 'stop': 199, 'sv_type': 'dDUP_INV', 'target': 101, 'sv_len': 48},
        ]
        expected_fragments = [
            (2, 50, None, None, 'INV'),
            (2, 50, 99, 100, 'COPYinv-PASTE'),
            (151, 199, None, None, 'INV'),
            (151, 199, 101, 102, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A_->a_a', '_A->a_a'])

    def test_nrtra(self):
        """
        nrTRA has three records. Two A intervals interlock with an overlap of one, and a B interval is enclosed
        and adjacent to the far endpoints of the A pair
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'nrTRA_A'},
            {'start': 2, 'stop': 59, 'qual': 100, 'id': 'nrTRA_B'},
            {'start': 19, 'stop': 60, 'qual': 100, 'id': 'nrTRA_A'},
            {'start': 101, 'stop': 120, 'qual': 100, 'id': 'nrTRA_A'},
            {'start': 102, 'stop': 124, 'qual': 100, 'id': 'nrTRA_B'},
            {'start': 119, 'stop': 125, 'qual': 100, 'id': 'nrTRA_A'},
        ]
        expected_result = [
            {'start': 2, 'stop': 19, 'sv_type': 'nrTRA', 'target': 59, 'sv_len': 17},
            {'start': 120, 'stop': 124, 'sv_type': 'nrTRA', 'target': 101, 'sv_len': 4},
        ]
        expected_fragments = [
            (2, 19, 59, 60, 'CUT-PASTE'),
            (120, 124, 101, 102, 'CUT-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['A_->_A', '_A->A_'])

    def test_inv_nrtra(self):
        """
        INV_nrTRA has an A edge that surrounds the SV region, defined by B and C start or end edges pointing to a target
        Can transpose forward or backward
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INV_nrTRA_A'},
            {'start': 2, 'stop': 30, 'qual': 100, 'id': 'INV_nrTRA_B'},
            {'start': 19, 'stop': 29, 'qual': 100, 'id': 'INV_nrTRA_C'},
            {'start': 200, 'stop': 400, 'qual': 100, 'id': 'INV_nrTRA_A'},
            {'start': 150, 'stop': 399, 'qual': 100, 'id': 'INV_nrTRA_B'},
            {'start': 151, 'stop': 201, 'qual': 100, 'id': 'INV_nrTRA_C'},
        ]
        expected_result = [
            {'start': 2, 'stop': 19, 'sv_type': 'INV_nrTRA', 'target': 29, 'sv_len': 17},
            {'start': 201, 'stop': 399, 'sv_type': 'INV_nrTRA', 'target': 150, 'sv_len': 198},
        ]
        expected_fragments = [
            (2, 19, 29, 30, 'CUTinv-PASTE'),
            (201, 399, 150, 151, 'CUTinv-PASTE'),
        ]

        self.run_pipeline(data, expected_result, expected_fragments, ['A_->_a', '_A->a_'])

    def test_invdel(self):
        """
        INVdel has A and B interlocking pairs, with B before A. The section region is inverted, and the subsequent
        region is deleted
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdel_A', 'evidence': 30},
            {'start': 2, 'stop': 30, 'qual': 100, 'id': 'INVdel_B', 'evidence': 10},
        ]
        expected_result = [
            {'start': 2, 'stop': 31, 'sv_type': 'INVdel', 'sv_len': 29},
        ]
        expected_fragments = [
            (2, 21, None, None, 'INV'),
            (21, 31, None, None, 'CUT')
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['AB->a'])

    def test_delinv(self):
        """
        delINV has A and B interlocking adjacent pairs, with A before B. The second region is inverted,
        and the preceding region is deleted
        """
        data = [
            {'start': 1, 'stop': 29, 'qual': 100, 'id': 'delINV_A', 'evidence': 30},
            {'start': 15, 'stop': 30, 'qual': 100, 'id': 'delINV_B', 'evidence': 10},
        ]
        expected_result = [
            {'start': 1, 'stop': 30, 'sv_type': 'delINV', 'sv_len': 29},
        ]
        expected_fragments = [
            (1, 16, None, None, 'CUT'),
            (16, 30, None, None, 'INV'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['AB->b'])

    def test_invdup(self):
        """
        INVdup has A enclosing B, adjacent on the left, with all of A being a variant,
        but has fragments that need to be checked.
        """
        # Exact coordinates: both edges agree on the left boundary, so evidence does not change the layout
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdup_A', 'evidence': 30},
            {'start': 2, 'stop': 10, 'qual': 100, 'id': 'INVdup_B', 'evidence': 10},
        ]
        expected_result = [
            {'start': 2, 'stop': 21, 'sv_type': 'INVdup', 'sv_len': 19},
        ]
        expected_fragments = [
            (2, 10, None, None, 'INV'),
            (10, 21, 2, 3, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['AB->baB'])

        # B's start is imprecise. With A better supported, A.pos1 + 1 recovers the true left boundary at 2
        noisy_data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdup_A', 'evidence': 30},
            {'start': 5, 'stop': 10, 'qual': 100, 'id': 'INVdup_B', 'evidence': 10},
        ]
        self.run_pipeline(noisy_data, expected_result, expected_fragments, ['AB->baB'], tolerance=3)

        # Same imprecise edges, but now B is better supported, so its start defines the boundary instead
        noisy_data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdup_A', 'evidence': 10},
            {'start': 5, 'stop': 10, 'qual': 100, 'id': 'INVdup_B', 'evidence': 30},
        ]
        expected_result = [
            {'start': 5, 'stop': 21, 'sv_type': 'INVdup', 'sv_len': 16},
        ]
        expected_fragments = [
            (5, 10, None, None, 'INV'),
            (10, 21, 5, 6, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(noisy_data, expected_result, expected_fragments, ['AB->baB'], tolerance=3)

    def test_dupINV(self):
        """
        dupINV has A enclosing B adjacent on the right, with all of A being a variant,
        but has fragments that need to be checked.
        """
        # Exact coordinates: both edges agree on the right boundary, so evidence does not change the layout
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINV_A', 'evidence': 30},
            {'start': 10, 'stop': 19, 'qual': 100, 'id': 'dupINV_B', 'evidence': 10},
        ]
        expected_result = [
            {'start': 1, 'stop': 20, 'sv_type': 'dupINV', 'sv_len': 19},
        ]
        expected_fragments = [
            (1, 11, 20, 21, 'COPYinv-PASTE'),
            (11, 20, None, None, 'INV'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['AB->Aba'])

        # A's end is imprecise. With B better supported, B.pos2 + 1 recovers the true right boundary at 20
        noisy_data = [
            {'start': 1, 'stop': 23, 'qual': 100, 'id': 'dupINV_A', 'evidence': 10},
            {'start': 10, 'stop': 19, 'qual': 100, 'id': 'dupINV_B', 'evidence': 30},
        ]
        self.run_pipeline(noisy_data, expected_result, expected_fragments, ['AB->Aba'], tolerance=3)

        # Same imprecise edges, but now A is better supported, so its end defines the boundary instead
        noisy_data = [
            {'start': 1, 'stop': 23, 'qual': 100, 'id': 'dupINV_A', 'evidence': 30},
            {'start': 10, 'stop': 19, 'qual': 100, 'id': 'dupINV_B', 'evidence': 10},
        ]
        expected_result = [
            {'start': 1, 'stop': 23, 'sv_type': 'dupINV', 'sv_len': 22},
        ]
        expected_fragments = [
            (1, 11, 23, 24, 'COPYinv-PASTE'),
            (11, 23, None, None, 'INV'),
        ]
        self.run_pipeline(noisy_data, expected_result, expected_fragments, ['AB->Aba'], tolerance=3)

    def test_dupINVdel(self):
        """
        dupINVdel has A enclosing B, with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINVdel_A'},
            {'start': 5, 'stop': 15, 'qual': 100, 'id': 'dupINVdel_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 20, 'sv_type': 'dupINVdel', 'sv_len': 19},
        ]
        expected_fragments = [
            (1, 6, 16, 17, 'COPYinv-PASTE'),
            (6, 16, None, None, 'INV'),
            (16, 20, None, None, 'CUT'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['ABC->Aba'])


    def test_delINVdup(self):
        """
        dupINVdel has A enclosing B, with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'delINVdup_A'},
            {'start': 5, 'stop': 15, 'qual': 100, 'id': 'delINVdup_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 21, 'sv_type': 'delINVdup', 'sv_len': 19},
        ]
        expected_fragments = [
            (2, 5, None, None, 'CUT'),
            (5, 15, None, None, 'INV'),
            (15, 21, 2, 3, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['ABC->cbC'])

    def test_delINVdel(self):
        """
        delINVdel has A interlocking with another A with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'delINVdel_A'},
            {'start': 5, 'stop': 25, 'qual': 100, 'id': 'delINVdel_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 25, 'sv_type': 'delINVdel', 'sv_len': 23},
        ]
        expected_fragments = [
            (2, 5, None, None, 'CUT'),
            (5, 21, None, None, 'INV'),
            (21, 25, None, None, 'CUT'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['ABC->b'])

    def test_dupINVdup(self):
        """
        dupINVdup has A interlocking with another A with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINVdup_A'},
            {'start': 5, 'stop': 25, 'qual': 100, 'id': 'dupINVdup_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 26, 'sv_type': 'dupINVdup', 'sv_len': 25},
        ]
        expected_fragments = [
            (1, 6, 20, 21, 'COPYinv-PASTE'),
            (6, 20, None, None, 'INV'),
            (20, 26, 6, 7, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments, ['ABC->AcbaC'])
