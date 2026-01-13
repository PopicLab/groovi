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

        correct_match_vcf = [
            {'chrom': 'chr1', 'start': 1002, 'stop': 1400, 'sv_type': 'delINVdel', 'qual': 1, 'sv_len': 398,
             'gt': (0, 0), 'op_type': 'delINVdel'},
            {'chrom': 'chr1', 'start': 1001, 'stop': 1401, 'sv_type': 'dupINVdup', 'qual': 1, 'sv_len': 400,
             'gt': (0, 0), 'op_type': 'dupINVdup'},
            {'chrom': 'chr1', 'start': 121, 'stop': 131, 'sv_type': 'dDUP', 'qual': 1, 'sv_len': 10, 'target': 111,
             'gt': (0, 0), 'op_type': 'DUP'},
            {'chrom': 'chr1', 'start': 11, 'stop': 21, 'sv_type': 'dDUP', 'qual': 1, 'sv_len': 10, 'target': 31,
             'gt': (0, 0), 'op_type': 'DUP'},
            {'chrom': 'chr1', 'start': 31, 'stop': 51, 'sv_type': 'INV_dDUP', 'qual': 1, 'sv_len': 20, 'target': 11,
             'gt': (0, 0), 'op_type': 'INV_dDUP'},
            {'chrom': 'chr1', 'start': 101, 'stop': 120, 'sv_type': 'nrTRA', 'qual': 1, 'sv_len': 19, 'target': 11,
             'gt': (0, 0), 'op_type': 'nrTRA'},
            {'chrom': 'chr1', 'start': 11, 'stop': 21, 'sv_type': 'DEL', 'qual': 1, 'sv_len': 10, 'gt': (0, 0), 'op_type': 'CUT'}
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
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 109, 'stop': 120, 'qual': 999},
            {'chrom': 'chr1', 'sv_type': 'DEL', 'start': 1001, 'stop': 2001, 'qual': 10},
            {'chrom': 'chr2', 'sv_type': 'DEL', 'start': 10, 'stop': 100, 'qual': 1002},
        ]

        recs = []
        for rec_data in data:
            csv = ComplexSV(
                sv_type=rec_data['sv_type'],
                chrom=rec_data['chrom'],
                singleton_interval=(rec_data['start'], rec_data['stop']),
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
    def test_set_occupied_intervals(self):
        """
        Verifies the construction of the occupied_intervals field for singleton and non singleton
        Assumes that the ComplexSV class constructor calls set_occupied_intervals
        Individual SV type logic will be tested separately.
        """
        csv = ComplexSV("DEL",
                        chrom=None, qual=1.0,
                        singleton_interval=(4, 42))
        self.assertEqual([(4, 42)], csv.occupied_intervals)

        csv = ComplexSV("dDUP",
                        chrom=None, qual=1.0,
                        singleton_interval=None,
                        records=[SVBreakpoint(sv_type="dDUP_A", chr1='chr1', pos1=5,  chr2='chr1', pos2=15, qual=1),
                                 SVBreakpoint(sv_type="dDUP_B", chr1='chr1', pos1=6, chr2='chr2', pos2=20, qual=1)],
                        )
        self.assertEqual(sorted([(5, 6), (15, 20)]), sorted(csv.occupied_intervals))


    def test_get_bed_info(self):
        """
        Test that bed information is correctly constructed for various SV types
        Checks for correct name conversion
        Only tests data population and correct dispersion computation for one type.
        Individual SV types are tested separately
        """
        csv = ComplexSV(
            sv_type="DUP",
            chrom=1, qual=1.0,
            singleton_interval=(5, 15),
        )
        bed_info = csv.get_bed_info()
        assert len(bed_info) == 1
        self.assertEqual(1, bed_info[0]['chrom'])
        self.assertEqual(5, bed_info[0]['source_s'])
        self.assertEqual(15, bed_info[0]['source_e'])
        self.assertEqual(5, bed_info[0]['target_s'])
        self.assertEqual(15, bed_info[0]['target_e'])
        self.assertEqual('DUP', bed_info[0]['transform'])
        self.assertEqual(10, bed_info[0]['sv_len'])
        self.assertEqual('DUP', bed_info[0]['sv_type'])

        csv = ComplexSV(
            sv_type="nrTRA",
            chrom=1, qual=1.0,
            singleton_interval=None,
            records=[SVBreakpoint(sv_type='nrTRA_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=1),
                     SVBreakpoint(sv_type='nrTRA_B', chr1='chr1', pos1=6, chr2='chr1', pos2=20, qual=1)],
        )
        bed_info = csv.get_bed_info()
        self.assertEqual(1, bed_info[0]['chrom'])
        self.assertEqual(15, bed_info[0]['source_s'])
        self.assertEqual(20, bed_info[0]['source_e'])
        self.assertEqual(5, bed_info[0]['target_s'])
        self.assertEqual(5, bed_info[0]['target_e'])
        self.assertEqual('nrTRA', bed_info[0]['transform'])
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
            chrom=1, qual=0.9,
            singleton_interval=(5, 15),
        )

        vcf_info = csv.get_vcf_info()

        self.assertEqual(1, vcf_info['chrom'])
        self.assertEqual(5, vcf_info['start'] - 1)
        self.assertEqual(15, vcf_info['stop'] - 1)
        self.assertEqual('DEL', vcf_info['sv_type'])
        self.assertEqual(0.9, vcf_info['qual'])
        self.assertEqual(10, vcf_info['sv_len'])

        # Test dispersion SVs
        for dispersion_sv in ['nrTRA', 'dDUP', 'INV_dDUP']:

            csv = ComplexSV(
                sv_type=dispersion_sv,
                chrom=1, qual=0.9,
                singleton_interval=None,
                records=[SVBreakpoint(sv_type=f'{dispersion_sv}_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=1),
                         SVBreakpoint(sv_type=f'{dispersion_sv}_B', chr1='chr1', pos1=6, chr2='chr2', pos2=20, qual=1)],
            )

            vcf_info = csv.get_vcf_info()
            self.assertEqual(15, vcf_info['start'] - 1)
            self.assertEqual(20, vcf_info['stop'] - 1)
            self.assertEqual(dispersion_sv, vcf_info['sv_type'])

        # Test contained SVs

        csv = ComplexSV(
            sv_type="delINVdel", chrom=1, qual=0.9,
            records=[SVBreakpoint(sv_type=f'delINVdel_A', chr1='chr1', pos1=5, chr2='chr1', pos2=15, qual=1),
                     SVBreakpoint(sv_type=f'delINVdel_B', chr1='chr1', pos1=6, chr2='chr1', pos2=20, qual=1)],
        )
        vcf_info = csv.get_vcf_info()
        self.assertEqual(6, vcf_info['start'] - 1)
        self.assertEqual(19, vcf_info['stop'] - 1)


    def test_prioritize_evidence(self):
        """
        Test the toggle for filtering overlapping stitched SVs by .qual or by .evidence
        """
        overlapping_data = [
            [SVBreakpoint(sv_type='DEL', chr1='chr1', pos1=5, chr2='chr1', pos2=400, qual=30, gt=(None, None), evidence=20)],
            [SVBreakpoint(sv_type='INV', chr1='chr1', pos1=10, chr2='chr1', pos2=300, qual=10, gt=(None, None), evidence=30)],
            [SVBreakpoint(sv_type='DUP', chr1='chr1', pos1=401, chr2='chr1', pos2=1000, qual=40, gt=(None, None), evidence=3)],
            [SVBreakpoint(sv_type='INVdel_A', chr1='chr1', pos1=600, chr2='chr1', pos2=2000, qual=20, evidence=5),
             SVBreakpoint(sv_type='INVdel_B', chr1='chr1', pos1=601, chr2='chr1', pos2=3000, qual=25, evidence=7)],
            [SVBreakpoint(sv_type='dupINVdel_A', chr1='chr2', pos1=10, chr2='chr2', pos2=800, qual=0, evidence=0)],
        ]

        csvs = [ComplexSV.from_records(bkps) for bkps in overlapping_data]

        filtered_csvs = sorted(filter_merged_recs(csvs, prioritize_evidence=False), key=lambda x: (x.chrom, x.occupied_intervals))
        expected_result_qual = [csvs[0], csvs[2], csvs[4]]
        self.assertListEqual(filtered_csvs, expected_result_qual)

        filtered_csvs = sorted(filter_merged_recs(csvs, prioritize_evidence=True), key=lambda x: (x.chrom, x.occupied_intervals))
        expected_result_evidence = [csvs[1], csvs[3], csvs[4]]
        self.assertListEqual(filtered_csvs, expected_result_evidence)

class TestSVTypes(TestCase):

    def run_pipeline(self, input_recs, expected_calls, expected_fragments = None):
        """
        Generic test to run stitching on a set of input records, checking that the correct calls are returned.

        :param input_recs: list of dicts containing fields 'start', 'stop', 'id', and 'qual'
        :param expected_calls: list of dicts containing fields 'start', 'stop', 'sv_type', 'sv_len',
                                'target', and info 'fragments'
                Expected bp locations are zero indexed for both the input_recs and expected_calls,
                and this function will convert them to 1-indexed for comparison with VCF format output
        """
        recs = []
        for rec_data in input_recs:
            rec = SVBreakpoint(sv_type=rec_data['id'],
                               chr1='chr1',
                               pos1=rec_data['start'],
                               chr2='chr1',
                               pos2=rec_data['stop'],
                               qual=rec_data['qual'])

            recs.append(rec)

        merged_recs = find_matches(recs)
        # filtered_recs = filter_merged_recs(merged_recs)
        print(merged_recs)

        keys_to_compare = {'start', 'stop', 'sv_type', 'sv_len', 'target', 'info', 'fragments'}

        merged_rec_dicts = [rec.get_vcf_info() for rec in merged_recs]
        pruned_merged_dicts = [
            {key: d[key] for key in keys_to_compare if key in d}
            for d in merged_rec_dicts
        ]

        print(f"Process returned {pruned_merged_dicts}, expected {expected_calls}")

        # convert expected_calls to 1-indexed
        for sv in expected_calls:
            sv['start'] += 1
            sv['stop'] += 1
            if 'target' in sv:
                sv['target'] += 1

        self.assertEqual(sorted(expected_calls, key=lambda x: (x['start'], x['stop'], x['sv_type'])),
                         sorted(pruned_merged_dicts, key=lambda x: (x['start'], x['stop'], x['sv_type'])))

        if expected_fragments:
            bed_dicts = [rec.get_bed_info() for rec in merged_recs]
            for bed_dict in bed_dicts:
                check_values = [(rec['source_s'], rec['source_e'], rec['transform']) for rec in bed_dict]
                self.assertEqual(sorted(expected_fragments), sorted(check_values))

    def test_del(self):
        """
        DEL is a singleton record directly linking the start of the deletion to the end
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DEL'}]
        expected_result = [{'start': 5, 'stop': 15, 'sv_type': 'DEL', 'sv_len': 10}]
        self.run_pipeline(data, expected_result)

    def test_inv(self):
        """
        INV is an interlocking pair adjacent on both sides
        """
        deta = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'INV_A'},
                {'start': 6, 'stop': 16, 'qual': 0.9, 'id': 'INV_B'}]
        expected_result = [{'start': 6, 'stop': 15, 'sv_type': 'INV', 'sv_len': 9}]
        self.run_pipeline(deta, expected_result)

    def test_dup(self):
        """
        DUP is a singleton record indicating the region that is repeated once
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DUP'}]
        expected_result = [{'start': 5, 'stop': 15, 'sv_type': 'DUP', 'sv_len': 10}]
        self.run_pipeline(data, expected_result)

    def test_dup_inv(self):
        """
        DUP-INV has a structure where each SV should have two interlocking A edges and a contained B edge that has
        shared start and end with each other edge. The interlocking edges should be off by one.
        """
        data = [{'start': 5, 'stop': 15, 'qual': 0.9, 'id': 'DUP_INV_A'},
                {'start': 6, 'stop': 16, 'qual': 0.9, 'id': 'DUP_INV_C'},
                {'start': 6, 'stop': 15, 'qual': 0.9, 'id': 'DUP_INV_B'},
                ]
        expected_result = [{'start': 6, 'stop': 15, 'sv_type': 'DUP_INV', 'sv_len': 9}]
        self.run_pipeline(data, expected_result)

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

        self.run_pipeline(data, expected_result)

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
        self.run_pipeline(data, expected_result)

    def test_ddup_inv(self):
        """
        dDUP_INV has four record types A, B, C, D with A and B interlocking adjacent and C enveloping D,
        adjacent at the target. B and C share the same start, and A's end and D's start are shared.
        Symmetric version flips the CD structure so that the AB structure is on its right.
        """
        data = [
            {'start': 1, 'stop': 50, 'qual': 100, 'id': 'dDUP_INV_A'},
            {'start': 2, 'stop': 51, 'qual': 100, 'id': 'dDUP_INV_B'},
            {'start': 2, 'stop': 100, 'qual': 100, 'id': 'dDUP_INV_C'},
            {'start': 50, 'stop': 99, 'qual': 100, 'id': 'dDUP_INV_D'},
            {'start': 151, 'stop': 200, 'qual': 100, 'id': 'dDUP_INV_B'},
            {'start': 150, 'stop': 199, 'qual': 100, 'id': 'dDUP_INV_A'},
            {'start': 101, 'stop': 199, 'qual': 100, 'id': 'dDUP_INV_C'},
            {'start': 102, 'stop': 151, 'qual': 100, 'id': 'dDUP_INV_D'},
        ]
        expected_result = [
            {'start': 2, 'stop': 50, 'sv_type': 'dDUP_INV', 'target': 99, 'sv_len': 48},
            {'start': 151, 'stop': 199, 'sv_type': 'dDUP_INV', 'target': 101, 'sv_len': 48},
        ]
        self.run_pipeline(data, expected_result)

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
        self.run_pipeline(data, expected_result)

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

        self.run_pipeline(data, expected_result)

    def test_invdel(self):
        """
        INVdel has A and B interlocking pairs, with B before A. The section region is inverted, and the subsequent
        region is deleted
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdel_A'},
            {'start': 2, 'stop': 30, 'qual': 100, 'id': 'INVdel_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 29, 'sv_type': 'INVdel', 'sv_len': 27},
        ]
        self.run_pipeline(data, expected_result)

    def test_delinv(self):
        """
        delINV has A and B interlocking adjacent pairs, with A before B. The second region is inverted,
        and the preceding region is deleted
        """
        data = [
            {'start': 1, 'stop': 29, 'qual': 100, 'id': 'delINV_A'},
            {'start': 15, 'stop': 30, 'qual': 100, 'id': 'delINV_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 29, 'sv_type': 'delINV', 'sv_len': 27},
        ]
        self.run_pipeline(data, expected_result)

    def test_invdup(self):
        """
        INVdup has A enclosing B, adjacent on the left, with all of A being a variant,
        but has fragments that need to be checked.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'INVdup_A'},
            {'start': 2, 'stop': 10, 'qual': 100, 'id': 'INVdup_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 20, 'sv_type': 'INVdup', 'sv_len': 18},
        ]
        expected_fragments = [
            (2, 9, 'INV'),
            (10, 20, 'COPY_invPASTE'),
        ]
        self.run_pipeline(data, expected_result)

    def test_dupINV(self):
        """
        dupINV has A enclosing B adjacent on the right, with all of A being a variant,
        but has fragments that need to be checked.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINV_A'},
            {'start': 10, 'stop': 19, 'qual': 100, 'id': 'dupINV_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 19, 'sv_type': 'dupINV', 'sv_len': 18},
        ]
        expected_fragments = [
            (1, 10, 'COPY_invPASTE'),
            (11, 19, 'INV')
        ]
        self.run_pipeline(data, expected_result)

    def test_dupINVdel(self):
        """
        dupINVdel has A enclosing B, with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINVdel_A'},
            {'start': 5, 'stop': 15, 'qual': 100, 'id': 'dupINVdel_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 19, 'sv_type': 'dupINVdel', 'sv_len': 18},
        ]
        expected_fragments = [
            (1, 5, 'COPYinv-PASTE'),
            (6, 15, 'INV'),
            (16, 19, 'CUT'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments)


    def test_delINVdup(self):
        """
        dupINVdel has A enclosing B, with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'delINVdup_A'},
            {'start': 5, 'stop': 15, 'qual': 100, 'id': 'delINVdup_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 20, 'sv_type': 'delINVdup', 'sv_len': 18},
        ]
        expected_fragments = [
            (2, 4, 'CUT'),
            (5, 14, 'INV'),
            (15, 20, 'COPYinv-PASTE'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments)

    def test_delINVdel(self):
        """
        delINVdel has A interlocking with another A with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'delINVdel_A'},
            {'start': 5, 'stop': 25, 'qual': 100, 'id': 'delINVdel_B'},
        ]
        expected_result = [
            {'start': 2, 'stop': 24, 'sv_type': 'delINVdel', 'sv_len': 22},
        ]
        expected_fragments = [
            (2, 4, 'CUT'),
            (5, 20, 'INV'),
            (21, 24, 'CUT'),
        ]
        self.run_pipeline(data, expected_result, expected_fragments)

    def test_dupINVdup(self):
        """
        dupINVdup has A interlocking with another A with no adjacent endpoints, breaking into three fragments.
        """
        data = [
            {'start': 1, 'stop': 20, 'qual': 100, 'id': 'dupINVdup_A'},
            {'start': 5, 'stop': 25, 'qual': 100, 'id': 'dupINVdup_B'},
        ]
        expected_result = [
            {'start': 1, 'stop': 25, 'sv_type': 'dupINVdup', 'sv_len': 24},
        ]
        expected_fragments = [
            (1, 5, 'COPYinv-PASTE'),
            (6, 19, 'INV'),
            (20, 25, 'COPYinv-PASTE'),  # todo: these operations need a target? how do we define a target properly?
        ]
        self.run_pipeline(data, expected_result, expected_fragments)
