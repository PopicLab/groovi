import tempfile
from types import SimpleNamespace
from unittest import TestCase

import pysam

from groovi.seq.stitching import ComplexSV, export_callset_to_vcf
from groovi.seq.sv import SVBreakpoint


def bkp(sv_type, pos1, pos2):
    return SVBreakpoint(sv_type, chr1='chr1', pos1=pos1, chr2='chr1', pos2=pos2, qual=40)


def ops(fragments):
    return [(frag.operation, frag.start, frag.end, frag.target_start) for frag in fragments]


DEL_INV_DUP = ComplexSV('delINVdup', 'chr1', 40, {'A': bkp('delINVdup_A', 99, 399), 'B': bkp('delINVdup_B', 200, 300)})


class TestAlternativeGrammars(TestCase):
    def test_delinvdup(self):
        (grammar, fragments), (alt_grammar, alt_fragments) = DEL_INV_DUP.grammars
        self.assertEqual((grammar, alt_grammar), ('ABC->cbC', 'ABC->cBC'))
        self.assertEqual(ops(fragments), [('CUT', 100, 200, None), ('INV', 200, 300, None),
                                          ('COPYinv-PASTE', 300, 400, 100)])
        self.assertEqual(ops(alt_fragments), [('CUT', 100, 200, None), ('KEEP', 200, 300, None),
                                              ('COPYinv-PASTE', 300, 400, 100)])

    def test_dupinvdel(self):
        sv = ComplexSV('dupINVdel', 'chr1', 40, {'A': bkp('dupINVdel_A', 99, 500), 'B': bkp('dupINVdel_B', 199, 299)})
        (grammar, fragments), (alt_grammar, alt_fragments) = sv.grammars
        self.assertEqual((grammar, alt_grammar), ('ABC->Aba', 'ABC->ABa'))
        self.assertEqual(ops(fragments), [('COPYinv-PASTE', 99, 200, 300), ('INV', 200, 300, None),
                                          ('CUT', 300, 500, None)])
        self.assertEqual(ops(alt_fragments), [('COPYinv-PASTE', 99, 200, 300), ('KEEP', 200, 300, None),
                                              ('CUT', 300, 500, None)])

    def test_dupinvdup(self):
        sv = ComplexSV('dupINVdup', 'chr1', 40, {'A': bkp('dupINVdup_A', 99, 300), 'B': bkp('dupINVdup_B', 199, 399)})
        self.assertEqual([grammar for grammar, _ in sv.grammars], ['ABC->AcbaC', 'ABC->ABCaC', 'ABC->AcABC'])
        self.assertEqual([[(frag.operation, frag.start, frag.end, frag.target_start, frag.insord) for frag in fragments]
                          for _, fragments in sv.grammars[1:]],
                         [[('COPYinv-PASTE', 99, 200, 400, 0), ('KEEP', 200, 300, None, None),
                           ('COPY-PASTE', 300, 400, 400, 1)],
                          [('COPYinv-PASTE', 300, 400, 200, 0), ('KEEP', 200, 300, None, None),
                           ('COPY-PASTE', 99, 200, 200, 1)]])

    def test_nrtra_both_sources(self):
        sv = ComplexSV('nrTRA', 'chr1', 40, {'A1': bkp('nrTRA_A', 10, 100), 'B': bkp('nrTRA_B', 11, 119),
                                             'A2': bkp('nrTRA_A', 99, 120)})
        self.assertEqual([(grammar, ops(fragments)) for grammar, fragments in sv.grammars],
                         [('_A->A_', [('CUT-PASTE', 100, 120, 11)]),
                          ('A_->_A', [('CUT-PASTE', 11, 100, 120)])])

    def test_inv_ddup_spacer_inverted(self):
        right = ComplexSV('INV_dDUP_right', 'chr1', 40, {'A': bkp('INV_dDUP_A', 100, 300), 'B': bkp('INV_dDUP_B', 199, 499)})
        left = ComplexSV('INV_dDUP_left', 'chr1', 40, {'A': bkp('INV_dDUP_A', 99, 599), 'B': bkp('INV_dDUP_B', 300, 500)})
        self.assertEqual([(grammar, ops(fragments)) for grammar, fragments in right.grammars[1:]],
                         [('AB->Aba', [('COPYinv-PASTE', 100, 200, 500), ('INV', 200, 500, None)])])
        self.assertEqual([(grammar, ops(fragments)) for grammar, fragments in left.grammars[1:]],
                         [('AB->baB', [('COPYinv-PASTE', 500, 600, 100), ('INV', 100, 500, None)])])

    def test_unambiguous_sv_has_one_grammar(self):
        sv = ComplexSV('DEL', 'chr1', 40, [bkp('DEL', 10, 100)])
        self.assertEqual(len(sv.grammars), 1)


class TestExport(TestCase):
    def export(self, svs, sep_frag_recs):
        path = tempfile.NamedTemporaryFile(suffix='.vcf', delete=False).name
        export_callset_to_vcf(svs, path, [SimpleNamespace(name='chr1')], sep_frag_recs=sep_frag_recs)
        return list(pysam.VariantFile(path))

    def test_fragment_records_tagged_by_grammar(self):
        unambiguous = ComplexSV('DEL', 'chr1', 40, [bkp('DEL', 10, 50)])
        records = self.export([DEL_INV_DUP, unambiguous], sep_frag_recs=True)
        tagged = [(rec.id, rec.info['GRAMMAR_ID'], rec.info['GRAMMAR'], rec.info['OP_TYPE']) for rec in records[:6]]
        self.assertEqual(tagged, [('sv0_g0_0', 0, 'ABC->cbC', 'CUT'), ('sv0_g0_1', 0, 'ABC->cbC', 'INV'),
                                  ('sv0_g0_2', 0, 'ABC->cbC', 'COPYinv-PASTE'),
                                  ('sv0_g1_0', 1, 'ABC->cBC', 'CUT'), ('sv0_g1_1', 1, 'ABC->cBC', 'KEEP'),
                                  ('sv0_g1_2', 1, 'ABC->cBC', 'COPYinv-PASTE')])
        self.assertEqual(records[6].id, 'sv1_0')
        self.assertNotIn('GRAMMAR_ID', records[6].info)

    def test_single_record_lists_grammars(self):
        record, = self.export([DEL_INV_DUP], sep_frag_recs=False)
        self.assertEqual(record.info['GRAMMAR'], 'ABC->cbC|ABC->cBC')
        # pysam splits the comma-separated fragment list on read
        self.assertEqual(len(','.join(record.info['FRAGMENT_INTERVALS']).split('|')), 2)
