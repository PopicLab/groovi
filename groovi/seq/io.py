from groovi.seq.constants import *

from collections import namedtuple, OrderedDict
import pysam
from groovi.seq.sv import *

###############
#     VCF     #
###############

def vcf_iter(vcf_fname, include_types=None):
    vcf_file = pysam.VariantFile(vcf_fname)
    for rec in vcf_file.fetch():
        sv = SVBreakpoint.from_vcf(rec)
        if include_types and sv.type not in include_types: continue
        yield sv
    vcf_file.close()

def vcf_header(contigs, ctg_no_len=False):
    header = pysam.VariantHeader()
    for ctg in contigs:
        if not ctg_no_len: header.add_line("##contig=<ID=%s,length=%d>\n" % (ctg.name, ctg.len))
        else: header.add_line("##contig=<ID=%s>\n" % ctg.name)
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the SV">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV">')
    header.add_line('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of SV">')
    header.add_line('##INFO=<ID=ReadSupport,Number=1,Type=Integer,Description="Read support (exact)">')
    header.add_line('##INFO=<ID=ReadSupportFuzzy,Number=1,Type=Integer,Description="Read support (fuzzy)">')
    header.formats.add('GT', number=1, type='String', description="Genotype")
    header.add_sample("SAMPLE")
    return header

def write_vcf(svs, vcf_fname, fai_fname, sv_types=None, min_score=0, min_len=0):
    chr_index = ChrFAIndex.load(fai_fname)
    vcf = pysam.VariantFile(vcf_fname, 'w', header=vcf_header(chr_index.contigs()))
    svid = 0
    for sv in svs:
        if sv_types is not None and sv.type not in sv_types: continue
        if sv.len < min_len: continue
        if sv.qual < min_score: continue
        vcf.write(vcf.new_record(contig=sv.chr_name,
                                       start=sv.start + 1,
                                       stop=sv.end + 1,
                                       alleles=['N', "<%s>" % sv.type],
                                       id="groovi.%s.%d" % (sv.type, svid),
                                       info={'SVTYPE': sv.type, 'SVLEN': sv.len,
                                             'ReadSupport': sv.evidence,
                                             'ReadSupportFuzzy': sv.evidence_fuzzy},
                                       qual=sv.qual,
                                       filter='.',
                                       samples=[{'GT': sv.gt}]))
        svid += 1
    vcf.close()

Chr = namedtuple('Chr', 'tid name len')
class ChrFAIndex:
    def __init__(self):
        self.tid2chr = OrderedDict()
        self.name2chr = OrderedDict()

    @staticmethod
    def load(fai_fname, all=False):
        chr_index = ChrFAIndex()
        with open(fai_fname, "r") as faidx:
            for tid, line in enumerate(faidx):
                tid = tid + 1
                name, length, _, _, _ = line[:-1].split()
                if not all and ("_" in name or "chrM" in name or "chrEBV" in name or "hs" in name): continue
                chr_index.add(Chr(tid, name, int(length)))
        return chr_index

    def add(self, chr):
        self.tid2chr[chr.tid] = chr
        self.name2chr[chr.name] = chr

    def get_tid_by_name(self, chr_name):
        return self.name2chr[chr_name].tid

    def get_chr_by_tid(self, tid):
        return self.tid2chr[tid]

    def get_chr_by_name(self, chr_name):
        return self.name2chr[chr_name]

    def contigs(self):
        return self.tid2chr.values()

    def get_chr_names(self) -> list[str]:
        return list(self.name2chr.keys())

    def get_chr_ids(self) -> list[int]:
        return list(self.tid2chr.keys())

    def get_chr_lengths(self) -> dict[int, int]:
        return {tid: self.get_chr_by_tid(tid).len for tid in self.tid2chr}
    
    def get_tid_to_str_conversion(self) -> dict[int, str]:
        return {tid: chr.name for tid, chr in self.tid2chr.items()}


