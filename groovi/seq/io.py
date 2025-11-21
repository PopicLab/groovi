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

def write_bkp_vcf(vcf_fname, sv_bkps, chr_index):
    header = pysam.VariantHeader()
    header.add_line('##INFO=<ID=CHR2,Number=1,Type=String,Description="Name of second chromosome">')
    header.add_line('##INFO=<ID=BKP_TYPE,Number=1,Type=String,Description="Breakpoint type">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Breakpoint type">')
    header.add_line('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Read support">')
    header.add_line('##INFO=<ID=ORIENTATION,Number=1,Type=String,Description="Breakpoint orientation">')
    header.add_line('##INFO=<ID=SOURCE,Number=1,Type=String,Description="Breakpoint evidence source">')
    header.add_sample("SAMPLE")
    for ctg in chr_index.contigs(): header.add_line("##contig=<ID=%s>\n" % ctg.name)
    vcf = pysam.VariantFile(vcf_fname, 'w', header=header)
    for sv in sv_bkps:
        vcf.write(vcf.new_record(contig=sv.chr1, start=sv.pos1, stop=sv.pos2,
                                 info={"CHR2": sv.chr2, "BKP_TYPE": sv.type,
                                       "BKP_TYPE": sv.type, "SUPPORT": sv.evidence,
                                       "ORIENTATION": sv.orientation, "SOURCE": sv.source},
                                 alleles=['N', "<%s>" % sv.type],
                                 id=f"<{sv.type}>", qual=sv.qual, filter=None, samples=None),
                                 samples=[{'GT': (None, None)}])
    vcf.close()
    logging.info("Wrote breakpoints to %s" % vcf_fname)


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


