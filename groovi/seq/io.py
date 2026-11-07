from groovi.seq.constants import *

from collections import namedtuple, OrderedDict
import pysam
from groovi.seq.sv import *
from groovi.seq.stitch_rules import SV_REGISTRY

###############
#     VCF     #
###############

def vcf_iter(vcf_fname, include_types=None):
    vcf_file = pysam.VariantFile(vcf_fname)
    for rec in vcf_file.fetch():
        sv = SVJunction.from_vcf(rec)
        if include_types and sv.type not in include_types: continue
        yield sv
    vcf_file.close()

def write_junction_vcf(vcf_fname, sv_bkps, chr_index):
    header = pysam.VariantHeader()
    header.add_line('##INFO=<ID=CHR2,Number=1,Type=String,Description="Name of second chromosome">')
    header.add_line('##INFO=<ID=JTYPE,Number=1,Type=String,Description="Junction type">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Junction type">')
    header.add_line('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Read support">')
    header.add_line('##INFO=<ID=ORIENTATION,Number=1,Type=String,Description="Junction orientation">')
    header.add_line('##INFO=<ID=SOURCE,Number=1,Type=String,Description="Junction evidence source">')
    header.add_sample("SAMPLE")
    header.formats.add('GT', number=1, type='String', description='Genotype')
    for ctg in chr_index.contigs(): header.add_line("##contig=<ID=%s>\n" % ctg.name)
    vcf = pysam.VariantFile(vcf_fname, 'w', header=header)
    for sv in sv_bkps:
        vcf.write(vcf.new_record(contig=sv.chr1, start=sv.pos1, stop=sv.pos2, # start positions are off by one in vcf
                                 info={"CHR2": sv.chr2, "JTYPE": sv.type,
                                       "SVTYPE": sv.type, "SUPPORT": sv.evidence,
                                       "ORIENTATION": sv.orientation, "SOURCE": sv.source},
                                 alleles=['N', "<%s>" % sv.type],
                                 id=f"<{sv.type}>", qual=sv.qual, filter=None, samples=[{'GT': (None, None)}]))
    vcf.close()

def write_csv_vcf(filtered_svs, output_path, chr_index, split_records=False):
    header = pysam.VariantHeader()
    for ctg in chr_index.contigs(): header.add_line("##contig=<ID=%s>\n" % ctg.name)
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

    vcf_out_file = pysam.VariantFile(output_path, 'w', header=header)
    for parent_id, sv in enumerate(filtered_svs):
        vcf_info_dict = sv.get_vcf_info()
        ambiguous = len(sv.grammars) > 1
        if split_records:
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


