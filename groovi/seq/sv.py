import groovi.utils.types

TYPES_SINGLE_BKP = ["DEL", "INV", "DUP", "INVDUP"]
TYPES_MULTI_BKP = ["TRA"]
TYPES = TYPES_SINGLE_BKP + TYPES_MULTI_BKP
SVType = groovi.utils.types.make_enum("SVType",
                                   ["NEG"] + TYPES +
                                   [sv + "_HOM" for sv in TYPES] +
                                   [sv + "_HET" for sv in TYPES], __name__)
SVTypeSet = groovi.utils.types.make_enum("SVTypeSet",
                                      ["CPX12"], __name__)
SV_CLASS_SETS_GT = {SVTypeSet.CPX12}
TYPE_SET_TO_TYPES = {SVTypeSet.CPX12: [SVType.NEG, SVType.DEL_HOM, SVType.INV_HOM, SVType.DUP_HOM,
                                          SVType.DEL_HET, SVType.INV_HET, SVType.DUP_HET,
                                          SVType.INVDUP_HOM, SVType.INVDUP_HET]}

class SV:
    def __init__(self, sv_type, chr_name, start, end, qual, gt):
        self.type = sv_type
        self.chr_name = chr_name
        self.start = start
        self.end = end
        self.qual = qual
        self.gt = gt
        self.len = abs(end - start)
        self.evidence = None
        self.evidence_fuzzy = None
        self.annotation = None
        self.assign_internal_type()

    def update(self, start, end, evidence, evidence_fuzzy, annotation):
        self.start = start
        self.end = end
        self.len = abs(end - start)
        self.evidence = evidence
        self.evidence_fuzzy = evidence_fuzzy
        self.annotation = annotation

    @classmethod
    def from_vcf(cls, rec):
        sv_len = abs(rec.stop - rec.pos)
        gt = (None, None)
        if 'GT' in rec.samples[rec.samples[0].name]:
            gt = rec.samples[rec.samples[0].name]['GT']
        assert 'SVTYPE' in rec.info
        sv = SV(rec.info['SVTYPE'], rec.contig, int(rec.pos) - 1, int(rec.pos) - 1 + sv_len, rec.qual, gt)
        if 'ReadSupport' in rec.info and rec.info['ReadSupport'] is not None:
            sv.evidence = int(rec.info['ReadSupport'])
        if 'ReadSupportFuzzy' in rec.info and rec.info['ReadSupportFuzzy'] is not None:
            sv.evidence_fuzzy = int(rec.info['ReadSupportFuzzy'])
        return sv

    def assign_internal_type(self):
        gt_token = "" if self.gt == (None, None) else "_HOM" if self.gt == (1, 1) else "_HET"
        self.internal_type = SVType[self.type + gt_token]

    @staticmethod
    def parse_internal_type(sv_class_enum):
        name_tokens = sv_class_enum.name.split("_")
        if len(name_tokens) == 2: return name_tokens[0], (1, 1) if name_tokens[1] == "HOM" else (0, 1)
        return name_tokens[0], (None, None)

    def __str__(self):
        return '\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())

    @staticmethod
    def compare(sv1, sv2):
        return sv1.start - sv2.start

    @staticmethod
    def compare_by_score(sv1, sv2):
        return float(sv1.qual) - float(sv2.qual)

    @staticmethod
    def compare_by_support(sv1, sv2):
        return sv1.evidence - sv2.evidence

    def __lt__(self, sv):
        return self.start < sv.start

    @staticmethod
    def get_sv_type_labels(sv_classes):
        return {sv_class: i for i, sv_class in enumerate(sv_classes)}
