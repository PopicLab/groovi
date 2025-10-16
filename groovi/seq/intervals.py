class GenomeInterval:
    def __init__(self, source_chrom, target_chrom, start_pos, end_pos, **kwargs):
        self.source_chrom = source_chrom
        self.target_chrom = target_chrom
        self.start = start_pos
        self.end = end_pos

        # other attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_start_pos(self):
        return self.start
    
    def get_end_pos(self):
        return self.end
    
    def get_start_chrom(self):
        return self.source_chrom
    
    def get_target_chrom(self):
        return self.target_chrom

    def __len__(self):
        return self.end - self.start + 1

    def __str__(self):
        return "%s_%d_%d" % (self.chr_tid, self.start, self.end)

    def __lt__(self, interval):
        return self.start < interval.start

    def to_list(self):
        return [self.chr_tid, self.start, self.end]

    @staticmethod
    def from_list(interval_list):
        return GenomeInterval(interval_list[0], interval_list[1], interval_list[2])



