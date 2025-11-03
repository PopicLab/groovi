class GenomeInterval:
    def __init__(self, start_chrom, end_chrom, start_pos, end_pos, **kwargs):
        self.start_chrom = start_chrom
        self.end_chrom = end_chrom
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
        return self.start_chrom

    def get_target_chrom(self):
        return self.end_chrom

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
