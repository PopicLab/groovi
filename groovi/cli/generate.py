import logging


def generate(config, chr_names): # generates graphs for the specified list of chromosomes
    logging.root.setLevel(logging.INFO)
    for chr_name in chr_names:
        for chr in chr_names:
            # .... process alignments / build the SV index
            # .... build/store the chromosome graph...
            # .... generate stats .....
            pass
    return True


def main(config_yaml):
    # .... load config ....
    # .... set up chromosome partitions ....
    # .... generate graphs ....
    # .... generate stats for the entire dataset ....
    pass