import logging

def call(config, device, chr_names):  # runs SV calling on the specified device for the specified list of chromosomes
    logging.root.setLevel(logging.getLevelName(config.logging_level))
    # ... load model ...
    for chr in chr_names:
        # .... process alignments / build the SV index
        # .... build the chromosome graph...
        # .... classify edges .....
        # .... postprocess predictions ....
        pass
    return True

def main(config_yaml):
    # .... load config ....
    # .... set up chromosome partitions ....
    # ------ 1. Graph-based discovery ------
    # ------ 2. Genome-based post-processing / refinement ------
    # ------ 3. Output generation ------
    pass