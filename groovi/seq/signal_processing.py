from groovi.graph.constants import *
from groovi.seq.io import *
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pycues import *
from sklearn.cluster import DBSCAN
import logging
import time

class SVSignalProcessor:
    @dataclass
    class JunctionSoA:
        pos: np.ndarray
        orientation: np.ndarray
        evidence_type: np.ndarray
        support: np.ndarray
        @classmethod
        def from_aos(cls, junctions):
            return cls(
                pos=np.array([(j.pos1, j.pos2) for j in junctions]),
                orientation=np.array([j.orientation for j in junctions]),
                evidence_type=np.array([j.source for j in junctions]),
                support=np.array([j.support for j in junctions]))

    def __init__(self, config):
        self.config = config
        self.cues_config = CuesConfig()
        self.cues_config.ref = config.fa
        self.cues_config.min_discord_len = config.min_discord_len
        self.cues_config.max_discord_len = np.iinfo(np.int32).max
        self.cues_config.min_clipped_len = config.min_clipped_len
        self.cues_config.min_mapq = config.min_mapq
        self.cues_config.max_discordance = config.max_discordance
        self.cues_config.run_recovery_ins = config.recover_ins
        self.cues_config.run_recovery_clipping = config.recover_clipping
        self.cues_config.run_recovery_subfrags = config.recover_subfrag
        self.cues_config.run_recovery_missing_segments = config.recover_missing_segs
        self.cues_config.max_alns_to_process = np.iinfo(np.int32).max
        self.cues_config.record_inter_chromosomal_junctions = config.interchrom
        self.cues_config.n_threads = self.config.n_cpus
        self.cues_config.aln_batch_size = self.config.aln_batch_size
        self.cues_config.max_seg_alns_to_fetch = 200
        self.cues = Cues(self.cues_config)
        self.stats_to_collect = set(NODE_FEAT_TO_ALNSTAT[f] for f in self.config.node_aln_feature_set)


    def extract_junctions(self, cluster=True):
        if not self.config.use_cached_junctions:
            junctions = self.cues.collect_bkps_from_bam(self.config.bam).junctions
            logging.info("Total number of junctions: %d ", len(junctions))
            junction_records = [SVBreakpoint.from_junction(j) for j in junctions]
            io.write_bkp_vcf(self.config.bkp_raw_vcf_file_path, junction_records, self.config.fai_index)
        else:
            junctions = [sv.to_junction() for sv in io.vcf_iter(self.config.bkp_raw_vcf_file_path)]
        if cluster:
            junctions = self.cluster_junctions(junctions)
            logging.info("Number of junctions after clustering: %d ", len(junctions))
            junction_records = [SVBreakpoint.from_junction(j) for j in junctions]
            io.write_bkp_vcf(self.config.bkp_cluster_vcf_file_path, junction_records, self.config.fai_index)
        return junctions

    def cluster_junctions(self, junctions):
        CENTROID_SHIFT_BY_ORIENTATION = {
            JOrientation.kLL: np.array([0, 0]), JOrientation.kLR: np.array([0, 1]),
            JOrientation.kRL: np.array([1, 0]), JOrientation.kRR: np.array([1, 1])
        }
        # split junctions according to their chromosome(s) pre-clustering
        junction_index = defaultdict(list)  # maps unique chr pairs to list of junctions
        for j in junctions:
            junction_index[tuple(sorted([j.chr1, j.chr2]))].append(j)

        # set up the clustering algorithm
        dbscan = DBSCAN(eps=self.config.cluster_radius, min_samples=1, metric='manhattan')

        # cluster junctions connecting the same pair of chromosomes
        junctions = [] # will be re-populated with clustered junctions
        for chr1, chr2 in junction_index:
            data = self.JunctionSoA.from_aos(junction_index[(chr1, chr2)])
            labels = dbscan.fit_predict(data.pos)
            for label in np.unique(labels):
                cluster_mask = labels == label
                cluster_size = np.sum(data.support[cluster_mask])
                if cluster_size < self.config.cluster_support_threshold: continue
                orientations_uniq = np.unique(data.orientation[cluster_mask])

                if len(orientations_uniq) == 1:
                    # centroid = np.round(np.sum(data.pos[cluster_mask] *
                    #                            data.support[cluster_mask].reshape(-1, 1), axis=0) / cluster_size).astype(int)
                    weighted_positions = np.repeat(data.pos[cluster_mask], data.support[cluster_mask], axis=0)
                    centroid = np.median(weighted_positions, axis=0).astype(int)

                    junctions.append(Junction(chr1=chr1, chr2=chr2, pos1=centroid[0], pos2=centroid[1],
                                              orientation=data.orientation[cluster_mask][0],
                                              evidence_type=data.evidence_type[cluster_mask][0], # TODO keep all
                                              support=cluster_size))
                else:
                    orientation_idx = [np.flatnonzero((data.orientation == o) & cluster_mask) for o in orientations_uniq]
                    orientation_support = [np.sum(data.support[idx]) for idx in orientation_idx]
                    anchor_idx = np.argmax(orientation_support)
                    # anchor_centroid = np.round(np.sum(data.pos[orientation_idx[anchor_idx]] *
                    #                                 data.support[orientation_idx[anchor_idx], None], axis=0) /
                    #                          orientation_support[anchor_idx]).astype(int)
                    weighted_positions = np.repeat(data.pos[orientation_idx[anchor_idx]],
                                                   data.support[orientation_idx[anchor_idx]], axis=0)
                    anchor_centroid = np.median(weighted_positions, axis=0).astype(int)
                    for i, orientation in enumerate(orientations_uniq):
                        centroid = (anchor_centroid +
                                    CENTROID_SHIFT_BY_ORIENTATION[orientation] -
                                    CENTROID_SHIFT_BY_ORIENTATION[orientations_uniq[anchor_idx]])
                        junctions.append(Junction(chr1=chr1, chr2=chr2, pos1=centroid[0], pos2=centroid[1],
                                                  orientation=orientation,
                                                  evidence_type=data.evidence_type[orientation_idx[i]][0],
                                                  support=orientation_support[i]))
        return junctions

    def extract_segments(self, junctions) -> list[RefSegment]:
        chr2pos = defaultdict(list)
        chr2orientation = defaultdict(list)
        for j in junctions:
            chr2pos[j.chr1].append(j.pos1)
            chr2pos[j.chr2].append(j.pos2)
            chr2orientation[j.chr1].append(j.orientation.name[1])
            chr2orientation[j.chr2].append(j.orientation.name[2])
        segments = []
        for chr_name in chr2pos:
            start_pos = set()
            end_pos = set()
            for pos, orientation_at_bp in zip(chr2pos[chr_name], chr2orientation[chr_name]):
                if orientation_at_bp == 'L': end_pos.add(pos)
                elif orientation_at_bp == 'R': start_pos.add(pos)

            # for each break, create the other end of the edge
            for L_pos in end_pos: start_pos.add(L_pos + 1)
            for R_pos in start_pos: end_pos.add(R_pos - 1)

            # create entries for the start and end of chromosome
            start_pos.add(1)
            end_pos.add(self.config.fai_index.get_chr_by_name(chr_name).len)

            # sort and create segments
            segments.extend([RefSegment(chr_name, s, e) for s, e in zip(sorted(list(start_pos)), sorted(list(end_pos)))])
        logging.info("Number of reference segments: %d ", len(segments))
        return segments

    def generate_stats(self, segments):
        head_segment_rois = []
        tail_segment_rois = []
        for segment in segments: # define head and tail segment regions
            head_segment_rois.append(RefSegment(segment.chr, max(segment.start - self.config.node_stats_buffer_overlap, 0),
                                                min(segment.start + self.config.node_stats_buffer, segment.end) + 1))
            tail_segment_rois.append(RefSegment(segment.chr,
                                                max(segment.end - self.config.node_stats_buffer, segment.start),
                                                min(segment.end + 1 + self.config.node_stats_buffer_overlap,
                                                    self.config.fai_index.get_chr_by_name(segment.chr).len)))
        stats = (self.cues.collect_stats(self.config.bam, head_segment_rois, self.stats_to_collect),
                self.cues.collect_stats(self.config.bam, tail_segment_rois, self.stats_to_collect))
        logging.info("Finished node stats collection")
        return stats

