from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from pycues import *
import logging

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
        self.cues_config.max_alns_to_process = 10000
        self.cues_config.n_threads = 20
        self.cues = Cues(self.cues_config)
        self.stats_to_collect = {AlnStatistic.kRD, AlnStatistic.kRDAll, AlnStatistic.kClipLeft,
                                 AlnStatistic.kClipRight, AlnStatistic.kIns} # TODO: reset for node feature sets


    def get_aln_stats_for_range(self, segment: RefSegment):
        stats = {AlnStatistic.kRD, AlnStatistic.kRDAll, AlnStatistic.kClipLeft,
                 AlnStatistic.kClipRight, AlnStatistic.kIns}
        aln_stats = self.cues.collect_stats(self.config.bam, [segment], stats)

    def extract_junctions(self, cluster=True):
        junctions = self.cues.collect_bkps_from_bam(self.config.bam).junctions
        logging.info("Total number of junctions: %d ", len(junctions))
        if cluster:
            junctions = self.cluster_junctions(junctions)
            logging.info("Number of junctions after clustering: %d ", len(junctions))
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
        junctions = [] # will be re-populated with clusterd junctions
        for chr1, chr2 in junction_index:
            data = self.JunctionSoA.from_aos(junction_index[(chr1, chr2)])
            labels = dbscan.fit_predict(data.pos)
            for label in np.unique(labels):
                cluster_mask = labels == label
                cluster_size = np.sum(data.support[cluster_mask])
                if cluster_size <= self.config.cluster_support_threshold: continue
                orientations_uniq = np.unique(data.orientation[cluster_mask])
                if len(orientations_uniq) == 1:
                    centroid = np.round(np.sum(data.pos[cluster_mask] *
                                               data.support[cluster_mask].reshape(-1, 1), axis=0) / cluster_size).astype(int)
                    junctions.append(Junction(chr1=chr1, chr2=chr2, pos1=centroid[0], pos2=centroid[1],
                                                   orientation=data.orientation[cluster_mask][0],
                                                   evidence_type=data.evidence_type[cluster_mask][0], # TODO keep all
                                                   support=cluster_size))
                else:
                    orientation_idx = [np.flatnonzero((data.orientation == o) & cluster_mask) for o in orientations_uniq]
                    orientation_support = [np.sum(data.support[idx]) for idx in orientation_idx]
                    anchor_idx = np.argmax(orientation_support)
                    anchor_centroid = np.round(np.sum(data.pos[orientation_idx[anchor_idx]] *
                                                    data.support[orientation_idx[anchor_idx], None], axis=0) /
                                             orientation_support[anchor_idx]).astype(int)
                    for i, orientation in enumerate(orientations_uniq):
                        centroid = (anchor_centroid +
                                    CENTROID_SHIFT_BY_ORIENTATION[orientation] -
                                    CENTROID_SHIFT_BY_ORIENTATION[orientations_uniq[anchor_idx]])
                        junctions .append(Junction(chr1=chr1, chr2=chr2,
                                                       pos1=centroid[0], pos2=centroid[1],
                                                       orientation=orientation,
                                                       evidence_type=data.evidence_type[orientation_idx[i]][0],
                                                       support=orientation_support[i]
                                                       ))
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
            head_segment_rois.append(RefSegment(segment.chr, segment.start,
                                                min(segment.start + self.config.node_feature_radius, segment.end)))
            tail_segment_rois.append(RefSegment(segment.chr,
                                                max(segment.end - self.config.node_feature_radius, segment.start),
                                                segment.end))
        return (self.cues.collect_stats(self.config.bam, head_segment_rois, self.stats_to_collect),
                self.cues.collect_stats(self.config.bam, tail_segment_rois, self.stats_to_collect))
