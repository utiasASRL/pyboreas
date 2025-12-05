import multiprocessing
import os
from multiprocessing import Pool

from pyboreas.data.sequence import Sequence


class BoreasDataset:
    def __init__(
        self, root="/data/boreas/", split=None, verbose=False, labelFolder="labels"
    ):
        self.root = root
        self.split = split
        self.aeva_frames = []
        self.camera_frames = []
        self.lidar_frames = []
        self.radar_frames = []
        self.sequences = []
        self.seqDict = {}  # seq string to index
        self.map = None  # TODO: Load the HD map data
        self.labelFolder = labelFolder

        if split is None:
            split = sorted(
                [
                    [f]
                    for f in os.listdir(root)
                    if f.startswith("boreas-") and f != "boreas-test-gt"
                ]
            )

        # It takes a few seconds to construct each sequence, so we parallelize this
        pool = Pool(multiprocessing.cpu_count())
        self.sequences = list(pool.starmap(Sequence, ((root, s, self.labelFolder) for s in split)))
        self.sequences.sort(key=lambda x: x.ID)

        for seq in self.sequences:
            if seq.aeva_frames:
                self.aeva_frames += seq.aeva_frames
            self.camera_frames += seq.camera_frames
            self.lidar_frames += seq.lidar_frames
            self.radar_frames += seq.radar_frames
            self.seqDict[seq.ID] = len(self.seqDict)
            if verbose:
                seq.print()

        if verbose:
            print("total aeva frames: {}".format(len(self.aeva_frames)))
            print("total camera frames: {}".format(len(self.camera_frames)))
            print("total lidar frames: {}".format(len(self.lidar_frames)))
            print("total radar frames: {}".format(len(self.radar_frames)))

    def get_seq_from_ID(self, ID):
        return self.sequences[self.seqDict[ID]]

    def get_seq(self, idx):
        return self.sequences[idx]

    def get_camera(self, idx):
        self.camera_frames[idx].load_data()
        return self.camera_frames[idx]

    def get_lidar(self, idx):
        self.lidar_frames[idx].load_data()
        return self.lidar_frames[idx]

    def get_radar(self, idx):
        self.radar_frames[idx].load_data()
        return self.radar_frames[idx]

    def get_aeva(self, idx):
        self.aeva_frames[idx].load_data()
        return self.aeva_frames[idx]
