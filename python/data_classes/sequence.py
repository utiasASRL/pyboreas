import os

from calib import Calib
from sensors import Camera, Lidar, Radar


class Sequence:
    def __init__(self, root, seqSpec):
        self.seqID = seqSpec[0]
        self.path = root + self.seqID
        self.start = str(seqSpec[1])
        self.end = str(seqSpec[2])
        self.cameraFrames = os.listdir(self.path + '/camera/')
        self.lidarFrames = os.listdir(self.path + '/lidar/')
        self.radarFrames = os.listdir(self.path + '/radar/')
        self.cameraFrames = [self.path + '/camera/' + f for f in self.cameraFrames if self.start <= f and f <= self.end]
        self.lidarFrames = [self.path + '/lidar/' + f for f in self.lidarFrames if self.start <= f and f <= self.end]
        self.radarFrames = [self.path + '/lidar/' + f for f in self.radarFrames if self.start <= f and f <= self.end]
        self.cameraFrames.sort()
        self.lidarFrames.sort()
        self.radarFrames.sort()
        self.calib = Calib(root + self.seqID + '/calib/')

    # TODO: load printable metadata string

    @property
    def cam0(self):
        for f in self.cameraFrames:
            yield Camera(f)

    def get_camera(self, idx):
        return Camera(self.cameraFrames[idx])

    @property
    def lidar(self):
        for f in self.lidarFrames:
            yield Lidar(f)

    def get_lidar(self, idx):
        return Lidar(self.lidarFrames[idx])

    @property
    def radar(self):
        for f in self.radarFrames:
            yield Radar(f)

    def get_radar(self, idx):
        return Radar(self.radarFrames[idx])

    def visualize(self):
        pass

    # TODO: generate video for the entire sequences
    # option 1: display video, option 2: save video to file

    def get_pose(self, sensType, timestamp):
        pass
    # TODO
