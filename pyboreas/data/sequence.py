import os
import os.path as osp

from pyboreas.data.calib import Calib
from pyboreas.data.sensors import Camera, Lidar, Radar

class Sequence:
    def __init__(self, boreas_root, seqSpec):
        self.ID = seqSpec[0]
        if len(seqSpec) > 2:
            assert seqSpec[2] > seqSpec[1], 'Sequence timestamps must go forward in time'
            self.start_ts = str(seqSpec[1])
            self.end_ts = str(seqSpec[2])
        else:
            self.start_ts = '0'  # dummy start and end if not specified
            self.end_ts = '9' * 21
        self.seq_root = osp.join(boreas_root, self.ID)
        self.applanix_root = osp.join(self.seq_root, 'applanix')
        self.calib_root = osp.join(self.seq_root, 'calib')
        self.camera_root = osp.join(self.seq_root, 'camera')
        self.lidar_root = osp.join(self.seq_root, 'lidar')
        self.radar_root = osp.join(self.seq_root, 'radar')

        self._check_dataroot_valid(self.seq_root)

        self.calib = Calib(self.calib_root)

        cfile = osp.join(self.applanix_root, 'camera_poses.csv')
        lfile = osp.join(self.applanix_root, 'lidar_poses.csv')
        rfile = osp.join(self.applanix_root, 'radar_poses.csv')
        self.camera_frames = self._get_frames(cfile, self.camera_root, '.png', Camera)
        self.lidar_frames = self._get_frames(lfile, self.lidar_root, '.bin', Lidar)
        self.radar_frames = self._get_frames(rfile, self.radar_root, '.png', Radar)

    def get_camera(self, idx):
        self.camera_frames[idx].load_data()
        return self.camera_frames[idx]

    @property
    def camera(self):
        for camera_frame in self.camera_frames:
            camera_frame.load_data()
            yield camera_frame

    def get_lidar(self, idx):
        self.lidar_frames[idx].load_data()
        return self.lidar_frames[idx]

    @property
    def lidar(self):
        for lidar_frame in self.lidar_frames:
            lidar_frame.load_data()
            yield lidar_frame

    def get_radar(self, idx):
        self.radar_frames[idx].load_data()
        return self.radar_frames[idx]

    @property
    def radar(self):
        for radar_frame in radar_frames:
            radar_frame.load_data()
            yield radar_frame

    def _check_dataroot_valid(self, dataroot):
        if not osp.isdir(self.applanix_root):
            raise ValueError("Error: applanix dir missing from dataroot")
        if not osp.isdir(self.calib_root):
            raise ValueError("Error: calib dir missing from dataroot")
        if not osp.isdir(self.camera_root):
            os.mkdir(self.camera_root)
        if not osp.isdir(self.lidar_root):
            os.mkdir(self.lidar_root)
        if not osp.isdir(self.radar_root):
            os.mkdir(self.radar_root)

    def _get_frames(self, posefile, root, ext, SensorType):
        frames = []
        if osp.exists(posefile) and osp.isdir(root):
            with open(posefile, 'r') as f:
                f.readline()  # header
                for line in f:
                    data = line.split(',')
                    ts = data[0]
                    if self.start_ts <= ts and ts <= self.end_ts:
                        frame = SensorType(osp.join(root, ts + ext))
                        frame.init_pose(data)
                        frames.append(frame)
        return frames
