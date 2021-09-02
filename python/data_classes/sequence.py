from os import listdir, path

from calib import Calib
from sensors import Camera, Lidar, Radar


class Sequence:
    def __init__(self, boreas_root, seqSpec):
        self.seqID = seqSpec[0]
        self.seq_root = path.join(boreas_root, self.seqID)
        self._check_dataroot_valid(self.seq_root)

        self.camera_root = path.join(self.seq_root, 'camera')
        self.lidar_root = path.join(self.seq_root, 'lidar')
        self.radar_root = path.join(self.seq_root, 'radar')
        self.start_ts = str(seqSpec[1])
        self.end_ts = str(seqSpec[2])

        self.timestamps = sorted([int(path.splitext(f)[0]) for f in listdir(self.lidar_root)])  # Currently syncing to lidar timestamps
        self.seq_len = len(self.timestamps)
        self.lidar_paths = self._get_datapaths(self.lidar_root, self.start_ts, self.end_ts)
        self.camera_paths = self._get_datapaths(self.camera_root, self.start_ts, self.end_ts)
        self.radar_paths = self._get_datapaths(self.radar_root, self.start_ts, self.end_ts)
        self.calib = Calib(boreas_root + self.seqID + '/calib/')

        self.camera_synced = self._sync_camera_frames()

    # TODO: load printable metadata string

    @property
    def cam0(self):
        for f in self.camera_paths:
            yield Camera(f)

    def get_camera(self, idx):
        return Camera(self.camera_paths[idx])

    @property
    def lidar(self):
        for f in self.lidar_paths:
            yield Lidar(f)

    def get_lidar(self, idx):
        return Lidar(self.lidar_paths[idx])

    @property
    def radar(self):
        for f in self.radar_paths:
            yield Radar(f)

    def get_radar(self, idx):
        return Radar(self.radar_paths[idx])

    def visualize(self):
        pass

    # TODO: generate video for the entire sequences -> use boreas visualizer
    # option 1: display video, option 2: save video to file

    def get_pose(self, sensType, timestamp):
        pass
    # TODO

    def _check_dataroot_valid(self, dataroot):
        # Check if dataroot paths are valid
        if not path.exists(path.join(dataroot, "camera")):
            raise ValueError("Error: images dir missing from dataroot")
        if not path.exists(path.join(dataroot, "lidar")):
            raise ValueError("Error: lidar dir missing from dataroot")
        if not path.exists(path.join(dataroot, "applanix")):
            raise ValueError("Error: applnix dir missing from dataroot")
        if not path.exists(path.join(dataroot, "calib")):
            raise ValueError("Error: calib dir missing from dataroot")
        # if not path.exists(path.join(dataroot, "labels.json")):
        #     raise ValueError("Error: labels.json missing from dataroot")

    def _get_datapaths(self, dataroot, start_ts, end_ts):
        res = []
        for filename in listdir(dataroot):
            timestamp = path.splitext(filename)[0]
            if start_ts <= timestamp <= end_ts:  # Note, we are comparing strings here
                res.append(path.join(dataroot, filename))
        return sorted(res)

    def _sync_camera_frames(self):
        # Helper function for finding closest timestamp
        def get_closest_ts(query_time, targets):
            min_delta = 1e33  # Temp set to this, should be 1e9
            closest = -1
            for i in range(len(targets)):
                delta = abs(query_time - targets[i])
                if delta < min_delta:
                    min_delta = delta
                    closest = i
            assert (closest >= 0), "closest time to query: {} in rostimes not found.".format(query_time)
            return closest, targets[closest]

        # Find closest lidar timestamp for each camera frame
        res = []
        camera_timestamps = [int(f.replace('/', '.').split('.')[-2]) for f in self.camera_paths]
        for i in range(self.seq_len):
            closet_idx, closest_val = get_closest_ts(self.timestamps[i], camera_timestamps)
            res.append(self.camera_paths[closet_idx])
        return res

if __name__ == "__main__":
    # for debugging
    seq = Sequence("/home/shichen/datasets/", ["boreas_mini", 1606417230037163312, 1606417239986391931])