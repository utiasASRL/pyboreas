import os
import os.path as osp

from pyboreas.data.calib import Calib
from pyboreas.data.sensors import Camera, Lidar, Radar
from pyboreas.utils.utils import get_closest_index


class Sequence:
    """
    Class for working with an individual Boreas dataset sequence
    """
    def __init__(self, boreas_root, seqSpec, verbose=False):
        """init
        Args:
            boreas_root (str): path to root folder ex: /path/to/data/boreas/
            seqSpec (list): defines sequence ID, start_time, and end_time
            verbose (bool): whether or not to print info during initialization
        """
        self.ID = seqSpec[0]
        self.verbose = verbose
        if verbose:
            print('SEQ: {}'.format(self.ID))
        if len(seqSpec) > 2:
            assert seqSpec[2] > seqSpec[1], 'Sequence timestamps must go forward in time'
            self.start_ts = str(seqSpec[1])
            self.end_ts = str(seqSpec[2])
            if verbose:
                print('START: {} END: {}'.format(self.start_ts, self.end_ts))
        else:
            self.start_ts = '0'  # dummy start and end if not specified
            self.end_ts = '9' * 21
        self.seq_root = osp.join(boreas_root, self.ID)
        self.applanix_root = osp.join(self.seq_root, 'applanix')
        self.calib_root = osp.join(self.seq_root, 'calib')
        self.camera_root = osp.join(self.seq_root, 'camera')
        self.lidar_root = osp.join(self.seq_root, 'lidar')
        self.radar_root = osp.join(self.seq_root, 'radar')

        self._check_dataroot_valid()  # Check if folder structure correct

        self.calib = Calib(self.calib_root)
        # Creates list of frame objects for cam, lidar, radar, and inits poses
        self.get_all_frames()

        self._check_download()  # if verbose, prints warning when sensor data missing

        if verbose:
            print('camera frames: {}'.format(len(self.camera_frames)))
            print('lidar frames: {}'.format(len(self.lidar_frames)))
            print('radar frames: {}'.format(len(self.radar_frames)))
            print('-------------------------------')

    def get_camera(self, idx):
        self.camera_frames[idx].load_data()
        return self.camera_frames[idx]

    @property
    def camera(self):
        for camera_frame in self.camera_frames:
            camera_frame.load_data()
            yield camera_frame

    def get_camera_iter(self):
        """Retrieves an iterator on camera frames"""
        return iter(self.camera)

    def get_lidar(self, idx):
        self.lidar_frames[idx].load_data()
        return self.lidar_frames[idx]

    @property
    def lidar(self):
        for lidar_frame in self.lidar_frames:
            lidar_frame.load_data()
            yield lidar_frame

    def get_lidar_iter(self):
        """Retrieves an iterator on lidar frames"""
        return iter(self.lidar)

    def get_radar(self, idx):
        self.radar_frames[idx].load_data()
        return self.radar_frames[idx]

    @property
    def radar(self):
        for radar_frame in self.radar_frames:
            radar_frame.load_data()
            yield radar_frame

    def get_radar_iter(self):
        """Retrieves an iterator on radar frames"""
        return iter(self.radar)

    def _check_dataroot_valid(self):
        """Checks if the sequence folder structure is valid"""
        if not osp.isdir(self.applanix_root):
            raise ValueError("ERROR: applanix dir missing from dataroot")
        if not osp.isdir(self.calib_root):
            raise ValueError("ERROR: calib dir missing from dataroot")
        if not osp.isdir(self.camera_root):
            os.mkdir(self.camera_root)
        if not osp.isdir(self.lidar_root):
            os.mkdir(self.lidar_root)
        if not osp.isdir(self.radar_root):
            os.mkdir(self.radar_root)

    def _check_download(self):
        """Checks if all sensor data has been downloaded, prints a warning otherwise"""
        if len(os.listdir(self.camera_root)) < len(self.camera_frames):
            print('WARNING: camera images are not all downloaded')
        if len(os.listdir(self.lidar_root)) < len(self.lidar_frames):
            print('WARNING: lidar frames are not all downloaded')
        if len(os.listdir(self.radar_root)) < len(self.radar_frames):
            print('WARNING: radar scans are not all downloaded')
        gtfile = osp.join(self.applanix_root, 'gps_post_process.csv')
        if not osp.exists(gtfile):
            print('WARNING: this may be a test sequence, or the groundtruth is not yet available')

    def _get_frames(self, posefile, root, ext, SensorType):
        """Initializes sensor frame objects with their ground truth pose information
        Args:
            posefile (str): path to ../sensor_poses.csv
            root (str): path to the root of the sensor folder ../sensor/
            ext (str): file extension specific to this sensor type
            SensorType (cls): sensor class specific to this sensor type
        Returns:
            frames (list): list of sensor frame objects
        """
        frames = []
        if not osp.isdir(root):
            return frames
        if osp.exists(posefile):
            with open(posefile, 'r') as f:
                f.readline()  # header
                for line in f:
                    data = line.split(',')
                    ts = data[0]
                    if self.start_ts <= ts and ts <= self.end_ts:
                        frame = SensorType(osp.join(root, ts + ext))
                        frame.init_pose(data)
                        frames.append(frame)
        else:
            framenames = sorted([f for f in os.listdir(root) if ext in f])
            for framename in framenames:
                ts = framename.split(',')[0]
                if self.start_ts <= ts and ts <= self.end_ts:
                    frames.append(SensorType(osp.join(root, framename)))
        return frames

    def get_all_frames(self):
        """Convenience method for retrieving sensor frames of all types"""
        cfile = osp.join(self.applanix_root, 'camera_poses.csv')
        lfile = osp.join(self.applanix_root, 'lidar_poses.csv')
        rfile = osp.join(self.applanix_root, 'radar_poses.csv')
        self.camera_frames = self._get_frames(cfile, self.camera_root, '.png', Camera)
        self.lidar_frames = self._get_frames(lfile, self.lidar_root, '.bin', Lidar)
        self.radar_frames = self._get_frames(rfile, self.radar_root, '.png', Radar)

    def reset_frames(self):
        """Resets all frames, removes downloaded data"""
        self.get_all_frames()

    def synchronize_frames(self, ref='camera'):
        """Simulates having synchronous measurements
        Note: measurements still won't be at the exact same timestamp and will have different poses
        However, for a given reference index, the other measurements will be as close to the reference
        in time as they can be.
        Args:
            ref (str): [camera, lidar, or radar] this determines which sensor's frames will be used as the
                reference for synchronization. This sensor's list of frames will not be modified. However,
                the other two list of sensor frames will be modified so that each index will approximately
                align with the reference in time.
        """
        cstamps = [frame.timestamp for frame in self.camera_frames]
        lstamps = [frame.timestamp for frame in self.lidar_frames]
        rstamps = [frame.timestamp for frame in self.radar_frames]

        def get_closest(query_time, target_times, targets):
            closest = get_closest_index(query_time, target_times)
            assert(abs(query_time - target_times[closest]) < 1.0), 'query: {}'.format(query_time)
            return targets[closest]

        if ref == 'camera':
            self.lidar_frames = [get_closest(cstamp, lstamps, self.lidar_frames) for cstamp in cstamps]
            self.radar_frames = [get_closest(cstamp, rstamps, self.radar_frames) for cstamp in cstamps]
        elif ref == 'lidar':
            self.camera_frames = [get_closest(lstamp, cstamps, self.camera_frames) for lstamp in lstamps]
            self.radar_frames = [get_closest(lstamp, rstamps, self.radar_frames) for lstamp in lstamps]
        elif ref == 'radar':
            self.camera_frames = [get_closest(rstamp, cstamps, self.camera_frames) for rstamp in rstamps]
            self.lidar_frames = [get_closest(rstamp, lstamps, self.lidar_frames) for rstamp in rstamps]