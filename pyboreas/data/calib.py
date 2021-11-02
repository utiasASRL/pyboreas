import os.path as osp
import numpy as np


class Calib:
    """
    Class for loading and storing calibration matrices.
    """
    def __init__(self, calib_root):
        self.P0 = np.loadtxt(osp.join(calib_root, 'P_camera.txt'))
        self.T_applanix_lidar = np.loadtxt(osp.join(calib_root, 'T_applanix_lidar.txt'))
        self.T_camera_lidar = np.loadtxt(osp.join(calib_root, 'T_camera_lidar.txt'))
        self.T_radar_lidar = np.loadtxt(osp.join(calib_root, 'T_radar_lidar.txt'))

    def print_calibration(self):
        print('P0:')
        print(self.P0)
        print('T_applanix_lidar:')
        print(self.T_applanix_lidar)
        print('T_camera_lidar:')
        print(self.T_camera_lidar)
        print('T_radar_lidar:')
        print(self.T_radar_lidar)
