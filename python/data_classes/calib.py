import os.path as osp
import numpy as np

from utils.utils import rotToYawPitchRoll

class Calib:
    def __init__(self, calib_root):
        self.P0 = np.loadtxt(osp.join(calib_root, 'P_camera.txt'))
        self.T_applanix_lidar = np.loadtxt(osp.join(calib_root, 'T_applanix_lidar.txt'))
        self.T_camera_lidar = np.loadtxt(osp.join(calib_root, 'T_camera_lidar.txt'))
        self.T_radar_lidar = np.loadtxt(osp.join(calib_root, 'T_radar_lidar.txt'))

    def print_sensor_calibration(self): 
        vec_al = rotToYawPitchRoll(self.T_applanix_lidar[:3, :3]) * 180 / np.pi
        vec_cl = rotToYawPitchRoll(self.T_camera_lidar[:3, :3]) * 180 / np.pi
        vec_rl = rotToYawPitchRoll(self.T_radar_lidar[:3, :3]) * 180 / np.pi
        print('P0:')
        print(self.P0)
        print('T_applanix_lidar:')
        print(self.T_applanix_lidar)
        print(vec_al)
        print('T_camera_lidar:')
        print(self.T_camera_lidar)
        print(vec_cl)
        print('T_radar_lidar:')
        print(self.T_radar_lidar)
        print(vec_rl)
