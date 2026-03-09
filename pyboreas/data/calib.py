import os.path as osp

import numpy as np
import yaml


class Calib:
    """
    Class for loading and storing calibration matrices.
    """

    def __init__(self, calib_root):
        self.P0 = np.loadtxt(osp.join(calib_root, "P_camera.txt"))
        self.T_applanix_lidar = np.loadtxt(osp.join(calib_root, "T_applanix_lidar.txt"))
        self.T_camera_lidar = np.loadtxt(osp.join(calib_root, "T_camera_lidar.txt"))
        self.T_radar_lidar = np.loadtxt(osp.join(calib_root, "T_radar_lidar.txt"))
        # Load in optional calibrations for Boreas-RT data if they exist
        self.T_applanix_wheel = np.eye(4)
        self.T_applanix_dmu = np.eye(4)
        self.T_aeva_lidar = np.eye(4)
        self.T_imu_aeva = np.eye(4)
        self.wheel_radius = 0.34    # [m], default internal wheel radius for converting encoder ticks to distance
        self.radar_offset = -0.31   # [m], default internal radar range offset
        self.radar_doppler_beta = 0.049

        if osp.exists(osp.join(calib_root, "T_applanix_wheel.txt")):
            self.T_applanix_wheel = np.loadtxt(osp.join(calib_root, "T_applanix_wheel.txt"))
        if osp.exists(osp.join(calib_root, "T_applanix_dmu.txt")):
            self.T_applanix_dmu = np.loadtxt(osp.join(calib_root, "T_applanix_dmu.txt"))
        if osp.exists(osp.join(calib_root, "T_aeva_lidar.txt")):
            self.T_aeva_lidar = np.loadtxt(osp.join(calib_root, "T_aeva_lidar.txt"))
        if osp.exists(osp.join(calib_root, "T_imu_aeva.txt")):
            self.T_imu_aeva = np.loadtxt(osp.join(calib_root, "T_imu_aeva.txt"))
        if osp.exists(osp.join(calib_root, "misc_calibrations.yaml")):
            with open(osp.join(calib_root, "misc_calibrations.yaml"), 'r') as f:
                misc_calib = yaml.safe_load(f)
                self.wheel_radius = misc_calib.get('wheel_radius', self.wheel_radius)
                self.radar_offset = misc_calib.get('radar_offset', self.radar_offset)
                self.radar_doppler_beta = misc_calib.get('radar_doppler_beta', self.radar_doppler_beta)


    def print_calibration(self):
        print("P0:")
        print(self.P0)
        print("T_applanix_lidar:")
        print(self.T_applanix_lidar)
        print("T_camera_lidar:")
        print(self.T_camera_lidar)
        print("T_radar_lidar:")
        print(self.T_radar_lidar)
        print("T_applanix_wheel:")
        print(self.T_applanix_wheel)
        print("T_applanix_dmu:")
        print(self.T_applanix_dmu)
        print("T_aeva_lidar:")
        print(self.T_aeva_lidar)
        print("T_imu_aeva:")
        print(self.T_imu_aeva)
        print("wheel_radius: {} m".format(self.wheel_radius))
        print("radar_offset: {} m".format(self.radar_offset))
        print("radar_doppler_beta: {}".format(self.radar_doppler_beta))