import os.path as path

import numpy as np
from scipy.spatial.transform import Rotation as R


class Calib:
    def __init__(self, calib_dir):
        self.calib_dir = calib_dir

        self.P0 = np.loadtxt(path.join(calib_dir, 'P_camera.txt'))
        self.T_applanix_lidar = np.loadtxt(path.join(calib_dir, 'T_applanix_lidar.txt'))
        self.T_camera_lidar = np.loadtxt(path.join(calib_dir, 'T_camera_lidar.txt'))
        self.T_radar_lidar = np.loadtxt(path.join(calib_dir, 'T_radar_lidar.txt'))
        self.C_enu_ned = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

    def print_sensor_calibration(self):
        """
        Print debug info for sensor calibration data (camera, lidar and radar)
        """
        # Load transforms
        T_iv = self.T_applanix_lidar
        T_cv = self.T_camera_lidar
        T_rv = self.T_radar_lidar

        vec_iv = R.from_matrix(T_iv[0:3, 0:3]).as_euler('zyx', degrees=True)
        vec_cv = R.from_matrix(T_cv[0:3, 0:3]).as_euler('zyx', degrees=True)
        vec_rv = R.from_matrix(T_rv[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_ir = np.matmul(T_iv, np.linalg.inv(T_rv))
        vec_ir = R.from_matrix(T_ir[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_cr = np.matmul(T_cv, np.linalg.inv(T_rv))
        vec_cr = R.from_matrix(T_cr[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_ci = np.matmul(T_cv, np.linalg.inv(T_iv))
        vec_ci = R.from_matrix(T_ci[0:3, 0:3]).as_euler('zyx', degrees=True)

        print('---------P_cam----------')
        print(self.P0)
        print('---------T_iv----------')
        print(T_iv)
        print(vec_iv)
        print('---------T_cv----------')
        print(T_cv)
        print(vec_cv)
        print('---------T_rv----------')
        print(T_rv)
        print(vec_rv)
        print('---------T_iv*inv(T_rv)=T_ir----------')
        print(T_ir)
        print(vec_ir)
        print('----------T_cv*inv(T_rv)=T_cr---------')
        print(T_cr)
        print(vec_cr)
        print('----------T_cv*inv(T_iv)=T_ci---------')
        print(T_ci)
        print(vec_ci)
        print('-------------------')
