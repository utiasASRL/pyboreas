import os.path as path

import numpy as np
from scipy.spatial.transform import Rotation as R


class BoreasTransforms:
    def __init__(self, calib_dir):
        self.calib_dir = calib_dir
        self.P_cam, self.T_iv, self.T_cv = self._get_sensor_calibration_all(verbose=False)
        self.C_enu_ned = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

    def _get_sensor_calibration_all(self, verbose=False):
        """
        Helper function to get all sensor calibration data (camera, lidar and radar)
         :return P_cam: camera intrinsics matrix
                T_vi: imu to velodyne extrinsics matrix
                T_vc: camera to velodyne extrinsics matrix
                T_vr: radar to velodyne extrinsics matrix
        """
        # Get paths
        P_cam_file = path.join(self.calib_dir, "P_camera.txt")
        T_iv_file = path.join(self.calib_dir, "T_applanix_lidar.txt")
        T_cv_file = path.join(self.calib_dir, "T_camera_lidar.txt")
        T_rv_file = path.join(self.calib_dir, "T_radar_lidar.txt")

        return self.get_sensor_calibration(P_cam_file, T_iv_file, T_cv_file, T_rv_file, verbose)

    def get_sensor_calibration(self, P_cam_file, T_iv_file, T_cv_file, T_rv_file, verbose=False):
        """
        Extract sensor calibration data (camera, lidar and radar)
         :param P_cam_file: file containing camera intrinsics file
         :param T_vi_file: file containing imu to velodyne extrinsics
         :param T_vc_file: file containing camera to velodyne extrinsics
         :param T_vr_file: file containing radar to velodyne extrinsics
         :return P_cam: camera intrinsics matrix
                T_vi: imu to velodyne extrinsics matrix
                T_vc: camera to velodyne extrinsics matrix
                T_vr: radar to velodyne extrinsics matrix
        """
        # Load data
        P_cam = np.loadtxt(P_cam_file)
        T_iv = np.loadtxt(T_iv_file)
        T_cv = np.loadtxt(T_cv_file)
        T_rv = np.loadtxt(T_rv_file)

        vec_iv = R.from_matrix(T_iv[0:3, 0:3]).as_euler('zyx', degrees=True)
        vec_cv = R.from_matrix(T_cv[0:3, 0:3]).as_euler('zyx', degrees=True)
        vec_rv = R.from_matrix(T_rv[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_ir = np.matmul(T_iv, np.linalg.inv(T_rv))
        vec_ir = R.from_matrix(T_ir[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_cr = np.matmul(T_cv, np.linalg.inv(T_rv))
        vec_cr = R.from_matrix(T_cr[0:3, 0:3]).as_euler('zyx', degrees=True)

        T_ci = np.matmul(T_cv, np.linalg.inv(T_iv))
        vec_ci = R.from_matrix(T_ci[0:3, 0:3]).as_euler('zyx', degrees=True)

        if verbose:
            print('---------P_cam----------')
            print(P_cam)
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
        return P_cam, T_iv, T_cv