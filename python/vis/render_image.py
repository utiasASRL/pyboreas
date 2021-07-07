# TODO: transform bounding box into sensor frame at time
# TODO: project 3D bounding box onto 2D images (radar/lidar, camera front-face)
# TODO: render visualization as an image, (display it), (save it)
# TODO: plot odometry results vs. ground truth

import json
import glob

from matplotlib import cm
import numpy as np

import vis_utils
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def get_sensor_calibration(P_cam_file, T_iv_file, T_cv_file, T_rv_file):
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
    print('---------P_cam----------')
    print(P_cam)
    
    print('---------T_iv----------')
    print(T_iv)
    vec_iv = R.from_matrix(T_iv[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_iv)
    print('---------T_cv----------')
    print(T_cv)
    vec_cv = R.from_matrix(T_cv[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_cv)
    print('---------T_rv----------')
    print(T_rv)
    vec_rv = R.from_matrix(T_rv[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_rv)
    print('---------T_iv*inv(T_rv)=T_ir----------')
    T_ir = np.matmul(T_iv,np.linalg.inv(T_rv))
    print(T_ir)
    vec_ir = R.from_matrix(T_ir[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_ir)
    print('----------T_cv*inv(T_rv)=T_cr---------')
    T_cr = np.matmul(T_cv,np.linalg.inv(T_rv))
    print(T_cr)
    vec_cr = R.from_matrix(T_cr[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_cr)
    print('----------T_cv*inv(T_iv)=T_ci---------')
    T_ci = np.matmul(T_cv,np.linalg.inv(T_iv))
    print(T_ci)
    vec_ci = R.from_matrix(T_ci[0:3,0:3]).as_euler('zyx', degrees=True)
    print(vec_ci)
    print('-------------------')

def load_cubloids(label_file, idx):

    return True

if __name__ == '__main__':
    get_sensor_calibration("./calib/P_camera.txt","./calib/T_applanix_lidar.txt","./calib/T_camera_lidar.txt","./calib/T_radar_lidar.txt")
    load_cubloids("./samle_dataset/labels.json",0)

    data_files = sorted(glob.glob('./sample_dataset/data/task_point_cloud*.json'))  
    file = open(data_files[0], 'r')
    T, C = vis_utils.get_device_pose(json.load(file))
    print(T)
    print(C)

