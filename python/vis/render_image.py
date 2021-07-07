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


def get_sensor_calibration(P_cam_file, T_vi_file, T_vc_file, T_vr_file):
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
    T_vi = np.loadtxt(T_vi_file)
    T_vc = np.loadtxt(T_vc_file)
    T_vr = np.loadtxt(T_vr_file)
    print(P_cam)
    print(T_vi)
    print(T_vc)
    print(T_vr)
    print('---------inv(T_vi)*T_vr----------')
    m1 = np.matmul(np.linalg.inv(T_vi),T_vr)
    print(m1)
    vec1 = R.from_matrix(m1[0:3,0:3]).as_rotvec()
    print(vec1)
    print('----------inv(T_vc)*T_vr---------')
    m2 = np.matmul(np.linalg.inv(T_vc),T_vr)
    print(m2)
    vec2 = R.from_matrix(m2[0:3,0:3]).as_rotvec()
    print(vec2)
    
    print('----------inv(T_vc)*T_vi---------')
    m3 = np.matmul(np.linalg.inv(T_vc),T_vi)
    print(m3)
    vec3 = R.from_matrix(m3[0:3,0:3]).as_rotvec()
    print(vec3)
    print('-------------------')
    
    a = R.from_matrix(T_vi[0:3,0:3])
    print(a.as_rotvec())

    b = R.from_matrix(T_vr[0:3,0:3])
    print(b.as_rotvec())

    c = R.from_matrix(np.matmul(np.linalg.inv(T_vi),T_vc)[0:3,0:3])
    print(c.as_rotvec())
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

