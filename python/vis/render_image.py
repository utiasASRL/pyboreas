# TODO: transform bounding box into sensor frame at time
# TODO: project 3D bounding box onto 2D images (radar/lidar, camera front-face)
# TODO: render visualization as an image, (display it), (save it)
# TODO: plot odometry results vs. ground truth

import json
import glob
import cv2

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import vis_utils
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def get_closest(query_time, targets):
    min_delta = 1e9
    closest = -1
    for i in range(len(targets)):
        delta = abs(query_time - targets[i])
        if delta < min_delta:
            min_delta = delta
            closest = i
    assert(closest >= 0), "closest time to query: {} in rostimes not found.".format(query_time)
    return closest, targets[closest]

def sync_camera_lidar(data_file_paths, camera_file_paths):
    
    camera_timestamps = [int(f.replace('/','.').split('.')[-2]) for f in camera_file_paths]
    sync_map = []
    for i in range(len(data_file_paths)):
        data_file = open(data_file_paths[i], 'r')
        data_json = json.load(data_file)
        timestamp, corrected_timestamp = vis_utils.get_camera_timestamp(data_json)

        closet_idx, cloest_val = get_closest(corrected_timestamp, camera_timestamps)
        # print(i, closet_idx, timestamp, corrected_timestamp, cloest_val)
        # print("   off by ", abs(cloest_val - data_timestamp)*1e-11)
        sync_map.append(camera_file_paths[closet_idx])

    return sync_map


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
    return P_cam, T_iv, T_cv

def temp_transform(pcd):
    pcd = np.matmul(vis_utils.to_T(vis_utils.rot_z(-np.pi/2), np.zeros((3,1))), np.matmul(np.linalg.inv(T_cv), pcd))
    pcd = np.matmul(vis_utils.to_T(vis_utils.rot_y(np.pi), np.zeros((3,1))), pcd)
    return pcd

def render_image(label_file_path, data_file_paths, synced_cameras, idx, P_cam, T_iv, T_cv):
    label_file = open(label_file_path, 'r')
    data_file = open(data_file_paths[idx], 'r')

    label_json = json.load(label_file)
    data_json = json.load(data_file)

    T, C = vis_utils.get_device_pose(data_json)

    cubloids_raw = label_json[idx]
    points, boxes = vis_utils.transform_data_to_sensor_frame(data_json, label_json[idx]['cuboids'])
    #print(len(label_json))
    #print(len(cubloids_raw['cuboids']))
    #print(cubloids_raw['cuboids'])

    image_file = synced_cameras[idx]
    #print(image_file)
    image = cv2.imread(image_file, cv2.IMREAD_COLOR) 
    
    points = points.T
    #print(points)
    #print(boxes[0])
    
    # Lidar
    #points_camera_all = np.matmul(T_cv, np.matmul(np.linalg.inv(T_iv), points))
    points_camera_all = temp_transform(points)
    points_camera = np.array([])
    for i in range(points_camera_all.shape[1]):
        if points_camera_all[2,i] > 0:
            points_camera = np.concatenate((points_camera, points_camera_all[:,i]))
    points_camera = np.reshape(points_camera, (-1,4)).T
    pixel_camera = np.matmul(P_cam, points_camera)

    max_z = int(max(pixel_camera[2,:])/3)
    for i in range(pixel_camera.shape[1]):
        z = pixel_camera[2,i]
        x = int(pixel_camera[0,i] / z)
        y = int(pixel_camera[1,i] / z)
        if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
            c = cv2.applyColorMap(np.array([int(pixel_camera[2,i] / max_z*255)], dtype=np.uint8), cv2.COLORMAP_RAINBOW).squeeze().tolist()
            cv2.circle(image,(x,y), 1, c, 1)
    
    # Bounding boxes
    centroids_odom = np.array([]).reshape(3,0)
    for box in boxes:
        bb = box.get_raw_data()
        centroids_odom = np.hstack((centroids_odom, bb[0]))
    centroids_odom = np.vstack((centroids_odom, np.ones((1, len(boxes)))))
    centroids_camera_all = temp_transform(centroids_odom)
    centroids_camera = np.array([])
    for i in range(centroids_camera_all.shape[1]):
        if centroids_camera_all[2,i] > 0:
            centroids_camera = np.concatenate((centroids_camera, centroids_camera_all[:,i]))
    centroids_camera = np.reshape(centroids_camera, (-1,4)).T
    pixel_centroid_camera = np.matmul(P_cam, centroids_camera)
    for i in range(pixel_centroid_camera.shape[1]):
        z = pixel_centroid_camera[2,i]
        x = int(pixel_centroid_camera[0,i] / z)
        y = int(pixel_centroid_camera[1,i] / z)
        if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
            cv2.circle(image,(x,y), 5, [255,255,255], 10)

    cv2.destroyAllWindows()
    cv2.imshow(image_file, image) 
    cv2.waitKey(100)
    print()
     


if __name__ == '__main__':
    P_cam, T_iv, T_cv = get_sensor_calibration("./calib/P_camera.txt","./calib/T_applanix_lidar.txt","./calib/T_camera_lidar.txt","./calib/T_radar_lidar.txt")
    
    idx = 50
    label_file_path = "./sample_dataset/labels.json"
    data_file_paths = sorted(glob.glob('./sample_dataset/lidar_data/task_point_cloud*.json'), key=lambda x : int(''.join(filter(str.isdigit, x))))
    camera_file_paths = sorted(glob.glob('./sample_dataset/camera/*.png'), key=lambda x : int(''.join(filter(str.isdigit, x))))

    #synced_cameras = sync_camera_lidar(data_file_paths, camera_file_paths)
    #print(synced_cameras)
    for id in range(50):
        render_image(label_file_path, data_file_paths, camera_file_paths, id, P_cam, T_iv, T_cv)
