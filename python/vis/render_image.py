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

debug = False

def get_closest(query_time, targets, start=0):
    min_delta = 1e11
    closest = -1
    for i in range(start, len(targets)):
        delta = abs(query_time - targets[i])
        if delta < min_delta:
            min_delta = delta
            closest = i
    assert(closest >= 0), "closest time to query: {} in rostimes not found.".format(query_time)
    return closest, targets[closest]

def sync_camera_lidar(data_file_paths, camera_file_paths, offset=0):
    print("Syncing Camera-Lidar Data...")
    camera_timestamps = [int(f.replace('/','.').split('.')[-2]) for f in camera_file_paths]
    sync_map = []
    closet_idx = 0
    for i in tqdm(range(len(data_file_paths))):
        data_file = open(data_file_paths[i], 'r')
        data_json = json.load(data_file)
        timestamp, corrected_timestamp = vis_utils.get_camera_timestamp(data_json, offset)

        closet_idx, cloest_val = get_closest(corrected_timestamp, camera_timestamps, closet_idx)
        if debug:
            print(i, closet_idx, timestamp, corrected_timestamp, cloest_val)
            print("   off by ", abs(cloest_val - corrected_timestamp)*1e-9)
        sync_map.append(camera_file_paths[closet_idx])

    return sync_map

def temp_transform(T_cv, T_iv, pcd):
    pcd = np.matmul(T_cv, np.matmul(np.linalg.inv(T_iv), pcd))
    return pcd

def to_pixel(cam_matrix, point_camera):
    point_camera = np.reshape(point_camera, (-1,4)).T
    pixel_point_camera = np.matmul(cam_matrix, point_camera)

    z = pixel_point_camera[2]
    x = int(pixel_point_camera[0] / z)
    y = int(pixel_point_camera[1] / z)
    return [x, y, float(z)]

def draw_point(image, cam_matrix, point_camera, color, diameter, line_width):
    if point_camera[2] > 0:
        pixels = to_pixel(cam_matrix, point_camera)
        if pixels[0] > 0 and pixels[0] < image.shape[0] and pixels[1] > 0 and pixels[1] < image.shape[1]:
            cv2.circle(image,(pixels[0],pixels[1]), diameter, color, line_width)

def get_point_with_offset(pose, offset):
    T = np.eye(4)
    T[0:3,3] = np.array([offset[0],offset[1],offset[2]])
    return np.matmul(pose, T)

def get_box_corners(box):
    centroid_pose = vis_utils.to_T(box.rot, box.pos)
    dims = box.extent.reshape(-1)/2
    corners = {}

    front_top_left = get_point_with_offset(centroid_pose, [-dims[0],dims[1],dims[2]])
    front_top_right = get_point_with_offset(centroid_pose, [dims[0],dims[1],dims[2]])

    front_bottom_left = get_point_with_offset(centroid_pose, [-dims[0],dims[1],-dims[2]])
    front_bottom_right = get_point_with_offset(centroid_pose, [dims[0],dims[1],-dims[2]])

    back_top_left = get_point_with_offset(centroid_pose, [-dims[0],-dims[1],dims[2]])
    back_top_right = get_point_with_offset(centroid_pose, [dims[0],-dims[1],dims[2]])

    back_bottom_left = get_point_with_offset(centroid_pose, [-dims[0],-dims[1],-dims[2]])
    back_bottom_right = get_point_with_offset(centroid_pose, [dims[0],-dims[1],-dims[2]])

    corners['ftl'] = front_top_left
    corners['ftr'] = front_top_right
    corners['fbl'] = front_bottom_left
    corners['fbr'] = front_bottom_right
    corners['btl'] = back_top_left
    corners['btr'] = back_top_right
    corners['bbl'] = back_bottom_left
    corners['bbr'] = back_bottom_right

    return corners

def draw_box(image, T_cv, T_iv, cam_matrix, box, color, line_width, draw_corner_pts = False):
    corners = get_box_corners(box)

    corners_camera = {}
    corners_pixel = {}
    for key, value in corners.items():
        T_camera = temp_transform(T_cv, T_iv, value)
        p_camera = T_camera[:,3]
        if p_camera[2] <= 1e-5:
            return {}
        pixel = to_pixel(cam_matrix, p_camera)
        if pixel[0] < 0 or pixel[0] > image.shape[0] or \
             pixel[1] < 0 or pixel[1] > image.shape[1]:
            return {}
        corners_camera[key] = p_camera
        corners_pixel[key] = tuple(pixel[0:2])

    if (draw_corner_pts):
        draw_point(image, cam_matrix, corners_camera['ftl'], [0, 0, 255], 3,4) # [b,g,r]
        draw_point(image, cam_matrix, corners_camera['ftr'], [0, 255, 0], 3,4)
        draw_point(image, cam_matrix, corners_camera['fbl'], [255, 0, 0], 3,4)
        draw_point(image, cam_matrix, corners_camera['fbr'], [255, 255, 0], 3,4)
        draw_point(image, cam_matrix, corners_camera['btl'], [0, 0, 128], 3,4)
        draw_point(image, cam_matrix, corners_camera['btr'], [0, 128, 0], 3,4)
        draw_point(image, cam_matrix, corners_camera['bbl'], [128, 0, 0], 3,4)
        draw_point(image, cam_matrix, corners_camera['bbr'], [255, 255, 0], 3,4)

    cv2.line(image,corners_pixel['ftl'],corners_pixel['ftr'],color,line_width)
    cv2.line(image,corners_pixel['ftl'],corners_pixel['fbl'],color,line_width)
    cv2.line(image,corners_pixel['ftr'],corners_pixel['fbr'],color,line_width)
    cv2.line(image,corners_pixel['fbl'],corners_pixel['fbr'],color,line_width)

    cv2.line(image,corners_pixel['ftl'],corners_pixel['fbr'],color,line_width)
    cv2.line(image,corners_pixel['ftr'],corners_pixel['fbl'],color,line_width)

    cv2.line(image,corners_pixel['btl'],corners_pixel['btr'],color,line_width)
    cv2.line(image,corners_pixel['btl'],corners_pixel['bbl'],color,line_width)
    cv2.line(image,corners_pixel['btr'],corners_pixel['bbr'],color,line_width)
    cv2.line(image,corners_pixel['bbl'],corners_pixel['bbr'],color,line_width)

    cv2.line(image,corners_pixel['ftl'],corners_pixel['btl'],color,line_width)
    cv2.line(image,corners_pixel['ftr'],corners_pixel['btr'],color,line_width)
    cv2.line(image,corners_pixel['fbl'],corners_pixel['bbl'],color,line_width)
    cv2.line(image,corners_pixel['fbr'],corners_pixel['bbr'],color,line_width)
    return corners_pixel
    
def draw_label(image, corners_pixel, color, box):
    if len(corners_pixel) > 0:
        pos = (corners_pixel['ftl'][0], max(0, corners_pixel['ftl'][1]-10))
        text = box.label
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA, False)

def render_image(label_json, data_file_paths, synced_cameras, start_idx, idx, P_cam, T_iv, T_cv, write_to_disk=False):
    
    data_file = open(data_file_paths[idx], 'r')
    data_json = json.load(data_file)

    points, boxes = vis_utils.transform_data_to_sensor_frame(data_json, label_json[start_idx+idx]['cuboids'], 0.7, T_iv)

    image_file = synced_cameras[idx]
    image = cv2.imread(image_file, cv2.IMREAD_COLOR) 
    
    points = points.T
    
    # Lidar
    points_camera_all = temp_transform(T_cv, T_iv, points)
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
    for box in boxes:

        pose = vis_utils.to_T(box.rot, box.pos)
        T_centroid_camera = temp_transform(T_cv, T_iv, pose)
        centroid_camera = T_centroid_camera[:,3]
        
        draw_point(image, P_cam, centroid_camera, [255, 255, 255], 3,4)
        corners_pixel = draw_box(image, T_cv, T_iv, P_cam, box, [0,0,255], 2, False)
        draw_label(image, corners_pixel, [255, 255, 255], box)
        
    cv2.destroyAllWindows()
    cv2.imshow(image_file, image)
    if write_to_disk:
        cv2.imwrite('./sample_dataset/output/testSeq{}.png'.format(start_idx+idx), image)
    cv2.waitKey(100)
     

if __name__ == '__main__':

    dataset = "scale"

    P_cam, T_iv, T_cv = vis_utils.get_sensor_calibration_alt(dataset)

    start_idx = 10
    end_idx = 20
    
    dataset_folder = "./sample_" + dataset + "/"
    label_file_path = dataset_folder + "labels.json"
    data_file_paths = sorted(glob.glob(dataset_folder+'lidar_data/task_point_cloud*.json'), key=lambda x : int(''.join(filter(str.isdigit, x))))
    camera_file_paths = sorted(glob.glob(dataset_folder+'camera/*.png'), key=lambda x : int(''.join(filter(str.isdigit, x))))

    synced_cameras = sync_camera_lidar(data_file_paths[start_idx:end_idx], camera_file_paths, vis_utils.get_dataset_offset_camera_ts(dataset))

    print("Rendering Perspective Images...")
    label_file = open(label_file_path, 'r')
    label_json = json.load(label_file)  
    for id in tqdm(range(end_idx-start_idx)):
        render_image(label_json, data_file_paths[start_idx:end_idx], synced_cameras, start_idx, id, P_cam, T_iv, T_cv)
