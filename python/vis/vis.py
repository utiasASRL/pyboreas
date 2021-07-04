# TODO: transform bounding box into sensor frame at time
# TODO: project 3D bounding box onto 2D visualization (radar/lidar top-down, camera front-face)
# TODO: render visualization as an image, (display it), (save it)
# TODO: plot odometry results vs. ground truth

import json
import glob

import open3d as o3d
import open3d.ml.torch as ml3d
from matplotlib import cm
import numpy as np

import vis_utils
from tqdm import tqdm


def visualize_img_topdown(pcd_files, label_file, vis_camera='view_option.json'):
    """
    Visualizes point clouds and bounding boxes
     :param pcd_files: point cloud json file from 1 frame
     :param label_file: all Scale AI annotations from a project
     :param file_name: path to save image file
     :param vis_camera: open3d PinholeCameraParameters json file
    """
    # Load data
    pc_data = []
    # bb_data = []

    for i, pcd_file in enumerate(tqdm(pcd_files)):
        with open(pcd_file, 'r') as file:
            raw_data = json.load(file)
        with open(label_file, 'r') as file:
            raw_labels = json.load(file)[i]['cuboids']

        points, boxes = vis_utils.transform_data_to_sensor_frame(raw_data, raw_labels)
        points = points.astype(np.float32)

        frame_data = {
            'name': '{}'.format(pcd_file),
            'points': points
        }

        # bbox = ml3d.vis.BoundingBox3D()

        pc_data.append(frame_data)

    # Open3d ML Visualizer
    vis = ml3d.vis.Visualizer()
    vis.visualize(pc_data)
    vis.show_geometries_under("task", True)

if __name__ == '__main__':
    data_files = sorted(glob.glob('./sample_dataset/data/task_point_cloud*.json'))
    visualize_img_topdown(data_files, 'sample_dataset/labels.json')
