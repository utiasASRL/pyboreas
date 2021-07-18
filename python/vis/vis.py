# TODO: transform bounding box into sensor frame at time
# TODO: project 3D bounding box onto 2D visualization (radar/lidar top-down, camera front-face)
# TODO: render visualization as an image, (display it), (save it)
# TODO: plot odometry results vs. ground truth

import json
import glob
from collections import OrderedDict
from os import path

import cv2
import open3d as o3d
import open3d.ml.torch as ml3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm

import vis_utils

matplotlib.use("tkagg")  # slu: for testing with ide

class BoreasVisualizer:
    """Main class for loading the Boreas dataset for visualization.

    Loads relevant data, transforms, and labels and provides access to several visualization options.
    Currently only works for one track at a time.
    """

    def __init__(self, dataroot):
        """Initialize the class with the corresponding data, transforms and labels.

        Args:
            dataroot: Path to the directory where the dataset is stored
        """
        # Check if dataroot paths are valid
        if not path.exists(path.join(dataroot, "camera")):
            raise ValueError("Error: images dir missing from dataroot")
        if not path.exists(path.join(dataroot, "lidar_data")):
            raise ValueError("Error: lidar_data dir missing from dataroot")
        if not path.exists(path.join(dataroot, "labels.json")):
            raise ValueError("Error: labels.json missing from dataroot")

        # Instantiate class properties
        self.dataroot = dataroot  # Root directory for the dataset
        self.pcd_paths = sorted(glob.glob(path.join(dataroot, "lidar_data", "task_point_cloud*.json")))[0:3] #TEMP TESTING SLU  # Paths to the pointcloud jsons
        self.img_paths = sorted(glob.glob(path.join(dataroot, "camera", "*.png")))  # Paths to the camera images
        self.label_file = path.join(dataroot, "labels.json")  # Path to the label json
        self.timestamps = []                        # List of all timestamps (in order)
        self.lidar_data = []                        # List of all loaded lidar jsons (in order)
        self.images_raw = []                        # List of all loaded cv2 images (in order, not 1-1 with timestamps)
        self.images_synced = []                     # List of all synced images (in order)
        self.labels = []                            # List of all loaded label jsons (in order)
        self.track_length = len(self.pcd_paths)     # Length of current track

        # Load transforms
        self.P_cam, self.T_iv, self.T_cv = vis_utils.get_sensor_calibration("./calib/P_camera.txt",
                                                                            "./calib/T_applanix_lidar.txt",
                                                                            "./calib/T_camera_lidar.txt",
                                                                            "./calib/T_radar_lidar.txt",
                                                                            verbose=False)

        # Load pointcloud data & timestamps
        print("Loading Lidar Pointclouds...", flush=True)
        for pcd_path in tqdm(self.pcd_paths):
            with open(pcd_path, 'r') as file:
                raw_data = json.load(file)
                self.timestamps.append(raw_data["timestamp"])
                self.lidar_data.append(raw_data)
        # Load camera data
        print("Loading Images...", flush=True)
        for img_path in tqdm(self.img_paths):
            self.images_raw.append(cv2.imread(img_path, cv2.IMREAD_COLOR))
        self._sync_camera_frames()  # Sync each lidar frame to a corresponding camera frame
        # Load label data
        print("Loading Labels...", flush=True)
        with open(self.label_file, 'r') as file:
            raw_labels = json.load(file)
            for label in tqdm(raw_labels):
                self.labels.append(label['cuboids'])

    def visualize_track_topdown(self):
        pc_data = []
        # bb_data = []

        for i in range(self.track_length):
            curr_lidar_data = self.lidar_data[i]
            curr_lables = self.labels[i]

            points, boxes = vis_utils.transform_data_to_sensor_frame(curr_lidar_data, curr_lables)
            points = points.astype(np.float32)

            frame_data = {
                'name': 'lidar_points/frame_{}'.format(i),
                'points': points
            }

            # bbox = ml3d.vis.BoundingBox3D()

            pc_data.append(frame_data)

        # Open3d ML Visualizer
        vis = ml3d.vis.Visualizer()
        vis.visualize(pc_data)
        vis.show_geometries_under("task", True)

    def visualize_track_topdown_mpl(self, frame_idx, predictions=None):
        curr_lidar_data = self.lidar_data[frame_idx]
        curr_lables = self.labels[frame_idx]

        points, boxes = vis_utils.transform_data_to_sensor_frame(curr_lidar_data, curr_lables)
        points = points.astype(np.float32)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(points[:, 0], points[:, 1], s=0.1)

        for box in boxes:
            box.render_bbox_2d(ax)

        plt.show()
        plt.close()

    def get_cam2vel_transform(self, pcd):
        pcd = np.matmul(vis_utils.to_T(vis_utils.rot_z(-np.pi / 2), np.zeros((3, 1))), np.matmul(np.linalg.inv(self.T_cv), pcd))
        pcd = np.matmul(vis_utils.to_T(vis_utils.rot_y(np.pi), np.zeros((3, 1))), pcd)
        return pcd

    def visualize_frame_persp(self, frame_idx):
        raw_labels = self.labels[frame_idx]
        raw_pcd = self.lidar_data[frame_idx]

        # T, C = vis_utils.get_device_pose(raw_pcd)

        points, boxes = vis_utils.transform_data_to_sensor_frame(raw_pcd, raw_labels)
        image = self.images_synced[frame_idx]

        points = points.T

        # points_camera_all = np.matmul(T_cv, np.matmul(np.linalg.inv(T_iv), points))
        points_camera_all = self.get_cam2vel_transform(points)
        points_camera = np.array([])
        for i in range(points_camera_all.shape[1]):
            if points_camera_all[2, i] > 0:
                points_camera = np.concatenate((points_camera, points_camera_all[:, i]))
        points_camera = np.reshape(points_camera, (-1, 4)).T

        pixel_camera = np.matmul(self.P_cam, points_camera)
        max_z = int(max(pixel_camera[2, :]) / 3)
        for i in range(pixel_camera.shape[1]):
            z = pixel_camera[2, i]
            x = int(pixel_camera[0, i] / z)
            y = int(pixel_camera[1, i] / z)
            if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
                c = cv2.applyColorMap(np.array([int(pixel_camera[2, i] / max_z * 255)], dtype=np.uint8), cv2.COLORMAP_RAINBOW).squeeze().tolist()
                cv2.circle(image, (x, y), 1, c, 1)

        centroids_odom = np.array([]).reshape(3, 0)
        for bb in boxes:
            centroids_odom = np.hstack((centroids_odom, bb.pos))
        centroids_odom = np.vstack((centroids_odom, np.ones((1, len(boxes)))))
        centroids_camera_all = self.get_cam2vel_transform(centroids_odom)
        centroids_camera = np.array([])
        for i in range(centroids_camera_all.shape[1]):
            if centroids_camera_all[2, i] > 0:
                centroids_camera = np.concatenate((centroids_camera, centroids_camera_all[:, i]))
        centroids_camera = np.reshape(centroids_camera, (-1, 4)).T
        pixel_centroid_camera = np.matmul(self.P_cam, centroids_camera)
        for i in range(pixel_centroid_camera.shape[1]):
            z = pixel_centroid_camera[2, i]
            x = int(pixel_centroid_camera[0, i] / z)
            y = int(pixel_centroid_camera[1, i] / z)
            if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
                cv2.circle(image, (x, y), 5, [255, 255, 255], 10)

        cv2.destroyAllWindows()
        cv2.imshow("persp_img", image)
        cv2.waitKey(0)

    def _sync_camera_frames(self):
        # Helper function for finding closest timestamp
        def get_closest_ts(query_time, targets):
            min_delta = 1e33  # Temp set to this, should be 1e9
            closest = -1
            for i in range(len(targets)):
                delta = abs(query_time - targets[i])
                if delta < min_delta:
                    min_delta = delta
                    closest = i
            assert (closest >= 0), "closest time to query: {} in rostimes not found.".format(query_time)
            return closest, targets[closest]

        # Find closest lidar timestamp for each camera frame
        camera_timestamps = [int(f.replace('/', '.').split('.')[-2]) for f in self.img_paths]
        for i in range(self.track_length):
            timestamp, corrected_timestamp = vis_utils.get_camera_timestamp(self.lidar_data[i])
            closet_idx, cloest_val = get_closest_ts(corrected_timestamp, camera_timestamps)
            self.images_synced.append(self.images_raw[closet_idx])


if __name__ == '__main__':
    dataset = BoreasVisualizer("./sample_dataset")
    # dataset.visualize_track_topdown()
    dataset.visualize_track_topdown_mpl(0)
    # dataset.visualize_frame_persp(1)
