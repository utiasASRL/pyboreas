# TODO: transform bounding box into sensor frame at time
# TODO: project 3D bounding box onto 2D visualization (radar/lidar top-down, camera front-face)
# TODO: render visualization as an image, (display it), (save it)
# TODO: plot odometry results vs. ground truth

# TODO: separate functions for plot one and plot interactive. add persp and BEV plot/video export. fix video export. make boreastransforms class. fix ts_to_load

import sys
import json
import glob
from collections import OrderedDict
from os import path
import csv
import copy
from math import sin, cos, pi
from pathlib import Path

import cv2
import open3d as o3d
import open3d.ml.torch as ml3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import vis_utils
import plot_utils
from lidar_scan import LidarScan
from gps_pose import GPSPose

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
        if not path.exists(path.join(dataroot, "lidar")):
            raise ValueError("Error: lidar dir missing from dataroot")
        if not path.exists(path.join(dataroot, "applanix")):
            raise ValueError("Error: applnix dir missing from dataroot")
        if not path.exists(path.join(dataroot, "calib")):
            raise ValueError("Error: calib dir missing from dataroot")
        # if not path.exists(path.join(dataroot, "labels.json")):
        #     raise ValueError("Error: labels.json missing from dataroot")

        # Instantiate class properties
        self.dataroot = dataroot  # Root directory for the dataset
        self.pcd_paths = sorted(glob.glob(path.join(dataroot, "lidar", "*.bin")))  # Paths to the pointcloud jsons
        self.img_paths = sorted(glob.glob(path.join(dataroot, "camera", "*.png")))  # Paths to the camera images
        self.label_file = path.join(dataroot, "labels.json")  # Path to the label json
        self.timestamps = []                        # List of all timestamps (in order)
        self.lidar_scans = {}                       # Dict of all the lidar poses (by ros timestamp)
        self.gps_poses = {}                         # Dict of all the gps poses (by GPS time)
        self.images_raw = []                        # List of all loaded cv2 images (in order, not 1-1 with timestamps)
        self.images_synced = []                     # List of all synced images (in order)
        self.labels = []                            # List of all loaded label jsons (in order)
        self.track_length = len(self.pcd_paths)     # Length of current track

        # Load transforms
        self.P_cam, self.T_iv, self.T_cv = vis_utils.get_sensor_calibration_alt("boreas",
                                                                            verbose=False)
        self.C_enu_ned = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

        # Load lidar scans and timestamps (currently we use lidar timestamps as the reference)
        print("Loading Lidar Poses...")  # If dataset is complete, each entry in lidar pose should have its corresponding pointcloud file in ./lidar
        with open(path.join(self.dataroot, "applanix", "lidar_poses.csv")) as file:
            reader = csv.reader(file)
            next(reader)  # Extract headers
            missing_pcds = 0
            for row in tqdm(reader, file=sys.stdout):
                timestamp = int(row[0])
                pcd_path = path.join(self.dataroot, "lidar", str(timestamp) + ".bin")
                if not path.exists(pcd_path):  # Only process pointclouds we have data for
                    missing_pcds += 1
                    continue
                self.timestamps.append(timestamp)
                scan = np.fromfile(pcd_path, dtype=np.float32)
                points = scan.reshape((-1, 6))[:, :6]  # x, y, z, i, laser #, gps timestamp
                lidar_scan = LidarScan(timestamp, float(row[1]), np.asarray(row[2:5], dtype=np.float32), np.asarray(row[8:11], dtype=np.float32), points)
                self.lidar_scans[timestamp] = lidar_scan
            print("Failed to get point cloud data for {} scans".format(missing_pcds))
        # Load gps poses
        print("Loading GPS Poses...")
        with open(path.join(self.dataroot, "applanix", "gps_post_process.csv")) as file:
            reader = csv.reader(file)
            headers = next(reader)  # Extract headers
            for row in tqdm(reader, file=sys.stdout):
                gps_pose = GPSPose(float(row[0]), np.asarray(row[1:4], dtype=np.float32), np.asarray(row[7:10], dtype=np.float32))
                self.gps_poses[float(row[0])] = gps_pose
        # Load camera data
        print("Loading Images...")
        for img_path in tqdm(self.img_paths, file=sys.stdout):
            self.images_raw.append(cv2.imread(img_path, cv2.IMREAD_COLOR))
        self._sync_camera_frames()  # Sync each lidar frame to a corresponding camera frame
        # # Load label data
        # print("Loading Labels...", flush=True)
        # with open(self.label_file, 'r') as file:
        #     raw_labels = json.load(file)
        #     for label in tqdm(raw_labels):
        #         self.labels.append(label['cuboids'])

    def visualize_thirdperson(self):
        pc_data = []
        # bb_data = []

        for i in range(self.track_length):
            curr_lidar_data = self.lidar_scans[i].points
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

    def export_video_topdown(self):
        imgs = []
        # Render the matplotlib figs to images
        print("Exporting Topdown View to Video")
        for i in tqdm(range(len(self.timestamps)), file=sys.stdout):
            bplot = self.visualize_bev(frame_idx=i, show=False)
            canvas = FigureCanvas(bplot.fig)
            canvas.draw()
            graph_image = np.array(bplot.fig.canvas.get_renderer()._renderer)
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            imgs.append(graph_image)

        # Write the images to video
        out = cv2.VideoWriter('testing.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (700, 700))
        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()

    def visualize_bev(self, frame_idx, predictions=None, show=True):
        self.curr_ts_idx = frame_idx
        curr_ts = self.timestamps[self.curr_ts_idx]
        curr_lidar_scan = self.lidar_scans[curr_ts]

        boreas_plot = plot_utils.BoreasPlotter(self.timestamps, self.T_iv, self.lidar_scans)
        boreas_plot.update_plot_topdown(curr_lidar_scan)

        if show:
            plt.show()
            plt.draw()
        else:
            plt.close(boreas_plot.fig)

        return boreas_plot

    def get_cam2vel_transform(self, pcd):
        pcd = np.matmul(self.T_cv, pcd)
        return pcd

    def visualize_frame_persp(self, frame_idx):
        for i in tqdm(range(frame_idx, len(self.timestamps))):
            points = self.lidar_scans[i].points[:, 0:3]
            points = points[np.random.choice(len(points), int(0.5*len(points)), replace=False)]
            points = points.T
            points = np.vstack((points, np.ones(points.shape[1])))
            image = copy.deepcopy(self.images_synced[i])

            points_camera_all = self.get_cam2vel_transform(points)
            points_camera = np.array([])
            for i in range(points_camera_all.shape[1]):
                if points_camera_all[2,i] > 0:
                    points_camera = np.concatenate((points_camera, points_camera_all[:,i]))
            points_camera = np.reshape(points_camera, (-1,4)).T
            pixel_camera = np.matmul(self.P_cam, points_camera)

            max_z = int(max(pixel_camera[2,:])/3)
            for i in range(pixel_camera.shape[1]):
                z = pixel_camera[2,i]
                x = int(pixel_camera[0,i] / z)
                y = int(pixel_camera[1,i] / z)
                if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
                    c = cv2.applyColorMap(np.array([int(pixel_camera[2,i] / max_z*255)], dtype=np.uint8), cv2.COLORMAP_RAINBOW).squeeze().tolist()
                    cv2.circle(image,(x,y), 1, c, 1)

            cv2.destroyAllWindows()
            cv2.imshow("Image " + str(i), image)
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
            timestamp = self.timestamps[i]
            corrected_timestamp = timestamp + vis_utils.get_dataset_offset_camera_ts("boreas")
            closet_idx, cloest_val = get_closest_ts(corrected_timestamp, camera_timestamps)
            self.images_synced.append(self.images_raw[closet_idx])


if __name__ == '__main__':
    dataset = BoreasVisualizer("./sample_boreas")
    # dataset.visualize_track_topdown()
    # dataset.visualize_bev(0)
    # dataset.visualize_frame_persp(0)
    dataset.export_video_topdown()
