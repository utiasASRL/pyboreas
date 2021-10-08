from threading import Lock
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
import numpy as np
import cv2

from pyboreas.vis import map_utils


class BoreasPlotter:
    """
    Class for plotting persp and BEV for boreas stuff
    """
    def __init__(self, visualizer, curr_ts_idx, labels=None, mode='both'):
        # Data
        self.calib = visualizer.calib
        self.lidar_frames = visualizer.lidar_frames
        self.camera_frames = visualizer.camera_frames
        self.radar_frames = visualizer.radar_frames
        
        assert not (self.lidar_frames is None and self.camera_frames is None)  # We must have something to plot

        # For plotting
        self.mode = mode
        self.curr_ts_idx = curr_ts_idx
        self.plot_update_mutex = Lock()

        # Create plot based on mode
        if mode == 'both':
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
            self.bev_ax = self.ax[0]
            self.bev_ax.set_aspect('equal', adjustable='box')
            self.persp_ax = self.ax[1]
        elif mode == 'persp':
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
            self.bev_ax = None
            self.persp_ax = self.ax
        elif mode == 'bev':
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
            self.bev_ax = self.ax
            self.bev_ax.set_aspect('equal', adjustable='box')
            self.persp_ax = None

        # Make buttons for forward/reverse
        button_ax = plt.axes([0.05, 0.05, 0.05, 0.05])
        self.button_f = Button(button_ax, "<")  # Store buttons ref so they dont go out of scope and become useless
        self.button_f.on_clicked(self.on_click_bkwd)

        button_ax2 = plt.axes([0.90, 0.05, 0.05, 0.05])
        self.button_f2 = Button(button_ax2, ">")
        self.button_f2.on_clicked(self.on_click_fwd)

    def on_click_fwd(self, event):
        if not self.plot_update_mutex.acquire(timeout=0.5):
            return
        try:
            self.curr_ts_idx = min(self.curr_ts_idx + 1, len(self.lidar_frames) - 1)
            print("Visualizing Timestep Index: {}/{}...".format(self.curr_ts_idx, len(self.lidar_frames)), end=" ")
            self.update_plots(self.curr_ts_idx)
            print("Done")
        finally:
            self.plot_update_mutex.release()

    def on_click_bkwd(self, event):
        if not self.plot_update_mutex.acquire(timeout=0.5):
            return
        try:
            self.curr_ts_idx = max(self.curr_ts_idx - 1, 0)
            print("Visualizing Timestep Index: {}/{}...".format(self.curr_ts_idx, len(self.lidar_frames)), end=" ")
            self.update_plots(self.curr_ts_idx)
            print("Done")
        finally:
            self.plot_update_mutex.release()

    def update_plots(self, frame_idx):
        if self.mode == 'bev' or self.mode == 'both':
            lidar_frame = self.lidar_frames[frame_idx]
            self.bev_ax.clear()
            self.update_plot_bev(self.bev_ax, lidar_frame)

        if self.mode == 'persp' or self.mode == 'both':
            camera_frame = self.camera_frames[frame_idx]
            if self.lidar_frames is not None:
                lidar_frame = self.lidar_frames[frame_idx]
            else:
                lidar_frame = None
            self.persp_ax.clear()
            self.update_plot_persp(self.persp_ax, camera_frame, lidar_frame)

    def update_plot_bev(self, ax, lidar_frame, downsample_factor=0.3):
        # Calculate transformations for current data
        C_l_enu = lidar_frame.pose[:3, :3].T
        C_a_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_l_enu
        C_av = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_frame.load_data()

        # Draw map CURRENTLY HARDCODED
        map_utils.draw_map("./python/vis/boreas_lane.osm", ax, lidar_frame.pose[0, 3], lidar_frame.pose[1, 3], C_a_enu, utm=True)

        # Calculate point colors
        z_min = -3
        z_max = 5
        colors = cm.jet(((lidar_points[:, 2] - z_min) / (z_max - z_min)) + 0.2, 1)[:, 0:3]

        # Draw lidar points
        pcd_i = np.matmul(C_av[0:2, 0:2].reshape(1, 2, 2), lidar_points[:, 0:2].reshape(lidar_points.shape[0], 2, 1)).squeeze(-1)
        rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0]*downsample_factor), replace=False)
        self.scatter = ax.scatter(pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], color=colors[rand_idx, :], s=0.05)

        # Draw predictions (TODO)
        # for box in boxes:
        #     box.render_bbox_2d(ax)
        #
        # if predictions is not None:
        #     for box in predictions:
        #         box.render_bbox_2d(ax, color="k")

        # Set to scale labeling bounds
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        plt.draw()

    def update_plot_persp(self, ax, camera_data, lidar_frame, downsample_factor=0.3):
        if lidar_frame is None:  # If there is no lidar points, just show image
            ax.imshow(camera_data.load_data())
        else:
            points = lidar_frame.load_data()[:, 0:3]
            points = points[np.random.choice(points.shape[0], size=int(points.shape[0]*downsample_factor), replace=False)]
            points = points.T
            points = np.vstack((points, np.ones(points.shape[1])))

            points_camera_all = np.matmul(self.calib.T_camera_lidar, points)
            points_camera = np.array([])
            for i in range(points_camera_all.shape[1]):
                if points_camera_all[2, i] > 0:
                    points_camera = np.concatenate((points_camera, points_camera_all[:, i]))
            points_camera = np.reshape(points_camera, (-1, 4)).T
            pixel_camera = np.matmul(self.calib.P0, points_camera)

            image = copy.deepcopy(camera_data.load_data())

            max_z = int(max(pixel_camera[2, :]) / 3)
            for i in range(pixel_camera.shape[1]):
                z = pixel_camera[2, i]
                x = int(pixel_camera[0, i] / z)
                y = int(pixel_camera[1, i] / z)
                if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
                    c = cv2.applyColorMap(np.array([int(pixel_camera[2, i] / max_z * 255)], dtype=np.uint8), cv2.COLORMAP_RAINBOW).squeeze().tolist()
                    cv2.circle(image, (x, y), 1, c, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            plt.draw()