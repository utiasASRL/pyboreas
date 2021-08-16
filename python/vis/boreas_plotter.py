from threading import Lock

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
import numpy as np

import map_utils


class BoreasPlotter:
    """
    Class for plotting persp and BEV for boreas stuff
    """
    def __init__(self, timestamps, curr_ts_idx, transforms, lidar_scans=None, camera_data=None, camera_poses=None, labels=None):
        # Transform
        self.T = transforms
        # Data
        self.timestamps = timestamps
        self.lidar_scans = lidar_scans
        self.camera_data = camera_data
        assert not (self.lidar_scans is None and self.camera_data is None)  # We must have something to plot
        # For plot stuff
        self.curr_ts_idx = curr_ts_idx
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.plot_update_mutex = Lock()

        button_ax = plt.axes([0.05, 0.05, 0.05, 0.05])
        self.button_f = Button(button_ax, "<")  # Store buttons ref so they dont go out of scope and become useless
        self.button_f.on_clicked(self.on_click_bkwd)

        button_ax2 = plt.axes([0.90, 0.05, 0.05, 0.05])
        self.button_f2 = Button(button_ax2, ">")
        self.button_f2.on_clicked(self.on_click_fwd)

    def on_click_fwd(self, event):
        if not self.plot_update_mutex.acquire(timeout=0.5): return

        try:
            self.curr_ts_idx = min(self.curr_ts_idx + 1, len(self.timestamps) - 1)
            print("Visualizing Timestep Index: {}/{}...".format(self.curr_ts_idx, len(self.timestamps)), end=" ")

            self.ax.clear()
            curr_ts = self.timestamps[self.curr_ts_idx]
            curr_lidar_scan = self.lidar_scans[curr_ts]

            self.update_plot_topdown(curr_lidar_scan)

            print("Done")
        finally:
            self.plot_update_mutex.release()

    def on_click_bkwd(self, event):
        if not self.plot_update_mutex.acquire(timeout=0.5): return

        try:
            self.curr_ts_idx = max(self.curr_ts_idx - 1, 0)
            print("Visualizing Timestep Index: {}/{}...".format(self.curr_ts_idx, len(self.timestamps)), end=" ")

            self.ax.clear()
            curr_ts = self.timestamps[self.curr_ts_idx]
            curr_lidar_scan = self.lidar_scans[curr_ts]

            self.update_plot_topdown(curr_lidar_scan)

            print("Done")
        finally:
            self.plot_update_mutex.release()

    def update_plot_topdown(self, lidar_scan, downsample_factor=0.3):
        # Calculate transformations for current data
        C_v_enu = lidar_scan.get_C_v_enu().as_matrix()
        C_i_enu = self.T.T_iv[0:3, 0:3] @ C_v_enu
        C_iv = self.T.T_iv[0:3, 0:3]

        # Draw map
        map_utils.draw_map_without_lanelet("./sample_boreas/boreas_lane.osm", self.ax, lidar_scan.position[0], lidar_scan.position[1], C_i_enu, utm=True)

        # Calculate point colors
        z_min = -3
        z_max = 5
        colors = cm.jet(((lidar_scan.points[:, 2] - z_min) / (z_max - z_min)) + 0.2, 1)[:, 0:3]

        # Draw lidar points
        pcd_i = np.matmul(C_iv[0:2, 0:2].reshape(1, 2, 2), lidar_scan.points[:, 0:2].reshape(lidar_scan.points.shape[0], 2, 1)).squeeze(-1)
        rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0]*downsample_factor), replace=False)
        self.scatter = self.ax.scatter(pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], color=colors[rand_idx, :], s=0.05)

        # Draw predictions (TODO)
        # for box in boxes:
        #     box.render_bbox_2d(ax)
        #
        # if predictions is not None:
        #     for box in predictions:
        #         box.render_bbox_2d(ax, color="k")

        # Set to scale labeling bounds
        self.ax.set_xlim(-75, 75)
        self.ax.set_ylim(-75, 75)

        plt.draw()