# TODO: plot odometry results vs. ground truth

import sys
import glob
from os import path
import csv

import cv2
import open3d.ml.torch as ml3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from tqdm import tqdm

from vis import vis_utils
from vis import boreas_plotter
from vis import boreas_plotly
from data_classes.sequence import Sequence

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import map_utils
import plotly.graph_objects as go

matplotlib.use("tkagg")  # slu: for testing with ide


class BoreasVisualizer:
    """Main class for loading the Boreas dataset for visualization.

    Loads relevant data, transforms, and labels and provides access to several visualization options.
    Currently only works for one track at a time.
    """

    def __init__(self, sequence):
        """Initialize the class with the corresponding data, transforms and labels.

        Args:
            sequence: Sequence object to base the visualization on
        """
        self.sequence = sequence
        self.track_length = len(sequence)
        self.timestamps = sequence.timestamps
        # # Load label data
        # print("Loading Labels...", flush=True)
        # with open(self.label_file, 'r') as file:
        #     raw_labels = json.load(file)
        #     for label in tqdm(raw_labels):
        #         self.labels.append(label['cuboids'])

    def visualize_thirdperson(self):
        # Currently not working
        pc_data = []
        # bb_data = []

        for i in range(self.track_length):
            curr_lidar_data = self.sequence.lidar_scans[i].points
            curr_lables = self.sequence.labels[i]

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

    def visualize(self, frame_idx, predictions=None, mode='both', show=True):
        """
        Visualize the sequence.

        Args:
            frame_idx: the frame index in the current sequence to visualize
            predictions: user generated predictions to also be visualized
            mode: 'persp' for perspective (camera) view, 'bev' for top down view, or 'both'
            show: flag to show the matplotlib plot. Should only be off for exporting to video

        Returns: the BoreasPlotter object used for visualizing

        """
        # boreas_plot = boreas_plotter.BoreasPlotter(self.sequence,
        #                                            frame_idx,
        #                                            mode=mode)
        # boreas_plot.update_plots(frame_idx)
        # if show:
        #     plt.show()
        # else:
        #     plt.close(boreas_plot.fig)
        #
        # return boreas_plot
        boreas_plotly.visualize(self.sequence, frame_idx)

    def export_vis_video(self, name, mode='both'):
        """
        Exports the current sequence visualization to video. Defaults to current working directory for save location.

        Args:
            name: name for the exported video file (don't include extension)
            mode: 'persp' for perspective (camera) view, 'bev' for top down view, or 'both'
        """
        imgs = []
        # Render the matplotlib figs to images
        print("Exporting Visualization to Video")
        for i in tqdm(range(len(self.timestamps)), file=sys.stdout):
            bplot = self.visualize(frame_idx=i, mode=mode, show=False)
            canvas = FigureCanvas(bplot.fig)
            canvas.draw()
            graph_image = np.array(bplot.fig.canvas.get_renderer()._renderer)
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            imgs.append(graph_image)

        # Write the images to video
        # MJPG encoder is part of cv2
        out = cv2.VideoWriter(name + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, (graph_image.shape[1], graph_image.shape[0]))
        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()


if __name__ == '__main__':
    seq = Sequence("/home/shichen/datasets/", ["boreas_mini", 1606417230036848128, 1606417239992609024])

    calib = seq.calib

    def get_pcd(idx):
        # load points
        curr_ts = seq.timestamps[idx]
        lidar_scan = seq.lidar_dict[curr_ts]

        # Calculate transformations for current data
        C_v_enu = lidar_scan.get_C_v_enu().as_matrix()
        C_i_enu = calib.T_applanix_lidar[0:3, 0:3] @ C_v_enu
        C_iv = calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_scan.load_points()
        # Draw lidar points
        pcd_i = np.matmul(C_iv[0:2, 0:2].reshape(1, 2, 2), lidar_points[:, 0:2].reshape(lidar_points.shape[0], 2, 1)).squeeze(-1)
        rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0] * 0.5), replace=False)
        return pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], lidar_scan, C_i_enu

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            id='timestep-slider',
            min=0,
            max=seq.seq_len,
            value=0,
            marks={str(idx): str(idx) for idx in range(0, seq.seq_len, 5)},
            step=1
        )
    ])

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('timestep-slider', 'value'))
    def update_figure(idx):
        pcd_x, pcd_y, lidar_scan, C_i_enu = get_pcd(idx)
        fig = go.Figure()
        # fig = px.scatter(x=pcd_x, y=pcd_y)
        map_utils.draw_map_plotly("/home/shichen/datasets/boreas_mini/boreas_lane.osm", fig, lidar_scan.position[0], lidar_scan.position[1], C_i_enu, utm=True)
        fig.add_trace(
                    go.Scattergl(x=pcd_x, y=pcd_y, mode='markers', visible=True, marker_size=0.5, marker_color='blue')
                )
        fig.update_traces(marker_size=0.5)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000
        )
        fig.update_xaxes(range=[-75, 75])
        fig.update_yaxes(range=[-75, 75])

        fig.update_layout(showlegend=False)

        return fig

    app.run_server(debug=True)


    # dataset = BoreasVisualizer(sequence)
    # # for name in ["both", "persp", "bev"]:
    # #     dataset.export_vis_video(name, name)
    # dataset.visualize(0)
