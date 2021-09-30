import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from copy import deepcopy
import cv2

from vis import map_utils

class BoreasPlotly:
    def __init__(self, visualizer):
        # Data
        self.seq = visualizer.sequence
        self.calib = visualizer.sequence.calib

    def get_pcd(self, idx):
        # load points
        lidar_frame = self.seq.lidar_frames[idx]

        # Calculate transformations for current data
        C_l_enu = lidar_frame.pose[:3, :3].T
        C_a_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_l_enu
        C_a_l = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_frame.load_data()[:,0:3]
        # Draw lidar points
        rand_idx = np.random.choice(lidar_points.shape[0], size=int(lidar_points.shape[0]*0.3), replace=False)
        return lidar_points[rand_idx, :], lidar_frame, C_a_enu, C_a_l

    def get_cam(self, idx):
        camera_frame = self.seq.camera_frames[idx]
        camera_image = camera_frame.load_data()
        return camera_image

    def visualize(self, frame_idx):
        app = dash.Dash(__name__)

        app.layout = html.Div(children=[
            html.H1(children='Graphs'),
            
            html.Div(children=[
            dcc.Graph(id='graph-with-slider')], style={'display': 'inline-block'}),

            html.Div(children=[
            dcc.Graph(id='graph-with-slider2')], style={'display': 'inline-block'}),

            dcc.Slider(
                    id='timestep-slider',
                    min=0,
                    max=len(self.seq.lidar_frames),
                    value=0,
                    marks={str(idx): str(idx) for idx in range(0, len(self.seq.lidar_frames), 5)},
                    step=1
                    ),

            dcc.Interval(id='animator',
                         interval=2000,  # in milliseconds
                         n_intervals=0,
                         disabled=True,
                         ),
            html.Button("Play/Pause", id="play_btn")
        ])

        @app.callback(
            Output('graph-with-slider', 'figure'),
            Output('graph-with-slider2', 'figure'),
            Input('timestep-slider', 'value')
        )
        def update_figure(idx):
            pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx)

            # BEV
            pcd_a = np.matmul(C_a_l[0:2, 0:2].reshape(1, 2, 2), pcd[:, 0:2].reshape(pcd.shape[0], 2, 1)).squeeze(-1)
            fig_bev = deepcopy(go.Figure())
            map_utils.draw_map_plotly("/home/jqian/datasets/boreas-devkit/boreas_mini_v2/boreas_lane.osm", fig_bev, lidar_scan.pose[0, 3], lidar_scan.pose[1, 3], C_a_enu, utm=True)
            fig_bev.add_trace(
                go.Scattergl(x=pcd_a[:,0], y=pcd_a[:,1], mode='markers', visible=True, marker_size=0.5, marker_color='blue')
            )
            fig_bev.update_traces(marker_size=0.5)
            fig_bev.update_layout(
                autosize=False,
                width=800,
                height=800
            )
            fig_bev.update_xaxes(range=[-75, 75])
            fig_bev.update_yaxes(range=[-75, 75])
            fig_bev.update_layout(showlegend=False)

            # Perspective
            fig_persp = deepcopy(go.Figure())
            image = deepcopy(self.get_cam(idx))

            pcd = pcd.T
            pcd = np.vstack((pcd, np.ones(pcd.shape[1])))

            points_camera_all = np.matmul(self.calib.T_camera_lidar, pcd)
            points_camera = np.array([])
            for i in range(points_camera_all.shape[1]):
                if points_camera_all[2, i] > 0:
                    points_camera = np.concatenate((points_camera, points_camera_all[:, i]))
            points_camera = np.reshape(points_camera, (-1, 4)).T
            pixel_camera = np.matmul(self.calib.P0, points_camera)

            max_z = int(max(pixel_camera[2, :]) / 3)
            for i in range(pixel_camera.shape[1]):
                z = pixel_camera[2, i]
                x = int(pixel_camera[0, i] / z)
                y = int(pixel_camera[1, i] / z)
                if x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]:
                    c = cv2.applyColorMap(np.array([int(pixel_camera[2, i] / max_z * 255)], dtype=np.uint8), cv2.COLORMAP_RAINBOW).squeeze().tolist()
                    cv2.circle(image, (x, y), 2, c, 1)

            fig_persp = px.imshow(image)
            fig_persp.update_layout(
                autosize=False,
                height=1000
            )

            return fig_bev, fig_persp
        
        @app.callback(
            Output('timestep-slider', 'value'),
            Input('animator', 'n_intervals'),
            State('timestep-slider', 'value'),
        )
        def on_click(n_intervals, slider_idx):
            if n_intervals is None:
                return 0
            else:
                return (slider_idx + 1) % len(self.seq.lidar_frames)

        @app.callback(
            Output('animator', 'disabled'),
            Input('play_btn', 'n_clicks'),
            State('animator', 'disabled')
        )
        def toggle(n, playing):
            return not playing

        app.run_server(debug=True)