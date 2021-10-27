import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from copy import deepcopy
import base64
from PIL import Image
from matplotlib import cm
import io
import time

from vis import map_utils

class BoreasPlotly:
    def __init__(self, visualizer):
        # Data
        self.seq = visualizer.sequence
        self.calib = visualizer.sequence.calib
        self.camera_frames = visualizer.camera_frames
        self.radar_frames = visualizer.radar_frames

    def get_pcd(self, idx, down_sample_rate=0.5):
        # load points
        lidar_frame = self.seq.get_lidar(idx)

        # Calculate transformations for current data
        C_l_enu = lidar_frame.pose[:3, :3].T
        C_a_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_l_enu
        C_a_l = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_frame.load_data()[:,0:3]
        # Draw lidar points
        rand_idx = np.random.choice(lidar_points.shape[0], size=int(lidar_points.shape[0]*down_sample_rate), replace=False)
        return lidar_points[rand_idx, :], lidar_frame, C_a_enu, C_a_l

    def get_cam(self, idx):
        camera_frame = self.camera_frames[idx]
        camera_image = camera_frame.load_data()
        return camera_image

    def get_cam_path(self, idx):
        cam = self.camera_frames[idx]
        return cam.path

    def get_radar(self, idx, grid_size, grid_res):
        radar_frame = self.radar_frames[idx]
        radar_frame.load_data()
        radar_ndarray = radar_frame.get_cartesian(grid_res, grid_size)
        return radar_ndarray

    def visualize(self, frame_idx):
        app = dash.Dash(__name__)

        app.layout = html.Div(children=[
            html.H1(children='Graphs'),
            
            html.Div(children=[
            dcc.Graph(id='bev_plot')], style={'display': 'inline-block'}),

            html.Div(children=[
            dcc.Graph(id='persp_plot')], style={'display': 'inline-block'}),

            html.Div(children=[
            dcc.Graph(id='radar_plot')], style={'display': 'inline-block'}),

            html.Div(children=[
            dcc.Graph(id='color_lidar_plot')], style={'display': 'inline-block'}),

            dcc.Slider(
                    id='timestep_slider',
                    min=0,
                    max=len(self.seq.lidar_frames),
                    value=max(0, min(frame_idx, len(self.seq.lidar_frames))),
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
            Output('bev_plot', 'figure'),
            Input('timestep_slider', 'value')
        )
        def update_figure(idx):
            pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx, 0.5)

            # BEV
            pcd_a = np.matmul(C_a_l[0:2, 0:2].reshape(1, 2, 2), pcd[:, 0:2].reshape(pcd.shape[0], 2, 1)).squeeze(-1)
            fig_bev = go.Figure()
            map_utils.draw_map_plotly("/home/shichen/datasets/boreas_mini/boreas_lane.osm", fig_bev, lidar_scan.pose[0, 3], lidar_scan.pose[1, 3], C_a_enu, utm=True)
            fig_bev.add_trace(
                go.Scattergl(x=pcd_a[:,0], y=pcd_a[:,1], mode='markers', visible=True, marker_size=0.5, marker_color='blue')
            )
            fig_bev.update_traces(marker_size=0.5)
            fig_bev.update_layout(
                autosize=False,
                width=600,
                height=600,
                showlegend=False
            )
            fig_bev.update_xaxes(range=[-75, 75])
            fig_bev.update_yaxes(range=[-75, 75])

            return fig_bev


        @app.callback(
            Output('persp_plot', 'figure'),
            Input('timestep_slider', 'value')
        )
        def update_persp(idx):
            # Perspective
            # Project lidar points into pixel frame
            pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx, 0.5)
            pcd = pcd.T
            pcd = np.vstack((pcd, np.ones(pcd.shape[1])))
            points_camera = np.matmul(self.calib.T_camera_lidar, pcd)
            points_camera = np.matmul(self.calib.P0, points_camera)
            pixel_camera = np.divide(points_camera, points_camera[2, :])
            # Only select valid lidar points
            valid_pixel_idx = (points_camera[2, :] > 0) & (pixel_camera[1, :] > 0) & (pixel_camera[1, :] < 2048) & (pixel_camera[0, :] > 0) & (pixel_camera[0, :] < 2448)
            valid_pixel_x = pixel_camera[0][valid_pixel_idx]
            valid_pixel_y = pixel_camera[1][valid_pixel_idx]
            valid_pixel_z = points_camera[2][valid_pixel_idx]

            # Image in figure background taken from https://plotly.com/python/images/#zoom-on-static-images
            fig_persp = go.Figure()
            img_width = 2448
            img_height = 2048
            scale_factor = 0.3

            # Add invisible scatter trace.
            # This trace is added to help the autoresize logic work.
            fig_persp.add_trace(
                go.Scatter(
                    x=[0, img_width],
                    y=[0, img_height],
                    mode="markers",
                    marker_opacity=0
                )
            )

            # Configure axes
            fig_persp.update_xaxes(
                visible=False,
                range=[0, img_width]
            )
            fig_persp.update_yaxes(  # Range reversed so that plotting lidar works properly
                visible=False,
                range=[img_height, 0]
            )

            # Load image. Use 64bit encoding for speed
            with open(self.get_cam_path(idx), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                # add the prefix that plotly will want when using the string as source
            encoded_image = "data:image/png;base64," + encoded_string

            # Draw image
            fig_persp.add_layout_image(
                dict(
                    x=0,
                    sizex=img_width,
                    y=0,
                    sizey=img_height,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=encoded_image)
            )

            # Draw lidar
            fig_persp.add_trace(
                    go.Scattergl(x=valid_pixel_x,
                                 y=valid_pixel_y,
                                 mode='markers',
                                 visible=True,
                                 marker_size=1,
                                 marker_color=valid_pixel_z,
                                 marker_colorscale='rainbow')
                )

            # Configure other layout
            fig_persp.update_layout(
                width=img_width * scale_factor,
                height=img_height * scale_factor,
                autosize=False,
                showlegend=False
            )

            return fig_persp

        @app.callback(
            Output('radar_plot', 'figure'),
            Input('timestep_slider', 'value')
        )
        def update_radar(idx):
            # Radar
            fig_radar = go.Figure()
            grid_size = 600
            grid_res = 0.5
            radar_image = self.get_radar(idx, grid_size, grid_res)

            # Draw image
            radar_pil_image = Image.fromarray(np.uint8(cm.gist_gray(radar_image)[:,:,0:3]*255))
            rawBytes = io.BytesIO()
            radar_pil_image.save(rawBytes, "PNG")
            rawBytes.seek(0)
            encoded_radar_string = base64.b64encode(rawBytes.read()).decode()
            encoded_radar_image = "data:image/png;base64," + encoded_radar_string
            
            fig_radar.update_xaxes(
                visible=False,
                range=[0, grid_size]
            )

            fig_radar.update_yaxes(
                visible=False,
                range=[grid_size, 0]
            )

            fig_radar.add_layout_image(
                dict(
                    x=0,
                    sizex=grid_size,
                    y=0,
                    sizey=grid_size,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=encoded_radar_image)
            )
            fig_radar.update_layout(
                width=grid_size,
                height=grid_size,
                autosize=False,
                showlegend=False
            )

            return fig_radar

        @app.callback(
            Output('color_lidar_plot', 'figure'),
            Input('timestep_slider', 'value')
        )
        def update_color_lidar(idx):
            # Colored lidar
            pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx, 0.5)
            pcd = pcd.T
            pcd = np.vstack((pcd, np.ones(pcd.shape[1])))
            points_camera_f = np.matmul(self.calib.T_camera_lidar, pcd)
            points_camera = np.matmul(self.calib.P0, points_camera_f)
            pixel_camera = np.divide(points_camera, points_camera[2, :])
            # Only select valid lidar points
            valid_pixel_idx = (points_camera[2, :] > 0) & (pixel_camera[1, :] > 0) & (pixel_camera[1, :] < 2048) & (pixel_camera[0, :] > 0) & (pixel_camera[0, :] < 2448)
            valid_pixel_x = pixel_camera[0][valid_pixel_idx]
            valid_pixel_y = pixel_camera[1][valid_pixel_idx]
            valid_pixel_z = points_camera[2][valid_pixel_idx]

            fig_color_lidar = go.Figure()

            image = self.get_cam(idx)

            colors = image[valid_pixel_y.astype(int), valid_pixel_x.astype(int)]
            colors_str = [f'rgb({colors[i][0]}, {colors[i][1]}, {colors[i][2]})' for i in range(colors.shape[0])]

            fig_color_lidar.add_trace(
                go.Scatter3d(
                    x=valid_pixel_x,
                    y=valid_pixel_y,
                    z=valid_pixel_z,
                    mode='markers',
                    marker_size=2,
                    marker_color=colors_str,
                )
            )

            camera = dict(
                up=dict(x=0, y=-3, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.1, y=-0.1, z=-2)
            )
            fig_color_lidar.update_layout(
                scene_camera=camera,
                width=1200,
                height=600,
                autosize=False,
            )

            return fig_color_lidar
        
        @app.callback(
            Output('timestep_slider', 'value'),
            Input('animator', 'n_intervals'),
            State('timestep_slider', 'value'),
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
            if n:
                return not playing
            return playing

        app.run_server(debug=False)