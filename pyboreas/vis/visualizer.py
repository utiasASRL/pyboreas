import base64
import io
from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from dash.dependencies import Input, Output, State
from matplotlib import cm

from pyboreas.data.sequence import Sequence
from pyboreas.vis import map_utils


class BoreasVisualizer:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calib = sequence.calib
        self.track_length = len(sequence.lidar_frames)
        self.lidar_frames = sequence.lidar_frames
        lstamps = [frame.timestamp for frame in sequence.lidar_frames]
        cstamps = [frame.timestamp for frame in sequence.camera_frames]
        rstamps = [frame.timestamp for frame in sequence.radar_frames]
        # Get corresponding camera and radar frame for each lidar frame
        self.camera_frames = [self._get_closest_frame(lstamp, cstamps, sequence.camera_frames) for lstamp in lstamps]
        self.radar_frames = [self._get_closest_frame(lstamp, rstamps, sequence.radar_frames) for lstamp in lstamps]
        self.render_selection = {'lidar_bev': True, 'radar_bev': True, 'cam_persp': True, '3d_lidar': True}
        self.plot_functions = {'lidar_bev': self.plot_lidar_bev, 'radar_bev': self.plot_radar_bev, 'cam_persp': self.plot_cam_persp, '3d_lidar': self.plot_3d_lidar}
        self.plots_initialized = False

    def _get_closest_frame(self, query_time, target_times, targets):
        times = np.array(target_times)
        closest = np.argmin(np.abs(times - query_time))
        assert (np.abs(query_time - times[closest]) < 1.0), "closest time to query: {} in rostimes not found.".format(query_time)
        return targets[closest]

    def get_pcd(self, idx, down_sample_rate=0.5):
        # load points
        lidar_frame = self.sequence.get_lidar(idx)

        # Calculate transformations for current data
        C_l_enu = lidar_frame.pose[:3, :3].T
        C_a_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_l_enu
        C_a_l = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_frame.load_data()[:, 0:3]
        # Draw lidar points
        rand_idx = np.random.choice(lidar_points.shape[0], size=int(lidar_points.shape[0] * down_sample_rate), replace=False)
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
        radar_ndarray = radar_frame.polar_to_cart(grid_res, grid_size)
        return radar_ndarray

    def plot_lidar_bev(self, idx):
        pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx, 0.5)
        pcd_a = np.matmul(C_a_l[0:2, 0:2].reshape(1, 2, 2), pcd[:, 0:2].reshape(pcd.shape[0], 2, 1)).squeeze(-1)

        fig_bev = go.Figure()
        map_utils.draw_map_plotly(Path(__file__).parent.absolute() / "boreas_lane.osm", fig_bev, lidar_scan.pose[0, 3], lidar_scan.pose[1, 3], C_a_enu, utm=True)
        fig_bev.add_trace(go.Scattergl(x=pcd_a[:, 0], y=pcd_a[:, 1], mode='markers', visible=True, marker_size=0.5, marker_color='blue'))
        fig_bev.update_traces(marker_size=0.5)
        fig_bev.update_layout(
            title="BEV Visualization",
            autosize=False,
            width=700,
            height=700,
            showlegend=False
        )
        fig_bev.update_xaxes(range=[-75, 75])
        fig_bev.update_yaxes(range=[-75, 75])

        return fig_bev

    def plot_radar_bev(self, idx):
        # Radar
        fig_radar = go.Figure()
        grid_size = 700
        grid_res = 0.5
        radar_image = self.get_radar(idx, grid_size, grid_res)

        # Draw image
        radar_pil_image = Image.fromarray(np.uint8(cm.gist_gray(radar_image)[:, :, 0:3] * 255))
        rawBytes = io.BytesIO()
        radar_pil_image.save(rawBytes, "PNG")
        rawBytes.seek(0)
        encoded_radar_string = base64.b64encode(rawBytes.read()).decode()
        encoded_radar_image = "data:image/png;base64," + encoded_radar_string

        fig_radar.add_layout_image(
            dict(
                x=-75,
                sizex=150,
                y=-75,
                sizey=150,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=encoded_radar_image)
        )

        # Configure plot
        fig_radar.update_xaxes(
            range=[-75, 75],
            showgrid=False,
            zeroline=False
        )

        fig_radar.update_yaxes(
            range=[75, -75],
            showgrid=False,
            zeroline=False
        )

        fig_radar.update_layout(
            title="Radar Visualization",
            width=grid_size,
            height=grid_size,
            autosize=False,
            showlegend=False
        )

        return fig_radar

    def plot_cam_persp(self, idx):
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
        scale_factor = 700 / 2448

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig_persp.add_trace(go.Scatter(x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0))

        # Load image. Use 64bit encoding for speed
        with open(self.get_cam_path(idx), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()  # add the prefix that plotly will want when using the string as source
        encoded_image = "data:image/png;base64," + encoded_string

        # Draw image
        fig_persp.add_layout_image(
            dict(x=0,
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

        # Configure axes
        fig_persp.update_xaxes(
            visible=False,
            range=[0, img_width]
        )
        fig_persp.update_yaxes(  # Range reversed so that plotting lidar works properly
            visible=False,
            range=[img_height, 0]
        )

        # Configure figure layout
        fig_persp.update_layout(
            title="Perspective Visualization",
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            autosize=False,
            showlegend=False
        )

        return fig_persp

    def plot_3d_lidar(self, idx):
        # Colored lidar
        pcd, lidar_scan, C_a_enu, C_a_l = self.get_pcd(idx, 0.5)
        pcd = pcd.T
        pcd = np.vstack((pcd, np.ones(pcd.shape[1])))
        points_camera_f = np.matmul(self.calib.T_camera_lidar, pcd)
        points_camera = np.matmul(self.calib.P0, points_camera_f)
        pixel_camera = np.divide(points_camera, points_camera[2, :])
        # Only select valid lidar points
        valid_pixel_idx = (points_camera[2, :] > 0) & (points_camera[2, :] < 100) & \
                          (pixel_camera[1, :] > 0) & (pixel_camera[1, :] < 2048) & \
                          (pixel_camera[0, :] > 0) & (pixel_camera[0, :] < 2448)
        valid_pixel_x = pixel_camera[0][valid_pixel_idx]
        valid_pixel_y = pixel_camera[1][valid_pixel_idx]

        valid_coord_x = points_camera_f[0][valid_pixel_idx]
        valid_coord_y = points_camera_f[1][valid_pixel_idx]
        valid_coord_z = points_camera_f[2][valid_pixel_idx]

        # Layout initialization fix
        if not self.plots_initialized:
            layout = dict()
            self.plots_initialized = True
        else:
            layout = dict(uirevision='latest')
        fig_color_lidar = go.Figure(layout=layout)

        # Get lidar point colors from image
        image = self.get_cam(idx)
        colors = image[valid_pixel_y.astype(int), valid_pixel_x.astype(int)]
        colors_str = [f'rgb({colors[i][0]}, {colors[i][1]}, {colors[i][2]})' for i in range(colors.shape[0])]

        # Plot points
        fig_color_lidar.add_trace(
            go.Scatter3d(
                x=valid_coord_x,
                y=valid_coord_y,
                z=valid_coord_z,
                mode='markers',
                marker_size=2,
                marker_color=colors_str,
            )
        )

        # Configure scene & camera
        scene = dict(aspectratio=dict(x=1, y=1, z=1), xaxis=dict(range=[-100, 100]), yaxis=dict(range=[-100, 100]), zaxis=dict(range=[-100, 100]))
        camera = dict(
            up=dict(x=0, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0., y=0, z=-0.03)
        )

        fig_color_lidar.update_layout(
            title="Colored Lidar Visualization",
            scene=scene,
            scene_camera=camera,
            width=700,
            height=600,
            autosize=False,
        )

        return fig_color_lidar

    def visualize(self, frame_idx):
        app = dash.Dash(__name__)

        app.layout = html.Div(children=[
            html.H1(children='Boreas Visualizer', style={'fontFamily': 'helvetica', 'textAlign': 'center'}),

            html.Div(children=[
                html.H3(children='Render Options (more options = slower)', style={'fontFamily': 'helvetica', 'textAlign': 'left'}),
                dcc.Checklist(
                    id="render_options",
                    options=[
                        {'label': 'Lidar BEV', 'value': 'lidar_bev'},
                        {'label': 'Radar BEV', 'value': 'radar_bev'},
                        {'label': 'Camera Perspective', 'value': 'cam_persp'},
                        {'label': '3D Lidar', 'value': '3d_lidar'},
                    ],
                    value=['lidar_bev', 'radar_bev', 'cam_persp', '3d_lidar'],
                    labelStyle={'display': 'inline-block', 'padding': '0.1rem 0.5rem'},
                    style={'fontFamily': 'helvetica', 'textAlign': 'left'}
                )
            ], style={'padding-left': '10%'}),

            html.Div(children=[
                html.Div(id="lidar_bev_div",
                         children=[dcc.Graph(id='bev_plot')],
                         style={'display': 'inline-block'}),

                html.Div(id="radar_bev_div",
                         children=[dcc.Graph(id='radar_plot')],
                         style={'display': 'inline-block'}),
            ], style={'textAlign': 'center'}),

            html.Br(),

            html.Div(children=[
                html.Div(id="cam_persp_div",
                         children=[dcc.Graph(id='persp_plot')],
                         style={'display': 'inline-block'}),

                html.Div(id="3d_lidar_div",
                         children=[dcc.Graph(id='color_lidar_plot')],
                         style={'display': 'inline-block'}),
            ], style={'textAlign': 'center'}),

            html.Div(children=[
                dcc.Slider(
                    id='timestep_slider',
                    min=0,
                    max=len(self.sequence.lidar_frames),
                    value=max(0, min(frame_idx, len(self.sequence.lidar_frames))),
                    marks={str(idx): str(idx) for idx in range(0, len(self.sequence.lidar_frames), 5)},
                    step=1)
            ], style={'width': '70%', 'padding-left': '17%', 'padding-right': '13%'}),

            dcc.Interval(id='animator',
                         interval=2000,  # in milliseconds
                         n_intervals=0,
                         disabled=True,
                         ),

            html.Div(children=[
                html.Button("Play/Pause", id="play_btn")
            ], style={'textAlign': 'center'}),
        ], style={'width': '60%', 'padding-left': '20%', 'padding-right': '20%'})

        @app.callback(Output('lidar_bev_div', 'style'),
                      Output('radar_bev_div', 'style'),
                      Output('cam_persp_div', 'style'),
                      Output('3d_lidar_div', 'style'),
                      Input('render_options', 'value'))
        def render_selection(values):
            def toggle_render(name, toggle_on):
                display_on = {'display': 'inline-block'}
                display_off = {'display': 'none'}
                self.render_selection[name] = toggle_on
                if toggle_on:
                    return display_on
                else:
                    return display_off

            res = []
            for plot_name in self.render_selection.keys():
                res.append(toggle_render(plot_name, plot_name in values))

            return tuple(res)

        @app.callback(
            Output('bev_plot', 'figure'),
            Output('radar_plot', 'figure'),
            Output('persp_plot', 'figure'),
            Output('color_lidar_plot', 'figure'),
            Input('timestep_slider', 'value')
        )
        def plot_graphs(idx):
            res = []
            for plot_name in self.render_selection.keys():
                if self.render_selection[plot_name]:
                    res.append(self.plot_functions[plot_name](idx))
                else:
                    res.append(go.Figure())
            return tuple(res)

        @app.callback(
            Output('timestep_slider', 'value'),
            Input('animator', 'n_intervals'),
            State('timestep_slider', 'value'),
        )
        def on_click(n_intervals, slider_idx):
            if n_intervals is None:
                return 0
            else:
                return (slider_idx + 1) % len(self.sequence.lidar_frames)

        @app.callback(
            Output('animator', 'disabled'),
            Input('play_btn', 'n_clicks'),
            State('animator', 'disabled')
        )
        def toggle(n, playing):
            if n:
                return not playing
            return playing

        app.run_server(debug=True)


if __name__ == "__main__":
    seq = Sequence("/home/shichen/datasets/", ["boreas_mini_v2", 1616518050000000, 1616518060000000])
    # seq = Sequence("/home/jqian/datasets/boreas-devkit/",["boreas_mini_v2", 1616518050000000, 1616518060000000])
    visualizer = BoreasVisualizer(seq)
    visualizer.visualize(0)
