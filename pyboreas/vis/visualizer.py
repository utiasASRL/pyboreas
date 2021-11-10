import base64
import io
from pathlib import Path
import time
import copy

import dash
from dash import dcc
from dash import html
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from dash.dependencies import Input, Output, State
from matplotlib import cm

from pyboreas.vis import map_utils
from pyboreas.utils.utils import get_closest_frame, get_inverse_tf
from pyboreas.vis.vis_utils import bilinear_interp

# Disable Info Logging for Dash
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class BoreasVisualizer:
    def __init__(self, sequence):
        self.seq = copy.deepcopy(sequence)
        self.calib = sequence.calib
        self.track_length = len(sequence.lidar_frames)

        self.seq.synchronize_frames('lidar')
        self.lidar_frames = self.seq.lidar_frames
        self.radar_frames = self.seq.radar_frames
        self.camera_frames = self.seq.camera_frames

        self.render_selection = {'lidar_bev': True, 'radar_bev': True, 'cam_persp': True, '3d_lidar': True}
        self.plot_functions = {'lidar_bev': self.plot_lidar_bev, 'radar_bev': self.plot_radar_bev, 'cam_persp': self.plot_cam_persp, '3d_lidar': self.plot_3d_lidar}
        self.plots_initialized = False

        self.map_data, self.map_point_ids, self.map_points = map_utils.load_map(Path(__file__).parent.absolute() / "boreas_lane.osm")


    def plot_lidar_bev(self, idx):
        lid = self.seq.get_lidar(idx)
        lid.passthrough([-75, 75, -75, 75, -5, 10])
        lid.random_downsample(0.5)
        lid.transform(self.calib.T_applanix_lidar)
        T_a_enu = self.calib.T_applanix_lidar @ get_inverse_tf(lid.pose)
        C_a_enu = T_a_enu[:3, :3]
        points = lid.points
        lid.unload_data()

        fig_bev = go.Figure()
        points_a = map_utils.transform_points(self.map_points, C_a_enu, lid.pose[0, 3], lid.pose[1, 3])  # Transform loaded map points into current frame before plotting
        map_utils.draw_map_plotly(self.map_data, self.map_point_ids, points_a, fig_bev, cutoff_radius=100)
        fig_bev.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode='markers', visible=True, marker_size=0.5, marker_color='blue'))
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
        rad = self.seq.get_radar(idx)
        radar_image = rad.polar_to_cart(0.5, 712)
        rad.unload_data()
        mwidth = 640 * 0.2384

        # Draw image
        radar_pil_image = Image.fromarray((radar_image).astype(np.uint8))
        rawBytes = io.BytesIO()
        radar_pil_image.save(rawBytes, "PNG")
        rawBytes.seek(0)
        encoded_radar_string = base64.b64encode(rawBytes.read()).decode()
        encoded_radar_image = "data:image/png;base64," + encoded_radar_string

        fig_radar.add_layout_image(
            dict(
                x=-mwidth/2,
                sizex=mwidth,
                y=-mwidth/2,
                sizey=mwidth,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=encoded_radar_image),
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
            width=712,
            height=712,
            autosize=False,
            showlegend=False
        )

        return fig_radar

    def get_T_camera_lidar(self, idx):
        T_enu_camera = self.camera_frames[idx].pose
        T_enu_lidar = self.lidar_frames[idx].pose
        return np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)

    def plot_cam_persp(self, idx):
        # Project lidar points into pixel frame

        lid = self.seq.get_lidar(idx)
        lid.remove_motion(lid.body_rate)
        T_camera_lidar = self.get_T_camera_lidar(idx)
        lid.transform(T_camera_lidar)
        lid.passthrough([-75, 75, -20, 10, 2, 40])  # xmin, xmax, ymin, ymax, zmin, zmax
        uv, colors, _ = lid.project_onto_image(self.calib.P0)
        lid.unload_data()

        # Image in figure background taken from https://plotly.com/python/images/#zoom-on-static-images
        fig_persp = go.Figure()
        img_width = 2448
        img_height = 2048
        scale_factor = 700 / 2448

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig_persp.add_trace(go.Scatter(x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0))

        # Load image. Use 64bit encoding for speed
        with open(self.camera_frames[idx].path, "rb") as image_file:
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
            go.Scattergl(x=uv[:, 0],
                         y=uv[:, 1],
                         mode='markers',
                         visible=True,
                         marker_size=1,
                         marker_color=colors,
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

        lid = self.seq.get_lidar(idx)
        lid.remove_motion(lid.body_rate)
        T_camera_lidar = self.get_T_camera_lidar(idx)
        lid.transform(T_camera_lidar)
        lid.passthrough([-50, 50, -10, 5, 2, 80])  # xmin, xmax, ymin, ymax, zmin, zmax
        uv, _, mask = lid.project_onto_image(self.calib.P0)
        points = lid.points[mask][:, :3]
        lid.unload_data()

        # Layout initialization fix
        if not self.plots_initialized:
            layout = dict()
            self.plots_initialized = True
        else:
            layout = dict(uirevision='latest')
        fig_color_lidar = go.Figure(layout=layout)

        # Get lidar point colors from image
        camera_frame = self.seq.get_camera(idx)
        image = camera_frame.img
        camera_frame.unload_data()

        # colors = image[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        colors = bilinear_interp(image, uv[:, 0], uv[:, 1])
        colors_str = [f'rgb({colors[i][0]}, {colors[i][1]}, {colors[i][2]})' for i in range(colors.shape[0])]

        # Plot points
        fig_color_lidar.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker_size=1,
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
                    value=[name for name in self.render_selection.keys() if self.render_selection[name]],
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
                    max=len(self.lidar_frames),
                    value=max(0, min(frame_idx, len(self.lidar_frames))),
                    marks={str(idx): str(idx) for idx in range(0, len(self.lidar_frames), 5)},
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
                return (slider_idx + 1) % len(self.lidar_frames)

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

