import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

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
        C_i_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_l_enu
        C_iv = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_frame.load_data()
        # Draw lidar points
        pcd_i = np.matmul(C_iv[0:2, 0:2].reshape(1, 2, 2), lidar_points[:, 0:2].reshape(lidar_points.shape[0], 2, 1)).squeeze(-1)
        rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0]*0.3), replace=False)
        return pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], lidar_frame, C_i_enu

    def visualize(self, frame_idx):
        app = dash.Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(id='graph-with-slider'),
            dcc.Slider(
                    id='timestep-slider',
                    min=0,
                    max=len(self.seq.lidar_frames),
                    value=0,
                    marks={str(idx): str(idx) for idx in range(0, len(self.seq.lidar_frames), 5)},
                    step=1
                    ),
            dcc.Interval(id='animator',
                         interval=500,  # in milliseconds
                         n_intervals=0,
                         disabled=True,
                         ),
            html.Button("Play/Pause", id="play_btn")
        ])

        @app.callback(
            Output('graph-with-slider', 'figure'),
            Input('timestep-slider', 'value')
        )
        def update_figure(idx):
            pcd_x, pcd_y, lidar_scan, C_i_enu = self.get_pcd(idx)
            fig = go.Figure()
            map_utils.draw_map_plotly("/home/shichen/datasets/boreas_mini/boreas_lane.osm", fig, lidar_scan.pose[0, 3], lidar_scan.pose[1, 3], C_i_enu, utm=True)
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

        @app.callback(
            Output('timestep-slider', 'value'),
            Input('animator', 'n_intervals'),
            State('timestep-slider', 'value'),
        )
        def on_click(n_intervals, slider_idx):
            if n_intervals is None:
                return 0
            else:
                return slider_idx + 1

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