import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

from vis import map_utils

class BoreasPlotly:
    def __init__(self, sequence):
        # Data
        self.seq = sequence
        self.calib = sequence.calib
        self.lidar_scans = sequence.lidar_dict
        self.camera_imgs = sequence.camera_dict
        assert not (self.lidar_scans is None and self.camera_imgs is None)  # We must have something to plot

    def get_pcd(self, idx):
        # load points
        curr_ts = self.seq.timestamps[idx]
        lidar_scan = self.seq.lidar_dict[curr_ts]

        # Calculate transformations for current data
        C_v_enu = lidar_scan.get_C_v_enu().as_matrix()
        C_i_enu = self.calib.T_applanix_lidar[0:3, 0:3] @ C_v_enu
        C_iv = self.calib.T_applanix_lidar[0:3, 0:3]
        lidar_points = lidar_scan.load_points()
        # Draw lidar points
        pcd_i = np.matmul(C_iv[0:2, 0:2].reshape(1, 2, 2), lidar_points[:, 0:2].reshape(lidar_points.shape[0], 2, 1)).squeeze(-1)
        rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0] * 0.5), replace=False)
        return pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], lidar_scan, C_i_enu

    def visualize(self, frame_idx):
        app = dash.Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(id='graph-with-slider'),
            dcc.Slider(
                id='timestep-slider',
                min=0,
                max=self.seq.seq_len,
                value=0,
                marks={str(idx): str(idx) for idx in range(0, self.seq.seq_len, 5)},
                step=1
            )
        ])

        @app.callback(
            Output('graph-with-slider', 'figure'),
            Input('timestep-slider', 'value'))
        def update_figure(idx):
            pcd_x, pcd_y, lidar_scan, C_i_enu = self.get_pcd(idx)
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