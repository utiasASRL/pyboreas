import plotly.graph_objects as go
import numpy as np
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.express as px

import pandas as pd

def visualize(seq, frame_idx):
    f = go.Figure()

    calib = seq.calib

    # # Draw map CURRENTLY HARDCODED
    # map_utils.draw_map("./python/vis/boreas_lane.osm", ax, lidar_scan.position[0], lidar_scan.position[1], C_i_enu, utm=True)

    # # Calculate point colors
    # z_min = -3
    # z_max = 5
    # colors = cm.jet(((lidar_points[:, 2] - z_min) / (z_max - z_min)) + 0.2, 1)[:, 0:3]

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
        return pcd_i[rand_idx, 0], pcd_i[rand_idx, 1]

    # for step in np.arange(0, seq.seq_len, 1):
    #     # load points
    #     curr_ts = seq.timestamps[step]
    #     lidar_scan = seq.lidar_dict[curr_ts]
    #
    #     # Calculate transformations for current data
    #     C_v_enu = lidar_scan.get_C_v_enu().as_matrix()
    #     C_i_enu = calib.T_applanix_lidar[0:3, 0:3] @ C_v_enu
    #     C_iv = calib.T_applanix_lidar[0:3, 0:3]
    #     lidar_points = lidar_scan.load_points()
    #     # Draw lidar points
    #     pcd_i = np.matmul(C_iv[0:2, 0:2].reshape(1, 2, 2), lidar_points[:, 0:2].reshape(lidar_points.shape[0], 2, 1)).squeeze(-1)
    #     rand_idx = np.random.choice(pcd_i.shape[0], size=int(pcd_i.shape[0] * 0.5), replace=False)
    #     # self.scatter = ax.scatter(pcd_i[rand_idx, 0], pcd_i[rand_idx, 1], color=colors[rand_idx, :], s=0.05)
    #
    #     # # Set to scale labeling bounds
    #     # ax.set_xlim(-75, 75)
    #     # ax.set_ylim(-75, 75)
    #     #
    #     # plt.draw()
    #
    #     f.add_trace(
    #         go.Scattergl(x=pcd_i[rand_idx, 0], y=pcd_i[rand_idx, 1], mode='markers', visible=False, marker_size=0.5)
    #     )
    #
    # f.data[10].visible = True
    #
    # steps = []
    # for i in range(len(f.data)):
    #     step = dict(
    #         method="update",
    #         args=[{"visible": [False] * len(f.data)},
    #               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    #     )
    #     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    #     steps.append(step)
    #
    # sliders = [dict(
    #     active=10,
    #     currentvalue={"prefix": "Frame: "},
    #     pad={"t": 50},
    #     steps=steps
    # )]
    #
    # f.update_layout(
    #     autosize=False,
    #     width=1000,
    #     height=1000,
    #     sliders=sliders
    # )
    # f.update_xaxes(range=[-75, 75])
# f.update_yaxes(range=[-75, 75])
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            id='timestep-slider',
            min=0,
            max=seq.seq_len,
            value=0,
            # marks={str(year): str(year) for year in df['year'].unique()},
            step=1
        )
    ])

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('timestep-slider', 'value'))
    def update_figure(idx):
        pcd_x, pcd_y = get_pcd(idx)

        fig = px.scatter(x=pcd_x, y=pcd_y,
                         marker_size=0.5)

        fig.update_layout(transition_duration=500)

        return fig

    app.run_server(debug=True)
    print("hello")



# if __name__ == '__main__':
#     app.run_server(debug=True)