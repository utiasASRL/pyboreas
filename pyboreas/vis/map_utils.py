import math
import xml.etree.ElementTree as xml
import numpy as np
import pyproj
import plotly.graph_objects as go


def load_map(filename):
    map_data = xml.parse(filename).getroot()
    map_point_ids = []
    map_points = []

    # zone = math.floor((x_origin + 180.) / 6) + 1  # convert longitude to utm zone. doesnt work for all zones
    zone = 17  # Hardcode for now
    projector = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone, datum='WGS84')

    for node in map_data.findall("node"):
        point = projector(float(node.get('lon')), float(node.get('lat')))
        map_point_ids.append(int(node.get('id')))
        map_points.append(list(point))

    return map_data, map_point_ids, map_points


def transform_points(points, rot_mtx, x_origin, y_origin):
    # Transform loaded points into a given frame
    pts = np.array(points).T
    pts[0, :] -= x_origin
    pts[1, :] -= y_origin
    rotated = np.matmul(rot_mtx[0:2, 0:2], pts)
    return rotated.T.tolist()


def get_way_points_bounded(element, point_dict, radius=None):
    # Get all points in a way in the osm file within a certain radius
    x_pts = []
    y_pts = []
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        if radius is not None:
            if point[0] ** 2 + point[1] **2 > radius ** 2:
                continue
        x_pts.append(point[0])
        y_pts.append(point[1])
    return x_pts, y_pts


def draw_map_plotly(map_data, map_point_ids, map_points, fig, cutoff_radius=None):
    """
    Draws a map from map data/pts on the given axis. Uses plotly instead of mpl.
    """
    # Create dictionary of point id to point data for faster access
    point_dict = dict(zip(map_point_ids, map_points))

    all_x, all_y = [], []
    for way in map_data.findall('way'):  # RN: just plot everything as black
        x_list, y_list = get_way_points_bounded(way, point_dict, cutoff_radius)
        if len(x_list) == 0 or len(y_list) == 0: continue
        all_x += x_list
        all_x.append(float('nan'))
        all_y += y_list
        all_y.append(float('nan'))
    fig.add_trace(go.Scattergl(x=all_x, y=all_y, mode="lines", marker_color="black", line_width=0.5))
