import math
import xml.etree.ElementTree as xml
import numpy as np
import pyproj


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class MapframeProjector:
    """
    Class to convert incoming utm or lat/long points to be in a certain frame of reference (x/y origin & rotation).
    **If using lat/long, convention is x is long, y is lat.**
    """
    def __init__(self, x_origin, y_origin, rot_mtx=None, utm=False):
        if not utm:
            self.zone = math.floor((x_origin+180.)/6)+1  # convert to utm zone. doesnt work for all zones
            self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
            [self.x_origin, self.y_origin] = self.p(x_origin, y_origin)
        elif utm:
            self.zone = 17  # Hardcode for now
            self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
            self.x_origin = x_origin
            self.y_origin = y_origin

        if rot_mtx is not None:
            self.rot_mtx = rot_mtx[0:2, 0:2]  # Just take 2d component (x, y) of rotation matrix
        else:
            self.rot_mtx = None

    def proj_latlon(self, lat, lon):
        if self.rot_mtx is not None:
            [x, y] = self.p(lon, lat)
            rotated = np.matmul(self.rot_mtx, np.array([[x-self.x_origin], [y-self.y_origin]]))
            return rotated[0][0], rotated[1][0]
        else:
            [x, y] = self.p(lon, lat)
            return [x - self.x_origin, y - self.y_origin]


def get_way_points(element, point_dict):
    # Get all points in a way in the osm file
    x_pts = []
    y_pts = []
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_pts.append(point.x)
        y_pts.append(point.y)
    return x_pts, y_pts


def draw_map(filename, axes, x_origin, y_origin, rot_mtx=None, utm=False):
    """
    Draws a map from an osm file on the given axis, wrt a given frame.

    Args:
        filename: path to the osm map file to draw
        axes: axes to draw the map on
        x_origin: x coords of origin frame
        y_origin: y coords of origin frame
        rot_mtx: rotation matrix of the map from wrt to an ENU frame
        utm: flag for whether the origin is expressed in UTM or lat/long. If using lat/long, x is long, y is lat
    """
    map_data = xml.parse(filename).getroot()
    projector = MapframeProjector(x_origin, y_origin, rot_mtx, utm)

    point_dict = {}
    for node in map_data.findall("node"):
        point = Point()
        point.x, point.y = projector.proj_latlon(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    axes.patch.set_facecolor('white')
    for way in map_data.findall('way'):  # RN: just plot everything as black
        type_dict = dict(color="black", linewidth=1, zorder=10)
        x_list, y_list = get_way_points(way, point_dict)
        axes.plot(x_list, y_list, **type_dict)
