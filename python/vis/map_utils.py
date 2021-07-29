import math
import xml.etree.ElementTree as xml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj

import csv

import sys

matplotlib.use("tkagg")

def get_value_list(d):
    assert isinstance(d, dict)
    if sys.version_info[0] == 2:
        item_list = d.values()
    elif sys.version_info[0] == 3:
        item_list = list(d.values())
    else:
        # should not happen
        raise RuntimeError("Only python 2 and 3 supported.")
    assert isinstance(item_list, list)
    return item_list


def get_item_iterator(d):
    assert isinstance(d, dict)
    if sys.version_info[0] == 2:
        item_iter = d.iteritems()
        assert hasattr(item_iter, "next")
    elif sys.version_info[0] == 3:
        item_iter = iter(d.items())
        assert hasattr(item_iter, "__next__")
    else:
        # should not happen
        raise RuntimeError("Only python 2 and 3 supported.")
    assert hasattr(item_iter, "__iter__")
    return item_iter

def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None

def plot_color_samplemap(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "name":
            if tag.get("v") == "Allen Road":
                return 1
            elif tag.get("v").startswith("LIne") or tag.get("v").startswith("Line"):
                return -1
            else:
                return 0
    return -1

def plot_color_boreas_lane(element):
    return 1

def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, x_origin, y_origin, rot_mtx=None, utm=False):  # Convention: x is lat, y is long if using lat/long
        if utm == False:
            self.lat_origin = x_origin
            self.lon_origin = y_origin
            self.zone = math.floor((y_origin+180.)/6)+1  # works for most tiles, and for all in the dataset
            self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
            [self.x_origin, self.y_origin] = [0,0]#temp testing# self.p(y_origin, x_origin)
        elif utm == True:
            self.zone = 17  # Hardcode for now
            self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
            self.x_origin = x_origin
            self.y_origin = y_origin
        if rot_mtx is not None:
            self.rot_mtx = rot_mtx[0:2, 0:2]  # Just take 2d component (x, y) of rotation matrix
        else:
            self.rot_mtx = None

    def latlon2xy(self, lat, lon):
        if self.rot_mtx is not None:
            [x, y] = self.p(lon, lat)
            rotated = np.matmul(self.rot_mtx, np.array([[x-self.x_origin], [y-self.y_origin]]))
            return rotated[0][0], rotated[1][0]
        else:
            [x, y] = self.p(lon, lat)
            return [x - self.x_origin, y - self.y_origin]


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes, tight=False):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    if tight:
        axes.set_xlim([min_x, max_x])
        axes.set_ylim([min_y, max_y])
    else:
        axes.set_xlim([min_x - 10, max_x + 10])
        axes.set_ylim([min_y - 10, max_y + 10])

def draw_map(map_path, ax):
    # # Load and draw the lanelet2 map (requires lanelet2 (source catkin_ws in interaction))
    lat_origin = 43.78396145783  # origin is necessary to correctly project the lat lon values in the osm file to the local
    lon_origin = -79.47012023102  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
    # projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
    # laneletmap = lanelet2.io.load(map_path, projector)
    # map_vis_lanelet2.draw_lanelet_map(laneletmap, ax)
    draw_map_without_lanelet(map_path, ax, lat_origin, lon_origin)

def draw_map_without_lanelet(filename, axes, x_origin, y_origin, rot_mtx=None, utm=False):
    axes.patch.set_facecolor('white')

    projector = LL2XYProjector(x_origin, y_origin, rot_mtx, utm)

    map_data = xml.parse(filename).getroot()

    point_dict = {}
    for node in map_data.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    unknown_linestring_types = list()

    for way in map_data.findall('way'):
        color = plot_color_boreas_lane(way)
        if color == 1:
            type_dict = dict(color="black", linewidth=1, zorder=10)
            x_list, y_list = get_x_y_lists(way, point_dict)
            plt.plot(x_list, y_list, **type_dict)
        elif color == 0:
            type_dict = dict(color="gray", linewidth=1, zorder=10)
            x_list, y_list = get_x_y_lists(way, point_dict)
            plt.plot(x_list, y_list, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

if __name__ == "__main__":
    fig, ax = plt.subplots()
    draw_map("./sample_dataset/sample_map.osm", ax)
    # u_x = 623104
    # u_y = 4849388
    x = 624186.687436
    y = 4844692.17809
    x2 = 623947.146108
    y2 = 4844908.90913
    x3 = 624100.772252  # +18s
    y3 = 4844789.28905
    x4 = 624141.093729  # -18s
    y4 = 4844746.7083
    # diff_x = -4066.990882
    # diff_y = 621.629028708
    # ax.scatter(x, y, s=10)
    # ax.scatter(x - diff_y, y - diff_x, s=10)
    # ax.scatter(u_x, u_y, s=10)
    with open("./sample_dataset/gps_raw2.csv") as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if row[0] == "time": continue
            count += 1
            if count % 50 == 0:
                ax.scatter(float(row[1]), float(row[2]), s=10, color='r')

    # o_x = 624121.315803 + 4067.59276782
    # o_y = 4844768.47104 - 622.176477479

    o_x = x4 + 4067.59276782 - 175
    o_y = y4 - 622.176477479 + 175

    with open("./sample_dataset/gps.csv") as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if row[0] == "time": continue
            count += 1
            if count % 50 == 0:
                ax.scatter(float(row[1]) + o_x, float(row[2]) + o_y, s=10, color='g')

    plt.show()
