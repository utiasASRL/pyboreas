import argparse
import utm
import numpy as np
import folium

mapbox_token = 'pk.eyJ1Ijoia2VlbmJ1cm4yMDA0IiwiYSI6ImNraHh1bm13dTA1cXEycG4wbTdvZ2xlY3YifQ.dtUB8qJaN09IjcfcmnOc1Q'

# coords is a list of (lat, lng) tuples
def plot_polyline_folium(coords, graph_map=None, tiles="openstreetmap", zoom=1, fit_bounds=True, color='#FF0000',
                      linewidth=4, opacity=1, popup=None, attr=None, **kwargs):
    """
    Plot a list of (lat, lng) tuples on an interactive folium web map.
    ----------
    coords : List of (lat, lng) tuples
    graph_map : folium.folium.Map or folium.FeatureGroup
        if not None, plot the graph on this preexisting folium map object
    popup : string
        edge attribute to display in a pop-up when an edge is clicked
    tiles : string
        name of a folium tileset
    zoom : int
        initial zoom level for the map
    fit_bounds : bool
        if True, fit the map to the boundaries of the route's edges
    color : string
        color of the lines
    linewidth : numeric
        width of the lines
    opacity : numeric
        opacity of the lines
    kwargs : dict
        Extra keyword arguments passed through to folium
    Returns
    -------
    graph_map : folium.folium.Map
    """
    lat_sum = 0
    lng_sum = 0
    lats = []
    lngs = []
    for lat, lng in coords:
        lat_sum += lat
        lng_sum += lng
        lats.append(lat)
        lngs.append(lng)
    centroid = (lat_sum / len(coords), lng_sum / len(coords))

    if graph_map is None:
        graph_map = folium.Map(location=centroid, zoom_start=zoom, tiles=tiles, attr=attr)

    pl = folium.PolyLine(locations=coords, popup=popup, color=color, weight=linewidth, opacity=opacity, **kwargs)

    pl.add_to(graph_map)

    if fit_bounds and isinstance(graph_map, folium.Map):
        bounds = [(min(lats), min(lngs)), (max(lats), max(lngs))]
        graph_map.fit_bounds(bounds)

    return graph_map

def get_folder_from_file_path(path, backup=0):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1 - backup):
        newpath += elems[j] + "/"
    return newpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='gps_raw.csv', type=str,
                        help='raw or post-procesed GPS data csv file UTM coordinates. col1 = Easting, col2 = Northing')
    parser.add_argument('--use_satellite', default=False, type=bool, help='Choose whether to use satellite images or \
                        openstreetmap tiles.')
    args = parser.parse_args()

    new_point_threshold = 1.0

    root = get_folder_from_file_path(args.input, backup=1)

    coords = []
    with open(args.input, 'r') as f:
        f.readline()
        lines = f.readlines()
        x_prev = 0
        y_prev = 0
        for i in range(0, len(lines), 100):
            elems = lines[i].split(',')
            x = float(elems[1])
            y = float(elems[2])
            if np.sqrt((x - x_prev)**2 + (y - y_prev)**2) > new_point_threshold:
                lat, lng = utm.to_latlon(x, y, 17, 'N')
                coords.append((lat, lng))
                x_prev = x
                y_prev = y

    if args.use_satellite:
        tiles = 'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token=' + str(mapbox_token)
        map = plot_polyline_folium(coords, tiles=tiles, attr='Mapbox')
    else:
        map = plot_polyline_folium(coords)

    map.save(root + 'route.html')
