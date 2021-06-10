import argparse

import open3d as o3d
import numpy as np
from matplotlib import cm
from shutil import copyfile
import pandas as pd
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import datetime
import utils
import os

pd.options.mode.chained_assignment = None

mesh = o3d.io.read_triangle_mesh('Toyota Prius XW30 2010.obj', True) \
    .scale(0.05, center=(0, 0, 0))
mesh.compute_vertex_normals()
T = np.eye(4)
T[:3, :3] = mesh.get_rotation_matrix_from_xyz((np.pi / 2, np.pi / 4 + np.pi, 0))
T[0, 3] = 0
T[1, 3] = 0
T[2, 3] = -2
mesh.transform(T)


def get_lidar_projection(path):
    scan = np.fromfile(path, dtype=np.float32)
    points = scan.reshape((-1, 6))[:, :6]
    intensity = points[:, 3]
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    zmin = -10
    zmax = 30
    colors = cm.winter((intensity - zmin) / (zmax - zmin))[:, 0:3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json("render_option.json")

    vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        o3d.io.read_pinhole_camera_parameters('view_option.json'))

    output_path = save + 'lidar_out/lidar{:02}-{:04}.png'.format(t, k)
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


def get_radar_im(path, radar_resolution=0.0596, cart_resolution=0.25, cart_pixel_width=1080,
                 interpolate_crossover=True):
    # Load and convert radar data
    _, azimuths, _, fft_data = utils.load_radar(path)
    out = utils.radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                         interpolate_crossover)
    out = np.squeeze(out, axis=0)
    out = np.expand_dims(out, axis=2)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out = cv2.arrowedLine(out, (540, 580), (540, 500), (1, 0, 0), 12)
    return out


def min_transform(prev_pose, df):
    x = prev_pose['x']
    y = prev_pose['y']
    heading = prev_pose['heading']
    df['dist'] = np.sqrt((x - df['x']) ** 2 + (y - df['y']) ** 2)
    df2 = df[df['dist'] < 3]

    df2['rotation_err'] = np.nan
    for idx, rows in df2.iterrows():
        n = utils.rotationError(np.matmul(np.transpose(utils.get_rotation(heading)), utils.get_rotation(df.loc[idx, 'heading'])))
        df2.loc[idx, 'rotation_err'] = n

    return df2['rotation_err'].idxmin()


if __name__ == '__main__':
    k = 0
    t = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/workspace/nas/ASRL/2021-Boreas/',
                        type=str, help='location of root folder containing the directories of each sequence')
    parser.add_argument('--repo', default='/workspace/Documents/VideoSequence/',
                        type=str, help='where to save image folders and video file?')
    args = parser.parse_args()
    root = args.root
    save = args.save

    working_folders = ['boreas-2020-11-26-13-58', 'boreas-2020-12-01-13-26', 'boreas-2021-01-15-12-17',
                       'boreas-2021-01-26-10-59', 'boreas-2021-01-26-11-59', 'boreas-2021-02-02-14-07',
                       'boreas-2021-02-09-12-55']

    folders = sorted(os.listdir(root))

    idx = 0
    df = pd.read_csv(root + folders[0] + '/applanix/camera_poses.csv')
    prev_pose = df.iloc[idx, :]
    font = ImageFont.truetype("DejaVuSans.ttf", 72)

    for folder in folders:
        if folder not in working_folders:
            continue

        df = pd.read_csv(root + folder + '/' + '/applanix/camera_poses.csv')
        idx = min_transform(prev_pose, df)

        cp = root + folder + '/camera'
        lp = root + folder + '/lidar'
        rp = root + folder + '/radar'

        camera_paths = sorted(os.listdir(cp))
        lidar_paths = sorted(os.listdir(lp))
        radar_paths = sorted(os.listdir(rp))

        for c in camera_paths[idx:idx + 600]:
            k += 1
            min_lidar_idx = np.argmin(np.abs(int(c[:-4]) - np.array([int(x[:-4]) for x in lidar_paths])))
            i = lp + '/' + lidar_paths[min_lidar_idx]
            get_lidar_projection(i)
            min_radar_idx = np.argmin(np.abs(int(c[:-4]) - np.array([int(x[:-4]) for x in radar_paths if "png" in x])))
            i = rp + '/' + radar_paths[min_radar_idx]
            radar = get_radar_im(i)
            copyfile(cp + '/' + c, save + 'camera_images/camera{:02}-{:04}.png'.format(t, k))

            timestamp = datetime.datetime.fromtimestamp(int(float(c[:-4]) / 1000000000))
            camera = Image.open(save + 'camera_images/camera{:02}-{:04}.png'.format(t, k))
            radar *= 255
            radar = radar.astype(np.uint8)
            radar = Image.fromarray(radar)
            lidar = Image.open(save + 'lidar_out/lidar{:02}-{:04}.png'.format(t, k))
            canvas = Image.new('RGB', (3840, 2160), color=(26, 26, 26))
            camera.thumbnail([1930, 1615])
            canvas.paste(camera, (0, 0))
            canvas.paste(radar, (2345, 0))
            canvas.paste(lidar, (1930, 1080))
            draw = ImageDraw.Draw(canvas)
            draw.text((500, 1500), str(timestamp), (0, 0, 0), font=font)
            canvas.save(save + 'final/im{:02}-{:04}.png'.format(t, k))
            print(t, k)
        idx += 600
        prev_pose = df.iloc[idx, :]
        k = 0
        t += 1
