import io
import PIL
import numpy as np
import matplotlib.pyplot as plt

from pyboreas.data.bounding_boxes import BoundingBox2D
from pyboreas.utils.utils import rotToYawPitchRoll, yaw


def convert_plt_to_img(dpi=128):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)


def transform_bounding_boxes(T, C_yaw, raw_labels):
    """
    Generate bounding boxes from labels and transform them
    by a SE3 transformation
    :param T: required SE3 transformation
    :param C_yaw: yaw component of the SE3 transformation
    :param raw_labels: original label data
    """
    boxes = []
    for i in range(len(raw_labels)):
        # Load Labels
        bbox_raw_pos = np.concatenate(
            (np.fromiter(raw_labels[i]['position'].values(), dtype=float), [1]))
        # Create Bounding Box
        pos = np.matmul(T, np.array([bbox_raw_pos]).T)[:3]
        rotation = np.matmul(C_yaw, yaw(raw_labels[i]['yaw']))
        rotToYawPitchRoll(rotation)
        extent = np.array(list(raw_labels[i]['dimensions'].values())).reshape(3, 1)  # Convert to 2d
        box = BoundingBox2D(pos, rotation, extent, raw_labels[i]['label'])
        boxes.append(box)
    return boxes


def vis_camera(cam, figsize=(24.48, 20.48), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.imshow(cam.img[:, :, ::-1])
    ax.set_axis_off()
    plt.show()


def vis_lidar(lid, bounds=[-40, 40, -40, 40, -10, 30], figsize=(10, 10), cmap='winter',
              color='intensity', vmin=None, vmax=None, azim_delta=-75, elev_delta=-5):
    lid.passthrough(bounds)
    p = lid.points
    if color == 'x':
        c = p[:, 0]
    elif color == 'y':
        c = p[:, 1]
    elif color == 'z':
        c = p[:, 2]
    elif color == 'intensity':
        c = p[:, 3]
    elif color == 'ring':
        c = p[:, 4]
    elif color == 'time':
        c = p[:, 5]
    else:
        print('warning: color: {} is not valid'.format(color))
        c = p[:, 2]
    if vmin is None or vmax is None:
        vmin = np.min(c)
        vmax = np.max(c)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.azim += azim_delta
    ax.elev += elev_delta
    xs = p[:, 0]
    ys = p[:, 1]
    zs = p[:, 2]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.scatter(xs=xs, ys=ys, zs=zs, s=0.1, c=c, cmap=cmap,
               vmin=vmin, vmax=vmax, depthshade=False)
    plt.show()


def vis_radar(rad, figsize=(10, 10), dpi=100, cart_resolution=0.2384, cart_pixel_width=640, cmap='gray'):
    cart = rad.get_cartesian(cart_resolution=cart_resolution, cart_pixel_width=cart_pixel_width, in_place=False)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.imshow(cart, cmap=cmap)
    ax.set_axis_off()
    plt.show()
