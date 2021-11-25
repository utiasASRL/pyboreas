import io
import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pyboreas.utils.utils import rotToYawPitchRoll, yaw, get_transform2


def convert_plt_to_img(dpi=128):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)


def vis_camera(cam, figsize=(24.48, 20.48), dpi=100, show=True, save=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.imshow(cam.img)
    ax.set_axis_off()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    return ax


def vis_lidar(lid, figsize=(10, 10), cmap='winter',
              color='intensity', colorvec=None, vmin=None, vmax=None, azim_delta=-75, elev_delta=-5,
              show=True, save=None):
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
    if colorvec is not None:
        c = colorvec
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
    ax.set_axis_off()
    if colorvec is None:
        ax.scatter(xs=xs, ys=ys, zs=zs, s=0.1, c=c, cmap=cmap,
                   vmin=vmin, vmax=vmax, depthshade=False)
    else:
        ax.scatter(xs=xs, ys=ys, zs=zs, s=0.1, c=c, depthshade=False)
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    return ax


def vis_radar(rad, figsize=(10, 10), dpi=100, cart_resolution=0.2384, cart_pixel_width=640, cmap='gray',
              show=True, save=None):
    cart = rad.polar_to_cart(cart_resolution=cart_resolution, cart_pixel_width=cart_pixel_width, in_place=False)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.imshow(cart, cmap=cmap)
    ax.set_axis_off()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    return ax


def bilinear_interp(img, X, Y):

    x = np.array(X).squeeze()
    y = np.array(Y).squeeze()

    x1 = np.floor(x).astype(np.int32)
    x2 = np.ceil(x).astype(np.int32)
    y1 = np.floor(y).astype(np.int32)
    y2 = np.ceil(y).astype(np.int32)

    mask = np.where(x1 == x2)

    q11 = img[y1, x1]  # N x 3
    q12 = img[y2, x1]
    q21 = img[y1, x2]
    q22 = img[y2, x2]

    EPS = 1e-14
    x_21 = (x2 - x1 + EPS)
    x_2 = ((x2 - x) / x_21).reshape(-1, 1)
    x_1 = ((x - x1) / x_21).reshape(-1, 1)

    f_y1 = q11 * x_2 + q21 * x_1
    f_y2 = q12 * x_2 + q22 * x_1

    f_y1[mask] = q11[mask]
    f_y2[mask] = q22[mask]

    mask = np.where(y1 == y2)

    y_21 = (y2 - y1 + EPS)
    y_2 = ((y2 - y) / y_21).reshape(-1, 1)
    y_1 = ((y - y1) / y_21).reshape(-1, 1)

    f = y_2 * f_y1 + y_1 * f_y2
    f[mask] = f_y1[mask]

    return f.squeeze()


def draw_point(img, pixels, color=[0, 0, 255], diameter=3, line_width=4):
    cv2.circle(img,(pixels[0],pixels[1]), diameter, color, line_width)


def draw_box(img, uv, color, line_width, draw_corners=False, draw_box=True):
    assert(uv.shape[0] == 8 and uv.shape[1] >= 2)
    box = []
    for i in range(uv.shape[0]):
        box.append(tuple(uv[i, :].astype(np.int32)))

    corner_map = {'ftr':0, 'ftl':1, 'btl':2, 'btr':3,
              'fbr':4, 'fbl':5, 'bbl':6, 'bbr':7}
    if draw_corners:
        draw_point(img, box[corner_map['ftl']], [0, 0, 255], 3, 4) # [b,g,r]
        draw_point(img, box[corner_map['ftr']], [0, 255, 0], 3, 4)
        draw_point(img, box[corner_map['fbl']], [255, 0, 0], 3, 4)
        draw_point(img, box[corner_map['fbr']], [255, 255, 0], 3, 4)
        draw_point(img, box[corner_map['btl']], [0, 0, 128], 3, 4)
        draw_point(img, box[corner_map['btr']], [0, 128, 0], 3, 4)
        draw_point(img, box[corner_map['bbl']], [128, 0, 0], 3, 4)
        draw_point(img, box[corner_map['bbr']], [255, 255, 0], 3, 4)

    if draw_box:
        cv2.line(img, box[corner_map['ftl']], box[corner_map['ftr']], color, line_width)
        cv2.line(img, box[corner_map['ftl']], box[corner_map['fbl']], color, line_width)
        cv2.line(img, box[corner_map['ftr']], box[corner_map['fbr']], color, line_width)
        cv2.line(img, box[corner_map['fbl']], box[corner_map['fbr']], color, line_width)

        cv2.line(img, box[corner_map['ftl']], box[corner_map['fbr']], color, line_width)
        cv2.line(img, box[corner_map['ftr']], box[corner_map['fbl']], color, line_width)

        cv2.line(img, box[corner_map['btl']], box[corner_map['btr']], color, line_width)
        cv2.line(img, box[corner_map['btl']], box[corner_map['bbl']], color, line_width)
        cv2.line(img, box[corner_map['btr']], box[corner_map['bbr']], color, line_width)
        cv2.line(img, box[corner_map['bbl']], box[corner_map['bbr']], color, line_width)

        cv2.line(img, box[corner_map['ftl']], box[corner_map['btl']], color, line_width)
        cv2.line(img, box[corner_map['ftr']], box[corner_map['btr']], color, line_width)
        cv2.line(img, box[corner_map['fbl']], box[corner_map['bbl']], color, line_width)
        cv2.line(img, box[corner_map['fbr']], box[corner_map['bbr']], color, line_width)


def draw_boxes(img, UV, color=[0,0,255], line_width=2, draw_corners=False):
    # UV: list of 8x2 np arrays corresponding to BB corners projected onto image coordinates
    for uv in UV:
        if uv is None:
            continue
        draw_box(img, uv, color, line_width, draw_corners)