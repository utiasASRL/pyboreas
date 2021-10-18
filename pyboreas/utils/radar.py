################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

import numpy as np
import cv2
from pyboreas.utils.utils import get_time_from_filename

upgrade_time = 1632182400  # before: resolution = 0.0596, after: resolution = 0.04381


def load_radar(example_path):
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
    """
    # Hard coded configuration to simplify parsing code
    encoder_size = 5600
    resolution = 0.0596
    t = get_time_from_filename(example_path)
    if t > upgrade_time:
        resolution = 0.04381
    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
    min_range = int(round(2.5 / resolution))
    fft_data[:, :min_range] = 0
    fft_data = np.squeeze(fft_data)
    return timestamps, azimuths, valid, fft_data, resolution


def radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                             interpolate_crossover=True, fix_wobble=True):
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    EPS = 1e-14
    if fix_wobble and radar_resolution == 0.0596:
        azimuths = azimuths.reshape((1, 1, 400))  # 1 x 1 x 400
        sample_angle = np.expand_dims(sample_angle, axis=-1)  # H x W x 1
        diff = np.abs(azimuths - sample_angle)
        c3 = np.argmin(diff, axis=2)
        azimuths = azimuths.squeeze()
        c3 = c3.reshape(cart_pixel_width, cart_pixel_width)  # azimuth indices (closest)
        mindiff = sample_angle.squeeze() - azimuths[c3]
        sample_angle = sample_angle.squeeze()
        mindiff = mindiff.squeeze()

        subc3 = c3 * (c3 < 399)
        aplus = azimuths[subc3 + 1]
        a1 = azimuths[subc3]
        delta1 = mindiff * (mindiff > 0) * (c3 < 399) / (aplus - a1 + EPS)
        subc3 = c3 * (c3 > 0)
        a2 = azimuths[subc3]
        aminus = azimuths[1 + (c3 > 0) * (subc3 - 2)]
        delta2 = mindiff * (mindiff < 0) * (c3 > 0) / (a2 - aminus + EPS)
        sample_v = c3 + delta1 + delta2
        sample_v = sample_v.astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)


def mean_intensity_mask(polar_data, multiplier=3.0):
    """Thresholds on multiplier*np.mean(azimuth_data) to create a polar mask of likely target points.
    Args:
        polar_data (np.ndarray): num_azimuths x num_range_bins polar data
        multiplier (float): multiple of mean that we treshold on
    Returns:
        np.ndarray: binary polar mask corresponding to likely target points
    """
    num_azimuths, range_bins = polar_data.shape
    mask = np.zeros((num_azimuths, range_bins))
    for i in range(num_azimuths):
        m = np.mean(polar_data[i, :])
        mask[i, :] = polar_data[i, :] > multiplier * m
    return mask
