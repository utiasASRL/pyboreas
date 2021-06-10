"""
    Data augmentation functions.
    Authors: Keenan Burnett, David Yoon
"""
import numpy as np
import cv2
import torch
from utils.utils import get_transform, get_inverse_tf
from datasets.radar import radar_polar_to_cartesian
from datasets.oxford import mean_intensity_mask

def augmentBatch(batch, config):
    """Rotates the cartesian radar image by a random amount, adjusts the ground truth transform accordingly."""
    rot_max = config['augmentation']['rot_max']
    batch_size = config['batch_size']
    window_size = config['window_size']
    data = batch['data'].numpy()
    mask = batch['mask'].numpy()
    T_21 = batch['T_21'].numpy()
    _, C, H, W = data.shape
    for i in range(batch_size):
        rot = np.random.uniform(-rot_max, rot_max)
        T = get_transform(0, 0, rot)
        for j in range(1, window_size):
            k = j + i * window_size
            img = data[k].squeeze()
            mmg = mask[k].squeeze()
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 * j / np.pi, 1.0)
            data[i] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
            mask[i] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(1, H, W)
            T_21[i - 1] = np.matmul(T, T_21[i - 1])
    batch['data'] = torch.from_numpy(data)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_21'] = torch.from_numpy(T_21)
    return batch

def augmentBatch2(batch, config):
    """Rotates the cartesian radar image by a random amount, does NOT adjust ground truth transform.
        The keypoints must be unrotated later using the T_aug transform stored in the batch dict.
    """
    rot_max = config['augmentation']['rot_max']
    data = batch['data'].numpy()    # this seems to return a reference, not a copy
    mask = batch['mask'].numpy()
    B, C, H, W = data.shape
    T_aug = []
    for i in range(B):
        if np.mod(i, config['window_size']) == 0:
            continue
        img = data[i].squeeze()
        mmg = mask[i].squeeze()
        rot = np.random.uniform(-rot_max, rot_max)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 / np.pi, 1.0)
        data[i] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        mask[i] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        T_aug += [torch.from_numpy(get_transform(0, 0, -rot))]
    batch['data'] = torch.from_numpy(data)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_aug'] = T_aug
    return batch

def augmentBatch3(batch, config):
    """Shifts the polar radar image by a random amount, does NOT adjust ground truth transform.
        The keypoints must be unrotated later using the T_aug transform stored in the batch dict.
    """
    rot_max = config['augmentation']['rot_max']
    data = batch['data'].numpy()
    polar = batch['polar'].numpy()
    azimuths = batch['azimuths'].numpy()
    mask = batch['mask'].numpy()
    azimuth_res = 0.9 * np.pi / 180
    B, C, H, W = data.shape
    T_aug = []
    for i in range(B):
        if np.mod(i, config['window_size']) == 0:
            continue
        plr = polar[i].squeeze()
        azm = azimuths[i].squeeze()
        rot = np.random.uniform(-rot_max, rot_max)
        rot_azms = int(np.round(rot / azimuth_res))
        rot = rot_azms * azimuth_res
        plr = np.roll(plr, -1 * rot_azms, axis=0)
        cart = radar_polar_to_cartesian(azm, plr, config['radar_resolution'],
                                        config['cart_resolution'], config['cart_pixel_width'])  # 1 x H x W
        data[i] = cart[0]
        polar_mask = mean_intensity_mask(plr)
        msk = radar_polar_to_cartesian(azm, polar_mask, config['radar_resolution'],
                                        config['cart_resolution'], config['cart_pixel_width']).astype(np.float32)
        mask[i] = msk[0]
        T_aug += [torch.from_numpy(get_transform(0, 0, -rot))]
        polar[i] = plr
    batch['data'] = torch.from_numpy(data)
    batch['polar'] = torch.from_numpy(polar)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_aug'] = T_aug
    return batch
