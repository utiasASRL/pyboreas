"""
    PyTorch dataset class for The University of Toronto Boreas Dataset.
    Authors: Keenan Burnett
"""
# TODO: clean this up, make it more generalizable, test it
# TODO: turn this into a generic odometry pytorch dataset class (radar, lidar, vision)
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pytorch.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from utils.radar import load_radar, radar_polar_to_cartesian
from utils.utils import get_inverse_tf, get_transform
from pytorch.oxford import OxfordDataset, mean_intensity_mask

CTS350 = 0    # Oxford
CIR204 = 1    # Boreas
T_prime = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

def roll(r):
    return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]], dtype=np.float64)

def pitch(p):
    return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]], dtype=np.float64)

def yaw(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)

def yawPitchRollToRot(y, p, r):
    """Converts yaw-pitch-roll angles into a 3x3 rotation matrix: SO(3)
    Args:
        y (float): yaw
        p (float): pitch
        r (float): roll
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    Y = yaw(y)
    P = pitch(p)
    R = roll(r)
    C = np.matmul(P, Y)
    return np.matmul(R, C)

def rotToYawPitchRoll(C):
    """Converts a 3x3 rotation matrix SO(3) to yaw-pitch-roll angles
    Args:
        C (np.ndarray): 3x3 rotation matrix
    Returns:
        List[float]: yaw, pitch, roll angles
    """
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i]**2 + C[j, i]**2)
    if c_y > 1e-15:
        r = np.arctan2(C[j, i], C[i, i])
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(C[k, j], C[k, k])
    else:
        r = 0
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(-C[j, k], C[j, j])
    return y, p, r

def get_transform_boreas(gt):
    """Retrieve 4x4 homogeneous transform for a given parsed line of the ground truth pose csv
    Args:
        gt (List[float]): parsed line from ground truth csv file
    Returns:
        np.ndarray: 4x4 transformation matrix (pose of sensor)
    """
    T = np.identity(4, dtype=np.float64)
    C_enu_sensor = yawPitchRollToRot(gt[10], gt[9], gt[8])
    T[0, 3] = gt[2]
    T[1, 3] = gt[3]
    T[2, 3] = gt[4]
    T[0:3, 0:3] = C_enu_sensor
    return T

class BoreasDataset(OxfordDataset):
    """Boreas Radar Dataset"""
    def __init__(self, config, split='train'):
        super().__init__(config, split)
        self.navtech_version = CIR204
        #self.dataloader = dataloadercpp.DataLoader(self.config['radar_resolution'], self.config['cart_resolution'],
        #                                           self.config['cart_pixel_width'], self.navtech_version)

    def get_frames_with_gt(self, frames, gt_path):
        """Retrieves the subset of frames that have groundtruth
        Args:
            frames (List[AnyStr]): List of file names
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            List[AnyStr]: List of file names with ground truth
        """
        drop = self.config['window_size'] - 1
        return frames[:-drop]

    def get_groundtruth_odometry(self, radar_time, gt_path):
        """Retrieves the groundtruth 4x4 transform from current time to next
        Args:
            radar_time (int): UNIX INT64 time stamp that we want groundtruth for (also the filename for radar)
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            np.ndarray: 4x4 transformation matrix from current time to next (T_2_1)
        """
        def parse(gps_line):
            out = [float(x) for x in gps_line.split(',')]
            out[0] = int(gps_line.split(',')[0])
            return out
        gtfound = False
        min_delta = 0.1
        T_2_1 = np.identity(4, dtype=np.float32)
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i in range(len(lines) - 1):
                gt1 = parse(lines[i])
                delta = abs(float(gt1[0] - radar_time) / 1.0e9)
                if delta < min_delta:
                    gt2 = parse(lines[i + 1])
                    T_enu_r1 = np.matmul(get_transform_boreas(gt1), T_prime)
                    T_enu_r2 = np.matmul(get_transform_boreas(gt2), T_prime)
                    T_r2_r1 = np.matmul(get_inverse_tf(T_enu_r2), T_enu_r1)  # 4x4 SE(3)
                    heading, _, _ = rotToYawPitchRoll(T_r2_r1[0:3, 0:3])
                    T_2_1 = get_transform(T_r2_r1[0, 3], T_r2_r1[1, 3], heading)  # 4x4 SE(2)
                    min_delta = delta
                    gtfound = True
        assert(gtfound), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)
        return T_2_1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir + seq + '/radar/' + self.frames[idx]
        cart_frame = self.data_dir + seq + '/radar/cart/' + self.frames[idx]
        mask_frame = self.data_dir + seq + '/radar/mask/' + self.frames[idx]
        cart_pixel_width = self.config['cart_pixel_width']
        num_azimuths = 400
        range_bins = 3768
        if self.navtech_version == CIR204:
            range_bins = 3360

        # Numpy arrays need to be sized correctly before passing them to the dataloader.
        '''timestamps = np.zeros((num_azimuths, 1), dtype=np.int64)
        azimuths = np.zeros((num_azimuths, 1), dtype=np.float32)
        polar = np.zeros((num_azimuths, range_bins), dtype=np.float32)
        data = np.zeros((cart_pixel_width, cart_pixel_width), dtype=np.float32)
        mask = np.zeros((cart_pixel_width, cart_pixel_width), dtype=np.float32)

        self.dataloader.load_radar(frame, timestamps, azimuths, polar)
        self.dataloader.polar_to_cartesian(azimuths, polar, data)
        data = np.expand_dims(data, axis=0)

        polar_mask = mean_intensity_mask(polar)
        self.dataloader.polar_to_cartesian(azimuths, polar_mask, mask)
        mask = np.expand_dims(mask, axis=0)'''

        # Requires that the cartesian images and masks are pre-computed and stored alongside the dataset
        ###########
        timestamps, azimuths, _, polar = load_radar(frame)
        data = np.expand_dims(cv2.imread(cart_frame, cv2.IMREAD_GRAYSCALE).astype(np.float32), axis=0) / 255.0
        mask = np.expand_dims(cv2.imread(mask_frame, cv2.IMREAD_GRAYSCALE).astype(np.float32), axis=0) / 255.0
        ###########

        # Get ground truth transform between this frame and the next
        radar_time = int(self.frames[idx].split('.')[0])
        T_21 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/applanix/radar_poses.csv')
        time1 = timestamps[0, 0]
        if idx + 1 < len(self.frames):
            timestamps2, _, _, _ = load_radar(self.data_dir + seq + '/radar/' + self.frames[idx + 1])
            time2 = timestamps2[0, 0]
        else:
            time2 = time1 + 250000
        t_ref = np.array([time1, time2]).reshape(1, 2)
        azimuths = np.expand_dims(azimuths, axis=0)
        timestamps = np.expand_dims(timestamps, axis=0)
        return {'data': data, 'T_21': T_21, 't_ref': t_ref, 'mask': mask,
                'azimuths': azimuths, 'timestamps': timestamps}

def get_dataloaders_boreas(config):
    """Returns the dataloaders for training models in pytorch.
    Args:
        config (json): parsed configuration file
    Returns:
        train_loader (DataLoader)
        valid_loader (DataLoader)
        test_loader (DataLoader)
    """
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = BoreasDataset(config, 'train')
    valid_dataset = BoreasDataset(vconfig, 'validation')
    test_dataset = BoreasDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader
