"""
    PyTorch dataset class for the Oxford Radar Robotcar Dataset.
    Authors: Keenan Burnett
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from datasets.radar import load_radar, radar_polar_to_cartesian
from datasets.interpolate_poses import interpolate_ins_poses
from utils.utils import get_inverse_tf

def get_transform_oxford(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for given 2D parameters (x, y, theta).
    Note: (x,y) are position of frame 2 wrt frame 1 as measured in frame 1.
    Args:
        x (float): x translation
        x (float): y translation
        theta (float): rotation
    Returns:
        np.ndarray: 4x4 transformation matrix from next time to current (T_1_2)
    """
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_sequences(path, prefix='2019'):
    """Retrieves a list of all the sequences in the dataset with the given prefix.
        Sequences are subfolders underneath 'path'
    Args:
        path (AnyStr): path to the root data folder
        prefix (AnyStr): each sequence / subfolder must begin with this common string.
    Returns:
        List[AnyStr]: List of sequences / subfolder names.
    """
    sequences = [f for f in os.listdir(path) if prefix in f]
    sequences.sort()
    return sequences

def get_frames(path, extension='.png'):
    """Retrieves all the file names within a path that match the given extension.
    Args:
        path (AnyStr): path to the root/sequence/sensor/ folder
        extension (AnyStr): each data frame must end with this common string.
    Returns:
        List[AnyStr]: List of frames / file names.
    """
    frames = [f for f in os.listdir(path) if extension in f]
    frames.sort()
    return frames

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

class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset."""
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        dataset_prefix = ''
        if config['dataset'] == 'oxford':
            dataset_prefix = '2019'
        elif config['dataset'] == 'boreas':
            dataset_prefix = 'boreas'
        self.T_radar_imu = np.eye(4, dtype=np.float32)
        for i in range(3):
            self.T_radar_imu[i, 3] = config['steam']['ex_translation_vs_in_s'][i]
        sequences = get_sequences(self.data_dir, dataset_prefix)
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.frames = []
        self.seq_lens = []
        for seq in self.sequences:
            seq_frames = get_frames(self.data_dir + seq + '/radar/')
            seq_frames = self.get_frames_with_gt(seq_frames, self.data_dir + seq + '/gt/radar_odometry.csv')
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)

    def get_sequences_split(self, sequences, split):
        """Retrieves a list of sequence names depending on train/validation/test split.
        Args:
            sequences (List[AnyStr]): list of all the sequences, sorted lexicographically
            split (List[int]): indices of a specific split (train or val or test) aftering sorting sequences
        Returns:
            List[AnyStr]: list of sequences that belong to the specified split
        """
        self.split = self.config['train_split']
        if split == 'validation':
            self.split = self.config['validation_split']
        elif split == 'test':
            self.split = self.config['test_split']
        return [seq for i, seq in enumerate(sequences) if i in self.split]

    def get_frames_with_gt(self, frames, gt_path):
        """Retrieves the subset of frames that have groundtruth
        Note: For the Oxford Dataset we do a search from the end backwards because some
            of the sequences don't have GT as the end, but they all have GT at the beginning.
        Args:
            frames (List[AnyStr]): List of file names
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            List[AnyStr]: List of file names with ground truth
        """
        def check_if_frame_has_gt(frame, gt_lines):
            for i in range(len(gt_lines) - 1, -1, -1):
                line = gt_lines[i].split(',')
                if frame == int(line[9]):
                    return True
            return False
        frames_out = frames
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i in range(len(frames) - 1, -1, -1):
                frame = int(frames[i].split('.')[0])
                if check_if_frame_has_gt(frame, lines):
                    break
                frames_out.pop()
        return frames_out

    def get_groundtruth_odometry(self, radar_time, gt_path):
        """Retrieves the groundtruth 4x4 transform from current time to next
        Args:
            radar_time (int): UNIX INT64 timestamp that we want groundtruth for (also the filename for radar)
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            T_2_1 (np.ndarray): 4x4 transformation matrix from current time to next
            time1 (int): UNIX INT64 timestamp of the current frame
            time2 (int): UNIX INT64 timestamp of the next frame
        """
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(',')
                if int(line[9]) == radar_time:
                    T = get_transform_oxford(float(line[2]), float(line[3]), float(line[7]))  # from next time to current
                    return get_inverse_tf(T), int(line[1]), int(line[0]) # T_2_1 from current time step to the next
        assert(0), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)

    def get_groundruth_ins(self, time1, time2, gt_path):
        """Extracts ground truth transform T_2_1 from INS data, from current time1 to time2
        Args:
            time1 (int): UNIX INT64 timestamp of the current frame
            time2 (int): UNIX INT64 timestamp of the next frame
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            T_2_1 (np.ndarray): 4x4 transformation matrix from current time to next
        """
        T = np.array(interpolate_ins_poses(gt_path, [time1], time2)[0])
        return self.T_radar_imu @ T @ get_inverse_tf(self.T_radar_imu)

    def __len__(self):
        return len(self.frames)

    def get_seq_from_idx(self, idx):
        """Returns the name of the sequence that this idx belongs to.
        Args:
            idx (int): frame index in dataset
        Returns:
            AnyStr: name of the sequence that this idx belongs to
        """
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir + seq + '/radar/' + self.frames[idx]
        timestamps, azimuths, _, polar = load_radar(frame)
        data = radar_polar_to_cartesian(azimuths, polar, self.config['radar_resolution'],
                                        self.config['cart_resolution'], self.config['cart_pixel_width'])  # 1 x H x W
        polar_mask = mean_intensity_mask(polar)
        mask = radar_polar_to_cartesian(azimuths, polar_mask, self.config['radar_resolution'],
                                        self.config['cart_resolution'],
                                        self.config['cart_pixel_width']).astype(np.float32)
        # Get ground truth transform between this frame and the next
        radar_time = int(self.frames[idx].split('.')[0])

        T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/gt/radar_odometry.csv')
        if self.config['use_ins']:
            #T_21 = np.array(self.get_groundruth_ins(time1, time2, self.data_dir + seq + '/gps/ins.csv'))
            T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, self.data_dir + seq + '/gt/radar_odometry_ins.csv')
        t_ref = np.array([time1, time2]).reshape(1, 2)
        polar = np.expand_dims(polar, axis=0)
        azimuths = np.expand_dims(azimuths, axis=0)
        timestamps = np.expand_dims(timestamps, axis=0)
        return {'data': data, 'T_21': T_21, 't_ref': t_ref, 'mask': mask, 'polar': polar, 'azimuths': azimuths,
                'timestamps': timestamps}

def get_dataloaders(config):
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
    train_dataset = OxfordDataset(config, 'train')
    valid_dataset = OxfordDataset(vconfig, 'validation')
    test_dataset = OxfordDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader
