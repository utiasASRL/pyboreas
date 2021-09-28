import numpy as np

from data_classes.bounding_boxes import BoundingBox2D


def to_T(C, r):
    T = np.concatenate((C, r), axis=1)
    T = np.concatenate((T, [[0, 0, 0, 1]]), axis=0)
    return T


def get_dataset_offset_camera_ts(dataset):
    if dataset == "scale":
        return 6.3e9
    elif dataset == "boreas":
        return 0
    else:
        raise Exception('Unknown dataset!')


def transform_points(T, raw_pcd, keep_prob=1.0):
    """
    Transform raw lidar points by a SE3 transformation
    :param T_vo: required transformation
    :param raw_pcd: original point cloud data
    """
    points = np.zeros([len(raw_pcd), 3])
    for i in range(len(raw_pcd)):
        points[i][0] = raw_pcd[i]['x']
        points[i][1] = raw_pcd[i]['y']
        points[i][2] = raw_pcd[i]['z']
    points = points[np.random.choice(len(raw_pcd), int(keep_prob * len(raw_pcd)), replace=False)]
    points = np.matmul(T,
                       np.vstack((points.T, np.ones(points.shape[0]))))
    points = points.T[:]
    return points


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
        rotation = np.matmul(C_yaw, rot_z(raw_labels[i]['yaw']))
        rot_to_yaw_pitch_roll(rotation)
        extent = np.array(list(raw_labels[i]['dimensions'].values())).reshape(3, 1)  # Convert to 2d
        box = BoundingBox2D(pos, rotation, extent, raw_labels[i]['label'])
        boxes.append(box)
    return boxes


def rot_x(alpha):
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])


def rot_y(beta):
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                     [0, 1, 0],
                     [-np.sin(beta), 0, np.cos(beta)]])


def rot_z(gamma):
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])


def rot_to_yaw_pitch_roll(C, eps=1e-15):
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i] ** 2 + C[j, i] ** 2)
    if c_y > eps:
        roll = np.arctan2(C[j, i], C[i, i])
        pitch = np.arctan2(-C[k, i], c_y)
        yaw = np.arctan2(C[k, j], C[k, k])
    else:
        roll = 0
        pitch = np.arctan2(-C[k, i], c_y)
        yaw = np.arctan2(-C[j, k], C[j, j])
    return yaw, pitch, roll
