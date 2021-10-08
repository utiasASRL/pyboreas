import numpy as np

from pyboreas.data.bounding_boxes import BoundingBox2D
from pyboreas.utils.utils import rotToYawPitchRoll

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
        rotToYawPitchRoll(rotation)
        extent = np.array(list(raw_labels[i]['dimensions'].values())).reshape(3, 1)  # Convert to 2d
        box = BoundingBox2D(pos, rotation, extent, raw_labels[i]['label'])
        boxes.append(box)
    return boxes
