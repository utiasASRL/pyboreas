import os.path as osp
from pathlib import Path
import numpy as np
import cv2


def load_lidar(path):
    """Loads a pointcloud (np.ndarray) (N, 6) from path [x, y, z, intensity, laser_number, time]"""
    points = np.fromfile(path, dtype=np.float32).reshape((-1, 6)).astype(np.float64)
    t = get_time_from_filename(path)
    points[:, 5] += t
    return points


def roll(r):
    return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]], dtype=np.float64)


def pitch(p):
    return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]], dtype=np.float64)


def yaw(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)


def yawPitchRollToRot(y, p, r):
    Y = yaw(y)
    P = pitch(p)
    R = roll(r)
    C = np.matmul(P, Y)
    return np.matmul(R, C)


def rotToYawPitchRoll(C):
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i]**2 + C[j, i]**2)
    if c_y > 1e-14:
        r = np.arctan2(C[j, i], C[i, i])
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(C[k, j], C[k, k])
    else:
        r = 0
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(-C[j, k], C[j, j])
    return y, p, r


def get_transform(gt):
    """Retrieve 4x4 homogeneous transform for a given parsed line of the ground truth pose csv
    Args:
        gt (List[float]): parsed line from ground truth csv file
    Returns:
        np.ndarray: 4x4 transformation matrix (pose of sensor)
    """
    T = np.identity(4, dtype=np.float64)
    C_enu_sensor = yawPitchRollToRot(gt[9], gt[8], gt[7])
    T[0, 3] = gt[1]
    T[1, 3] = gt[2]
    T[2, 3] = gt[3]
    T[0:3, 0:3] = C_enu_sensor
    return T


def get_transform2(R, t):
    """Returns a 4x4 homogeneous 3D transform
    Args:
        R (np.ndarray): (3,3) rotation matrix
        t (np.ndarray): (3,1) translation vector
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.identity(4, dtype=R.dtype)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T


def get_transform3(x, y, theta, dtype=np.float32):
    """Returns a 4x4 homogeneous 3D transform for a given 2D (x, y, theta).
    Args:
        x (float): x-translation
        y (float): y-translation
        theta (float): rotation
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.identity(4, dtype=dtype)
    T[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


def quaternionToRot(qin):
    """Converts a quaternion to a rotation  matrix
    Args:
        qin (np.ndarray) (4,) [qx, qy, qz, qw] quaternion
    Returns:
        C (np.ndarray) (3,3) rotation matrix
    """
    q = qin.copy().reshape(4, 1)
    if np.matmul(q.transpose(), q) < 1e-14:
        return np.identity(3)
    xi = q[:3].reshape(3, 1)
    eta = q[3, 0]
    C = (eta**2 - np.matmul(xi.transpose(), xi)) * np.identity(3) + \
        2 * np.matmul(xi, xi.transpose()) - 2 * eta * carrot(xi)
    return C


def rotToQuaternion(C):
    """Converts a rotation matrix to a quaternion
    Note that the space of unit-length quaternions is a double-cover of SO(3)
    which means that, C maps to +/- q, so q --> C --> +/- q
    Args:
        C (np.ndarray) (3,3) rotation matrix
    Returns:
        q (np.ndarray) (4,1) [qx, qy, qz, qw] quaternion
    """
    eta = 0.5 * np.sqrt((1 + np.trace(C)))
    if np.abs(eta) < 1e-14:
        eta = 0
        xi = np.sqrt(np.diag(0.5 * (C + np.identity(3))))
        q = np.array([xi[0], xi[1], xi[2], eta]).reshape(4, 1)
    else:
        phi = wrapto2pi(2 * np.arccos(eta))
        eta = np.cos(phi / 2)
        xi_cross = (C.T - C) / (4 * eta)
        q = np.array([xi_cross[2, 1], xi_cross[0, 2], xi_cross[1, 0], eta]).reshape(4, 1)
    return q


def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = np.identity(4, dtype=T.dtype)
    R = T[:3, :3]
    t = T[:3, 3:]
    T2[:3, :3] = R.transpose()
    T2[:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2


def enforce_orthog(T, dim=3):
    """Enforces orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix.
    Args:
        T (np.ndarray): 4x4 transformation matrix
        dim (int): dimensionality of the transform 2==2D, 3==3D
    Returns:
        np.ndarray: 4x4 transformation matrix with orthogonality conditions on the rotation matrix enforced.
    """
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T


def carrot(xbar):
    """Overloaded operator. converts 3x1 vectors into a member of Lie Alebra so(3)
        Also, converts 6x1 vectors into a member of Lie Algebra se(3)
    Args:
        xbar (np.ndarray): if 3x1, xbar is a vector of rotation angles, if 6x1 a vector of 3 trans and 3 rot angles.
    Returns:
        np.ndarray: Lie Algebra 3x3 matrix so(3) if input 3x1, 4x4 matrix se(3) if input 6x1.
    """
    x = xbar.squeeze()
    if x.shape[0] == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        return np.array([[0, -x[5], x[4], x[0]],
                         [x[5], 0, -x[3], x[1]],
                         [-x[4], x[3], 0, x[2]],
                         [0, 0, 0, 1]])
    print('WARNING: attempted carrot operator on invalid vector shape')
    return xbar


def se3ToSE3(xi):
    """Converts 6x1 vectors representing the Lie Algebra, se(3) into a 4x4 homogeneous transform in SE(3)
        Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, r; 0 0 0 1] (4 x 4)
    Args:
        xi (np.ndarray): 6x1 vector
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.identity(4, dtype=np.float32)
    rho = xi[0:3].reshape(3, 1)
    phibar = xi[3:6].reshape(3, 1)
    phi = np.linalg.norm(phibar)
    R = np.identity(3)
    if phi != 0:
        phibar /= phi  # normalize
        I = np.identity(3)
        R = np.cos(phi) * I + (1 - np.cos(phi)) * phibar @ phibar.T + np.sin(phi) * carrot(phibar)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * phibar @ phibar.T + \
            carrot(phibar) * (1 - np.cos(phi)) / phi
        rho = J @ rho
    T[0:3, 0:3] = R
    T[0:3, 3:] = rho
    return T


def SE3Tose3(T):
    """Converts 4x4 homogeneous transforms in SE(3) to 6x1 vectors representing the Lie Algebra, se(3)
        SE(3) T = [C, r; 0 0 0 1] (4 x 4) --> Lie Vector xi = [rho, phi]^T (6 x 1)
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: 6x1 vector
    """
    R = T[0:3, 0:3]
    evals, evecs = np.linalg.eig(R)
    idx = -1
    for i in range(3):
        if evals[i].real != 0 and evals[i].imag == 0:
            idx = i
            break
    assert(idx != -1)
    abar = evecs[idx].real.reshape(3, 1)
    phi = np.arccos((np.trace(R) - 1) / 2)
    rho = T[0:3, 3:]
    if phi != 0:
        I = np.identity(3)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * abar @ abar.T + \
            carrot(abar) * (1 - np.cos(phi)) / phi
        rho = np.linalg.inv(J) @ rho
    xi = np.zeros((6, 1))
    xi[0:3, 0:] = rho
    xi[3:, 0:] = phi * abar
    return xi


def rotation_error(T):
    """Calculates a single rotation value corresponding to the upper-left 3x3 rotation matrix.
        Uses axis-angle representation to get a single number for rotation
    Args:
        T (np.ndarray): 4x4 transformation matrix T = [C, r; 0 0 0 1]
    Returns:
        float: rotation
    """
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(T, dim=3):
    """Calculates a euclidean distance corresponding to the translation vector within a 4x4 transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix T = [C, r; 0 0 0 1]
        dim (int): If dim=2 we only use x,y, otherwise we use all dims.
    Returns:
        float: translation distance
    """
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)


def wrapto2pi(phi):
    """Ensures that the output angle phi is within the interval [0, 2*pi)"""
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    return phi


def get_time_from_filename(file):
    """Retrieves an epoch time from a file name in seconds"""
    tstr = str(Path(file).stem)
    gpstime = float(tstr)
    timeconvert = 1e-9
    if len(tstr) < 19:
        timeconvert = 10**(-1 * (len(tstr) - 10))
    return gpstime * timeconvert


EARTH_SEMIMAJOR = 6378137.0
EARTH_SEMIMINOR = 6356752.0
EARTH_ECCEN = 0.081819190842622
a = EARTH_SEMIMAJOR
eccSquared = EARTH_ECCEN**2
eccPrimeSquared = (eccSquared) / (1 - eccSquared)
k0 = 0.9996     # scale factor
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 1.0 / DEG_TO_RAD


def LLtoUTM(latitude, longitude):
    """Converts a lat-long position into metric UTM coordinates (x-y-z)
    Args:
        latitude (float) in radians
        longitude (float) in radians
    Returns:
        UTMEasting (float): metric position in easting
        UTMNorthing (float): metric position in northing
        zoneNumber (float): UTM zone number
    """
    while longitude < -1 * np.pi:
        longitude += 2 * np.pi
    while longitude >= np.pi:
        longitude -= 2 * np.pi
    longDeg = longitude * RAD_TO_DEG
    latDeg = latitude * RAD_TO_DEG
    zoneNumber = int((longDeg + 180) / 6) + 1
    # +3 puts origin in middle of zone
    longOrigin = (zoneNumber - 1) * 6 - 180 + 3
    longOriginRad = longOrigin * DEG_TO_RAD
    N = a / np.sqrt(1 - eccSquared * np.sin(latitude) * np.sin(latitude))
    T = np.tan(latitude) * np.tan(latitude)
    C = eccPrimeSquared * np.cos(latitude) * np.cos(latitude)
    A = np.cos(latitude) * (longitude - longOriginRad)
    M = a * ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 - 5 * eccSquared * eccSquared * eccSquared / 256) * latitude -
        (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32 + 45 * eccSquared * eccSquared * eccSquared / 1024) * np.sin(2 * latitude) +
        (15 * eccSquared * eccSquared / 256 + 45 * eccSquared * eccSquared * eccSquared / 1024) * np.sin(4 * latitude) -
        (35 * eccSquared * eccSquared * eccSquared / 3072) * np.sin(6 * latitude))
    UTMEasting = k0 * N * (A + (1 - T + C) * A * A * A / 6 +
        (5 - 18 * T + T * T + 72 * C - 58 * eccPrimeSquared) * A * A * A * A * A / 120) + 500000.0
    UTMNorthing = k0 * (M + N * np.tan(latitude) *
        (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24 +
        (61 - 58 * T + T * T + 600 * C - 330 * eccPrimeSquared) * A * A * A * A * A * A / 720))

    if latitude < 0:
        # 10000000 meter offset for southern hemisphere
        UTMNorthing += 10000000.0

    return UTMEasting, UTMNorthing, zoneNumber


def get_gt_data_for_frame(root, sensType, frame):
    """Retrieves ground truth applanix data for a given sensor frame
    Args:
        root (str): path to the sequence root
        sensType (str): [camera, lidar, or radar]
        frame (str): name/timestampd of the given sensor frame (without the extension)
    Returns:
        gt (list): A list of ground truth values from the applanix sensor_poses.scv
    """
    posepath = osp.join(root, 'applanix', sensType + '_poses.csv')
    with open(posepath, 'r') as f:
        f.readline()  # header
        for line in f:
            if line.split(',')[0] == frame:
                return [float(x) for x in line.split(',')]
    assert(0), 'gt not found for root: {} sensType: {} frame: {}'.format(root, sensType, frame)
    return None


def binaryDistSearch(arr, l, r, x):
    """Performs a binary search to find the index of the element in arr
    which is closest to x. O(log n)
    Note: this function may also return the second-closest element to x,
    so a follow-up check must be done on one index above and below
    Args:
        arr (list): Sorted list of float values
        l (int): index of the left-most section of the array to search
        r (int): index of the right-most section of the array to search
        x (float): query value
    Returns:
        d (float): absolute distance between arr[idx] and x
        idx (int): index of the closest element in array to x
    """
    if r == l + 1:
        if abs(arr[l] - x) < abs(arr[r] - x):
            return abs(arr[l] - x), l
        else:
            return abs(arr[r] - x), r
    mid = l + (r - l) // 2
    if arr[mid] == x:
        return 0, mid
    elif r - l == 0:
        return abs(arr[l] - x), l
    elif arr[mid] > x:
        return binaryDistSearch(arr, l, mid - 1, x)
    else:
        return binaryDistSearch(arr, mid + 1, r, x)


def get_closest_index(query, targets):
    """Retrieves the index of the element in targets that is closest to query O(log n)
    Args:
        query (float): query value
        targets (list): Sorted list of float values
    Returns:
        idx (int): index of the closest element in the array to x
    """
    d, idx = binaryDistSearch(targets, 0, len(targets)-1, query)
    # check if index above or below is closer to query
    if targets[idx] < query and idx < len(targets) - 1:
        if abs(targets[idx + 1] - query) < d:
            return idx + 1
    elif targets[idx] > query and idx > 0:
        if abs(targets[idx - 1] - query) < d:
            return idx - 1
    return idx


def is_sorted(x):
    """Returns True is x is a sorted list, otherwise False"""
    return (np.diff(x) >= 0).all()
