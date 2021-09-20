import os
import numpy as np

def load_lidar(path, dim=6):
    points = np.fromfile(path, dtype=np.float32).reshape((-1, 6)).astype(np.float64)
    t = float(path.split('/')[-1].split('.')[0]) * 1e-6
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

def rotToYawPitchRoll(C, eps = 1e-15):
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i]**2 + C[j, i]**2)
    if c_y > eps:
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

def quaternionToRot(q):
    EPS = 1e-15
    if np.matmul(q.transpose(), q) < EPS:
        return np.identity(3)
    xi = q[0:3].reshape(3, 1)
    eta = q[3, 0]
    C = (eta**2 - np.matmul(xi.transpose(), xi)) * np.identity(3) +
        2 * np.matmul(xi, xi.transpose()) - 2 * eta * carrot(xi)
    return C.transpose()

def rotToQuaternion(C):
    phi = np.arccos((np.trace(C) - 1) / 2)
    evalues, evectors = np.linalg.eig(C)
    abar = None
    for i, evalue in enumerate(evalues):
        if evalue.imag == 0 and evalue.real != 0:
            abar = evectors[i]
    assert(abar is not None)
    xi = abar * np.sin(phi / 2)
    eta = np.cos(phi / 2)
    q = np.array([xi[0], x[1], x[2], eta]).reshape(4, 1)
    return q

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = np.identity(4, dtype=T.dtype)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
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

def rotationError(T):
    """Calculates a single rotation value corresponding to the upper-left 3x3 rotation matrix.
        Uses axis-angle representation to get a single number for rotation
    Args:
        T (np.ndarray): 4x4 transformation matrix T = [C, r; 0 0 0 1]
    Returns:
        float: rotation
    """
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(T, dim=2):
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

def computeMedianError(T_gt, T_pred):
    """Computes the median translation and rotation errors along with their standard deviations.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix
    Returns:
        t_err_med (float): median translation error
        t_err_std (float): standard dev translation error
        r_err_med (float): median rotation error
        r_err_std (float): standard dev rotation error
        t_err_mean (float): mean translation error
        r_err_mean (float): mean rotation error
        t_error (List[float]): list of all translation errors
        r_error (List[float]): list of all rotation errors
    """
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error), np.mean(t_error),
            np.mean(r_error), t_error, r_error]

def wrapto2pi(phi):
    """Ensures that the output angle phi is within the interval [0, 2*pi)"""
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    return phi
