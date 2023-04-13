import numpy as np
import numpy.linalg as npla


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
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        return np.array(
            [
                [0, -x[5], x[4], x[0]],
                [x[5], 0, -x[3], x[1]],
                [-x[4], x[3], 0, x[2]],
                [0, 0, 0, 1],
            ]
        )
    print("WARNING: attempted carrot operator on invalid vector shape")
    return xbar


def _vec2rot_analytical(aaxis_ba):
    phi_ba = npla.norm(aaxis_ba, axis=-2, keepdims=True)
    axis = aaxis_ba / phi_ba
    sp = np.sin(phi_ba)
    cp = np.cos(phi_ba)
    return cp * np.eye(3) + (1 - cp) * (axis @ axis.T) + sp * carrot(axis)


def _vec2rot_numerical(aaxis_ba, num_terms=10):
    C_ab = np.eye(3)
    x_small = carrot(aaxis_ba)
    x_small_n = np.eye(3)
    for n in range(1, num_terms + 1):
        x_small_n = x_small_n @ (x_small / n)
        C_ab = C_ab + x_small_n
    return C_ab


def _vec2rot(aaxis_ba, num_terms=0):
    tolerance = 1e-12
    phi_ba = npla.norm(aaxis_ba)
    if (phi_ba < tolerance) or (num_terms != 0):
        return _vec2rot_numerical(aaxis_ba, num_terms)
    else:
        return _vec2rot_analytical(aaxis_ba)


def _vec2jac_analytical(aaxis_ba):
    phi_ba = npla.norm(aaxis_ba)
    axis = aaxis_ba / phi_ba

    sph = np.sin(phi_ba) / phi_ba
    cph = (1 - np.cos(phi_ba)) / phi_ba

    return (
        sph * np.eye(3)
        + (1 - sph) * (axis @ np.swapaxes(axis, -1, -2))
        + cph * carrot(axis)
    )


def _vec2jac_numerical(aaxis_ba, num_terms=10):
    J_ab = np.eye(3)
    x_small = carrot(aaxis_ba)
    x_small_n = np.eye(3)
    for n in range(1, num_terms + 1):
        x_small_n = x_small_n @ (x_small / (n + 1))
        J_ab = J_ab + x_small_n

    return J_ab


def _vec2jac(aaxis_ba, num_terms=0):
    tolerance = 1e-12
    phi_ba = npla.norm(aaxis_ba)
    if (phi_ba < tolerance) or (num_terms != 0):
        return _vec2jac_numerical(aaxis_ba, num_terms)
    else:
        return _vec2jac_analytical(aaxis_ba)


def _vec2tran(xi):
    rho_ba = xi[:3]
    aaxis_ba = xi[3:]
    C_ab = _vec2rot(aaxis_ba)
    J_ab = _vec2jac(aaxis_ba)
    r_ba_ina = J_ab @ rho_ba
    T_ab = np.eye(4)
    T_ab[:3, :3] = C_ab
    T_ab[:3, 3:] = r_ba_ina
    return T_ab


def _rot2vec(C_ab):
    phi_ba = np.arccos(
        np.clip(0.5 * (np.trace(C_ab) - 1), -1, 1)
    )  # clip to avoid numerical issues
    sinphi_ba = np.sin(phi_ba)

    if np.abs(sinphi_ba) > 1e-9:  # General case: phi_ba is NOT near [0, pi, 2*pi]
        axis = (0.5 / sinphi_ba) * (
            np.array(
                [
                    C_ab[2, 1] - C_ab[1, 2],
                    C_ab[0, 2] - C_ab[2, 0],
                    C_ab[1, 0] - C_ab[0, 1],
                ]
            ).reshape(3, 1)
        )
        return phi_ba * axis

    elif np.abs(phi_ba) > 1e-9:  # phi_ba is near [pi, 2*pi]
        eigval, eigvec = npla.eig(C_ab)
        valid_eigval = np.abs(np.real(eigval) - 1) < 1e-10
        valid_axis = np.real(eigvec[:, valid_eigval])
        axis = valid_axis[:, valid_axis.shape[1] - 1].reshape(3, 1)
        aaxis_ba = phi_ba * axis
        if np.abs(np.trace(_vec2rot(aaxis_ba).T @ C_ab) - 3) > 1e-14:
            aaxis_ba = -aaxis_ba
        return aaxis_ba

    else:
        return np.array([[0.0, 0.0, 0.0]]).T


def _vec2jacinv_analytical(aaxis_ba):
    phi_ba = npla.norm(aaxis_ba)
    axis = aaxis_ba / phi_ba
    halfphi = 0.5 * phi_ba

    return (
        halfphi / np.tan(halfphi) * np.eye(3)
        + (1 - halfphi / np.tan(halfphi)) * (axis * axis.T)
        - halfphi * carrot(axis)
    )


def _vec2jacinv_numerical(aaxis_ba, num_terms=10):
    J_ab_inverse = np.eye(3)

    x_small = carrot(aaxis_ba)
    x_small_n = np.eye(3)

    bernoulli = np.array(
        [
            1.0,
            -0.5,
            1.0 / 6.0,
            0.0,
            -1.0 / 30.0,
            0.0,
            1.0 / 42.0,
            0.0,
            -1.0 / 30.0,
            0.0,
            5.0 / 66.0,
            0.0,
            -691.0 / 2730.0,
            0.0,
            7.0 / 6.0,
            0.0,
            -3617.0 / 510.0,
            0.0,
            43867.0 / 798.0,
            0.0,
            -174611.0 / 330.0,
        ]
    )

    for n in range(1, num_terms + 1):
        x_small_n = x_small_n @ (x_small / n)
        J_ab_inverse = J_ab_inverse + bernoulli[n] * x_small_n

    return J_ab_inverse


def _vec2jacinv(aaxis_ba, num_terms=0):
    tolerance = 1e-12
    phi_ba = npla.norm(aaxis_ba)
    if (phi_ba < tolerance) or (num_terms != 0):
        return _vec2jacinv_numerical(aaxis_ba, num_terms)
    else:
        return _vec2jacinv_analytical(aaxis_ba)


def _tran2vec(T_ab):
    """Compute the matrix log of a transformation matrix.
    Compute the inverse of the exponential map (the logarithmic map). This lets us go from a 4x4 transformation matrix
    back to a 6x1 se3 algebra vector (composed of a 3x1 axis-angle vector and 3x1 twist-translation vector). In some
    cases, when the rotation in the transformation matrix is 'numerically off', this involves some 'projection' back to
    SE(3).
      xi_ba = ln(T_ab)
    where xi_ba is the 6x1 se3 algebra vector. Alternatively, we that note that
      xi_ab = -xi_ba = ln(T_ba) = ln(T_ab^{-1})
    See Barfoot-TRO-2014 Appendix B2 for more information.
    Returns:
      np.ndarray: 6x1 se3 algebra vector xi_ba
    """

    aaxis_ba = _rot2vec(T_ab[:3, :3])
    rho_ba = _vec2jacinv(aaxis_ba) @ T_ab[:3, 3:]

    xi_ba = np.concatenate((rho_ba, aaxis_ba))

    return xi_ba
