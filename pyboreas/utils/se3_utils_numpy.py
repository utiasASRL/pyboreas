# numpy adaption of pytorch tensor code found here: https://github.com/utiasSTARS/dpc-net/blob/master/lie_algebra.py
# (some functions may be missing here, and not fully tested, so there may be bugs)
# David J. Yoon (ASRL)

import numpy as np

global EPS
# EPS = 1e-10
EPS = 1e-8

def adjoint(T):
    if T.ndim < 3:
        T = T[None]

    Tad = np.zeros((T.shape[0], 6, 6), dtype=T.dtype)
    C = T[:, :3, :3]
    Jrho = T[:, :3, -1]
    Tad[:, :3, :3] = C
    Tad[:, 3:, 3:] = C
    Tad[:, :3, 3:] = so3_wedge(Jrho) @ C
    return Tad

def cdot(q):
    # input Nx3 or Nx4
    out = np.zeros((q.shape[0], 4, 6), dtype=q.dtype)
    out[:, :3, :3] = np.eye(3, dtype=q.dtype)
    out[:, :3, 3:] = -so3_wedge(q)
    if q.shape[-1] == 4:
        out *= q[:, 3][:, None, None]

    # Nx4x6
    return out

def cdot2d(q):
    # input NxMx3 or NxMx4
    out = np.zeros((q.shape[0], q.shape[1], 4, 6), dtype=q.dtype)
    out[:, :, :3, :3] = np.eye(3, dtype=q.dtype)
    out[:, :, :3, 3:] = -so3_wedge_2d(q)

    if q.shape[-1] == 4:
        out *= q[:, :, 3][:, :, None, None]

    # NxMx4x6
    return out

def cdot_(q):
    # input (...)x3 or (...)x4
    out = np.zeros(np.concatenate([q.shape[:-1], [4, 6]]), dtype=q.dtype)
    out[..., :3, :3] = np.eye(3, dtype=q.dtype)
    out[..., :3, 3:] = -so3_wedge_(q)

    if q.shape[-1] == 4:
        # homogenous element was given and may not be 1
        out *= q[..., 3][..., None, None]

    # (...)x4x6
    return out

def so3_wedge(phi):
    # Returns Nx3x3 tensor with each 1x3 row vector in phi wedge'd

    Phi = np.zeros((phi.shape[0], 3, 3), dtype=phi.dtype)

    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return Phi

def so3_wedge_2d(phi):
    # Returns NxMx3x3 tensor with each 1x3 row vector in phi wedge'd

    Phi = np.zeros((phi.shape[0], phi.shape[1], 3, 3), dtype=phi.dtype)

    Phi[:, :, 0, 1] = -phi[:, :, 2]
    Phi[:, :, 1, 0] = phi[:, :, 2]
    Phi[:, :, 0, 2] = phi[:, :, 1]
    Phi[:, :, 2, 0] = -phi[:, :, 1]
    Phi[:, :, 1, 2] = -phi[:, :, 0]
    Phi[:, :, 2, 1] = phi[:, :, 0]
    return Phi

def so3_wedge_(phi):
    # Input (...)x3 tensor
    # Returns (...)x3x3 tensor with each 3d vector in phi wedge'd
    Phi = np.zeros(np.concatenate([phi.shape[:-1], [3, 3]]), dtype=phi.dtype)

    Phi[..., 0, 1] = -phi[..., 2]
    Phi[..., 1, 0] = phi[..., 2]
    Phi[..., 0, 2] = phi[..., 1]
    Phi[..., 2, 0] = -phi[..., 1]
    Phi[..., 1, 2] = -phi[..., 0]
    Phi[..., 2, 1] = phi[..., 0]
    return Phi

def se3_Q(rho, phi):
    #SE(3) Q function
    #Used in the SE(3) jacobians
    #See b

    ph = vec_norms(phi)

    # ph_test = phi.norm(p=2, dim=1)

    ph2 = ph*ph
    ph3 = ph2*ph
    ph4 = ph3*ph
    ph5 = ph4*ph

    rx = so3_wedge(rho)
    px = so3_wedge(phi)

    cph = np.cos(ph)
    sph = np.sin(ph)

    m1 = 0.5
    m2 = (ph - sph)/ph3
    m3 = (ph2 + 2. * cph - 2.)/(2.*ph4)
    m4 = (2.*ph - 3.*sph + ph*cph)/(2.*ph5)

    t1 = m1 * rx
    t2 = m2[:, :, None] * (px @ rx + rx @ px + px @ rx @ px)
    t3 = m3[:, :, None] * (px @ px @ rx + rx @ px @ px - 3. * px @ rx @ px)
    t4 = m4[:, :, None] * (px @ rx @ px @ px + px @ px @ rx @ px)

    Q = t1 + t2 + t3 + t4

    return Q

def se3_inv(T):

    if T.ndim < 3:
        T = T[None]

    #Batch invert Nx4x4 SE(3) matrices
    Rt = T[:, 0:3, 0:3].transpose(0, 2, 1)
    t = T[:, 0:3, 3:4]

    T_inv = T.copy()

    T_inv[:, 0:3, 0:3] = Rt
    T_inv[:, 0:3, 3:4] = -Rt @ t

    return T_inv

def batch_trace(R):
    #Takes in Nx3x3, computes trace of each 3x3 matrix, outputs Nx1 vector with traces
    # I = np.zeros((3,3), dtype=R.dtype)
    # I[0,0] = I[1,1] = I[2,2] = 1.0
    # return (R*I.expand_as(R)).sum(1, keepdim=True).sum(2, keepdim=True).view(R.size(0),-1)
    return np.trace(R, axis1=1, axis2=2)[:, None]

def batch_outer_prod(vecs):
    # Input: NxD vectors
    # Output: NxDxD outer products
    # N = vecs.size(0)
    # D = vecs.size(1)
    # return vecs.unsqueeze(2).expand(N, D, D) * vecs.unsqueeze(1).expand(N, D, D)
    return vecs[:, :, None] @ vecs[:, None, :]

def vec_norms(input):
    # Takes Nx3 tensor of row vectors an outputs Nx1 tensor with 2-norms of each row
    return np.linalg.norm(input, axis=1, keepdims=True)

def se3_wedge(xi):
    if xi.ndim == 1:
        xi = xi[None]
    #Returns Nx4x4 tensor with each 1x6 row vector in xi SE(3) wedge'd
    Xi = np.zeros((xi.shape[0], 4, 4), dtype=xi.dtype)
    rho = xi[:, 0:3]
    phi = xi[:, 3:6]
    Phi = so3_wedge(phi)

    Xi[:, 0:3, 0:3] = Phi
    Xi[:, 0:3, 3:4] = rho[:, :, None]

    return Xi

def so3_vee(Phi):
    # Returns Nx3 tensor with each 3x3 lie algebra element converted to a 1x3 coordinate vector
    phi = np.zeros((Phi.shape[0], 3), dtype=Phi.dtype)
    phi[:, 0] = Phi[:, 2, 1]
    phi[:, 1] = Phi[:, 0, 2]
    phi[:, 2] = Phi[:, 1, 0]
    return phi

def so3_exp(phi):
    # input: phi Nx3
    # output: perturbation Nx3x3

    if phi.ndim < 2:
        phi = phi[None]

    batch_size = phi.shape[0]

    # Take the norms of each row
    angles = vec_norms(phi)

    I = np.eye(3, dtype=phi.dtype)

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = (angles < EPS).reshape(-1)
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = np.nonzero(small_angles_mask)[0]

    if small_angles_num == batch_size:
        # Taylor expansion
        phi_w = so3_wedge(phi)
        return I[None] + phi_w

    # axes = phi / angles.expand(batch_size, 3)
    axes = phi / angles
    s = np.sin(angles)
    c = np.cos(angles)

    outer_prod_axes = batch_outer_prod(axes)
    R = c[:, :, None] * I[None] + (1 - c[:, :, None]) * outer_prod_axes + s[:, :, None] * so3_wedge(axes)

    if 0 < small_angles_num < batch_size:
        phi_w = so3_wedge(phi[small_angles_indices])
        small_exp = I[None] + phi_w
        R[small_angles_indices] = small_exp

    return R

def so3_log(R):
    #input: R 64x3x3
    #output: log(R) 64x3

    batch_size = R.shape[0]

    # The rotation axis (not unit-length) is given by
    axes = np.zeros((batch_size, 3), dtype=R.dtype)

    axes[:,0] = R[:, 2, 1] - R[:, 1, 2]
    axes[:,1] = R[:, 0, 2] - R[:, 2, 0]
    axes[:,2] = R[:, 1, 0] - R[:, 0, 1]


    # The sine of the rotation angle is half the norm of the axis
    # This does not work well??
    #sin_angles = 0.5 * vec_norms(axes)
    #angles = torch.atan2(sin_angles, cos_angles)

    # The cosine of the rotation angle is related to the trace of C

    #NOTE: clamp ensures that we don't get any nan's due to out of range numerical errors
    # angles = np.arccos( np.clip(0.5 * batch_trace(R) - 0.5, -1+EPS, 1-EPS) )
    angles = np.arccos( np.clip(0.5 * batch_trace(R) - 0.5, -1, 1) )    # removed EPS deltas, but still clip
    # angles = np.arccos( 0.5 * batch_trace(R) - 0.5 )
    sin_angles = np.sin(angles)


    # If angle is close to zero, use first-order Taylor expansion
    # small_angles_mask = angles.lt(EPS).view(-1)
    # small_angles_num = small_angles_mask.sum()
    small_angles_mask = (angles < EPS).reshape(-1)
    small_angles_num = small_angles_mask.sum()

    #This tensor is used to extract the 3x3 R's that correspond to small angles
    # small_angles_indices = small_angles_mask.nonzero(as_tuple=False).squeeze(1)     # TODO: check if this fix is correct
    small_angles_indices = np.nonzero(small_angles_mask)[0]


    if small_angles_num == 0:
        #Regular log
        ax_sin = axes / sin_angles
        logs = 0.5 * angles * ax_sin

    elif small_angles_num == batch_size:
        #Small angle Log
        # I = R.new(3, 3).zero_()
        # I[0,0] = I[1,1] = I[2,2] = 1.0
        # I = I.expand(batch_size, 3,3) #I is now batch_sizex3x3
        # logs = so3_vee(R - I)

        I = np.eye(3, dtype=R.dtype)
        logs = so3_vee(R - I[None])
    else:
        #Some combination of both
        I = np.eye(3, dtype=R.dtype)
        ax_sin = np.zeros_like(axes)
        ax_sin[~small_angles_mask] = axes[~small_angles_mask] / sin_angles[~small_angles_mask]
        logs = 0.5 * angles * ax_sin

        small_logs = so3_vee(R[small_angles_indices] - I[None])
        logs[small_angles_indices] = small_logs

    # above does not generally work when R is symmetric (i.e., angle = pi)
    # note: symmetric case where angle = 0 is covered above
    # sym_R_mask = np.linalg.norm(R - np.transpose(R, [0, 2, 1]), axis=(1, 2)) < EPS # note: numpy transpose sets permutation axes
    # sym_R_mask *= (~small_angles_mask)  # so we ignore angle = 0 cases
    sym_R_mask = np.abs(angles[:, 0] - np.pi) < EPS   # just check for pi. arccos returns [0, pi]

    if sym_R_mask.sum() == 0:
        return logs
    else:
        # need to find eigenvectors that correspond to eigenvalue of 1
        eval, evec = np.linalg.eigh(R[sym_R_mask])
        ids = (np.abs(eval - np.ones_like(eval))).argmin(axis= 1, keepdims=True).squeeze(-1)
        assert(ids.shape[0] == sym_R_mask.sum())   # make sure we are getting 1 per angle. Otherwise there are cases with multiple eigvals = 1
        symaxes = evec[np.arange(ids.shape[0]), :, ids]
        logs[sym_R_mask] = symaxes * angles[sym_R_mask]
        return logs

def so3_inv_left_jacobian(phi):
    """Inverse left SO(3) Jacobian (see Barfoot).
    """

    angles = vec_norms(phi)
    I = np.eye(3, dtype=phi.dtype)
    batch_size = phi.shape[0]

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = (angles < EPS).reshape(-1)
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = np.nonzero(small_angles_mask)[0]

    if small_angles_num == batch_size:
        return I[None] - 0.5*so3_wedge(phi)

    axes = np.zeros_like(phi)
    axes[~small_angles_mask] = phi[~small_angles_mask] / angles[~small_angles_mask]
    half_angles = 0.5 * angles
    cot_half_angles = np.zeros_like(half_angles)
    cot_half_angles[~small_angles_mask] = 1. / np.tan(half_angles[~small_angles_mask])

    #Compute outer products of the Nx3 axes vectors, and put them into a Nx3x3 tensor
    outer_prod_axes = batch_outer_prod(axes)

    #This piece of magic changes the vector so that it can multiply each 3x3 matrix of a Nx3x3 tensor
    # h_a_cot_a = (half_angles * cot_half_angles).view(-1,1,1).expand_as(I_full)
    # h_a = half_angles.view(-1,1,1).expand_as(I_full)
    # invJ = h_a_cot_a * I_full + (1 - h_a_cot_a) * outer_prod_axes - (h_a * so3_wedge(axes))

    h_a_cot_a = half_angles * cot_half_angles
    invJ = h_a_cot_a[:, :, None] * I[None] + (1. - h_a_cot_a[:, :, None]) * outer_prod_axes - (half_angles[:, :, None] * so3_wedge(axes))

    if 0 < small_angles_num < batch_size:
        small_invJ = I[None] - 0.5*so3_wedge(phi[small_angles_indices])
        invJ[small_angles_indices] = small_invJ

    return invJ

def se3_log(T):

    """Logarithmic map for SE(3)
    Computes a SE(3) tangent vector from a transformation
    matrix.
    This is the inverse operation to exp
    #input: T Nx4x4
    #output: log(T) Nx6
    """
    if T.ndim < 3:
        T = T[None]

    R = T[:,0:3,0:3]
    t = T[:,0:3,3:4]
    # sample_size = t.shape[0]
    phi = so3_log(R)
    invl_js = so3_inv_left_jacobian(phi)
    # rho = (invl_js.bmm(t)).view(sample_size, 3)
    rho = (invl_js @ t)[:, :, 0]
    # xi = torch.cat((rho, phi), 1)
    xi = np.concatenate((rho, phi), axis=1)

    return xi

def se3_curly_wedge(xi):
    #Returns Nx4x4 tensor with each 1x6 row vector in xi SE(3) curly wedge'd
    Xi = np.zeros((xi.shape[0], 6, 6), dtype=xi.dtype)
    rho = xi[:, 0:3]
    phi = xi[:, 3:6]
    Phi = so3_wedge(phi)
    Rho = so3_wedge(rho)

    Xi[:, 0:3, 0:3] = Phi
    Xi[:, 0:3, 3:6] = Rho
    Xi[:, 3:6, 3:6] = Phi

    return Xi

def se3_exp(xi):
    #input: xi Nx6
    #output: T Nx4x4
    #New efficient way without having to compute Q!

    if xi.ndim < 2:
        xi = xi[None]

    batch_size = xi.shape[0]
    phi = vec_norms(xi[:, 3:6])

    I = np.eye(4, dtype=xi.dtype)

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = (phi < EPS).reshape(-1)
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = np.nonzero(small_angles_mask)[0]

    if small_angles_num == batch_size:
        #Taylor expansion
        xi_w = se3_wedge(xi)
        return I[None] + xi_w

    xi_w = se3_wedge(xi)
    xi_w2 = xi_w @ xi_w
    xi_w3 = xi_w2 @ xi_w

    phi2 = np.power(phi, 2)
    phi3 = np.power(phi, 3)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # t2 = (1 - cos_phi)/phi2
    # t3 = (phi - sin_phi)/phi3
    t2 = np.zeros_like(phi)
    t2[~small_angles_mask] = (1 - cos_phi[~small_angles_mask])/phi2[~small_angles_mask]
    t3 = np.zeros_like(phi)
    t3[~small_angles_mask] = (phi[~small_angles_mask] - sin_phi[~small_angles_mask])/phi3[~small_angles_mask]
    T = I[None] + xi_w + t2[:, :, None]*xi_w2 + t3[:, :, None]*xi_w3

    if 0 < small_angles_num < batch_size:
        xi_w = se3_wedge(xi[small_angles_indices])
        small_exp = I[None] + xi_w
        T[small_angles_indices] = small_exp

    return T

def se3_inv_left_jacobian(xi):
    """Computes SE(3) inverse left jacobian of N xi vectors (arranged into NxD tensor)"""
    if xi.ndim < 2:
        xi = xi[None]

    rho = xi[:, 0:3]
    phi = xi[:, 3:6]

    batch_size = xi.shape[0]
    angles = vec_norms(xi[:, 3:6])

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = (angles < EPS).reshape(-1)
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = np.nonzero(small_angles_mask)[0]

    if small_angles_num == batch_size:
        #Taylor expansion
        I = np.eye(6, dtype=xi.dtype)
        return I[None] - 0.5*se3_curly_wedge(xi)

    invl_j = so3_inv_left_jacobian(phi)
    Q = se3_Q(rho, phi)
    zero_mat = np.zeros((batch_size, 3, 3), dtype=xi.dtype)

    upper_rows = np.concatenate((invl_j, -invl_j @ Q @ invl_j), axis=2)
    lower_rows = np.concatenate((zero_mat, invl_j), axis=2)

    inv_J = np.concatenate((upper_rows, lower_rows), axis=1)

    if 0 < small_angles_num < batch_size:
        I = np.eye(6, dtype=xi.dtype)
        small_inv_J =  I[None] - 0.5*se3_curly_wedge(xi[small_angles_indices])
        inv_J[small_angles_indices] = small_inv_J

    return inv_J

def roll(x):
    C = np.zeros((x.shape[0], 3, 3), dtype=x.dtype)
    C[:, 0, 0] = 1
    C[:, 1, 1] = np.cos(x)
    C[:, 1, 2] = np.sin(x)
    C[:, 2, 1] = -np.sin(x)
    C[:, 2, 2] = np.cos(x)
    return C

def pitch(x):
    C = np.zeros((x.shape[0], 3, 3), dtype=x.dtype)
    C[:, 0, 0] = np.cos(x)
    C[:, 0, 2] = -np.sin(x)
    C[:, 1, 1] = 1
    C[:, 2, 0] = np.sin(x)
    C[:, 2, 2] = np.cos(x)
    return C

def yaw(x):
    C = np.zeros((x.shape[0], 3, 3), dtype=x.dtype)
    C[:, 0, 0] = np.cos(x)
    C[:, 0, 1] = np.sin(x)
    C[:, 1, 0] = -np.sin(x)
    C[:, 1, 1] = np.cos(x)
    C[:, 2, 2] = 1
    return C

def ypr2rot(r, p, y):
    return roll(r) @ pitch(p) @ yaw(y)