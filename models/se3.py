# https://github.com/utiasSTARS/liegroups
# modified into "backward - able" version
import torch


def vee(Phi):
    if Phi.dim() < 3:
        Phi = Phi.unsqueeze(dim=0)

    if Phi.shape[1:3] != (3, 3):
        raise ValueError("Phi must have shape (3,3) or (N,3,3)")

    phi = Phi.new_empty(Phi.shape[0], 3)
    phi[:, 0] = Phi[:, 2, 1]
    phi[:, 1] = Phi[:, 0, 2]
    phi[:, 2] = Phi[:, 1, 0]
    return phi.squeeze()


def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze()


def trace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device) * mat).sum(dim=1).sum(dim=1)

    return tr.view(mat.shape[0])


def wedge(phi):
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    if phi.shape[1] != 3:
        raise ValueError(
            "phi must have shape ({},) or (N,{})".format(3, 3))

    Phi = phi.new_empty(phi.shape[0], 3, 3).zero_().type(phi.type())
    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return Phi.squeeze()


def se3_exp(xi):
    if xi.dim() < 2:
        xi = xi.unsqueeze(dim=0)

    if xi.shape[1] != 6 or xi.dim() != 2:
        raise ValueError(
            "xi must have shape ({},) or (N,{})".format(6, 6))

    rho = xi[:, :3]
    phi = xi[:, 3:]

    rot = so3_exp(phi)
    rot_jac = so3_left_jacobian(phi)

    if rot_jac.dim() < 3:
        rot_jac = rot_jac.unsqueeze(dim=0)
    if rho.dim() < 3:
        rho = rho.unsqueeze(dim=2)

    trans = torch.bmm(rot_jac, rho)

    if rot.dim() < 3:
        rot = rot.unsqueeze(dim=0)

    if trans.dim() < 2:
        trans = trans.unsqueeze(dim=0)

    T = torch.cat([rot, trans], dim=2)

    return T


def so3_exp(phi):
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    mat = phi.new_empty(phi.shape[0], 3, 3, device=phi.device)
    angle = phi.norm(p=2, dim=1)

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = isclose(angle, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze(dim=1)

    if len(small_angle_inds) > 0:
        mat[small_angle_inds] = \
            torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(mat[small_angle_inds]) + \
            wedge(phi[small_angle_inds])

    # Otherwise...
    large_angle_mask = ~small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze(dim=1)

    if len(large_angle_inds) > 0:
        angle = angle[large_angle_inds]
        axis = phi[large_angle_inds] / \
               angle.unsqueeze(dim=1).expand(len(angle), 3)
        s = angle.sin().unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(mat[large_angle_inds])
        c = angle.cos().unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(mat[large_angle_inds])

        A = c * torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(dim=0).expand_as(
            mat[large_angle_inds])
        B = (1. - c) * outer(axis, axis)
        C = s * wedge(axis)

        mat[large_angle_inds] = A + B + C

    return mat.squeeze()


def so3_left_jacobian(phi):
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    if phi.shape[1] != 3:
        raise ValueError(
            "phi must have shape ({},) or (N,{})".format(3, 3))

    jac = phi.new_empty(phi.shape[0], 3, 3)
    angle = phi.norm(p=2, dim=1)

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = isclose(angle, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze(dim=1)
    if len(small_angle_inds) > 0:
        jac[small_angle_inds] = \
            torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(jac[small_angle_inds]) + \
            0.5 * wedge(phi[small_angle_inds])

    # Otherwise...
    large_angle_mask = ~small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze(dim=1)

    if len(large_angle_inds) > 0:
        angle = angle[large_angle_inds]
        axis = phi[large_angle_inds] / \
            angle.unsqueeze(dim=1).expand(len(angle), 3)
        s = angle.sin()
        c = angle.cos()

        A = (s / angle).unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(jac[large_angle_inds]) * \
            torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(dim=0).expand_as(
            jac[large_angle_inds])
        B = (1. - s / angle).unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(jac[large_angle_inds]) * \
            outer(axis, axis)
        C = ((1. - c) / angle).unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(jac[large_angle_inds]) * \
            wedge(axis.squeeze())

        jac[large_angle_inds] = A + B + C

    return jac.squeeze()


def se3_log(mat):
    if mat.dim() < 3:
        mat = mat.unsqueeze(0)

    rot = mat[:, 0:3, 0:3]
    trans = mat[:, 0:3, 3]

    phi = so3_log(rot)
    inv_rot_jac = so3_inv_left_jacobian(phi)

    if inv_rot_jac.dim() < 3:
        inv_rot_jac = inv_rot_jac.unsqueeze(dim=0)

    if trans.dim() < 2:
        trans = trans.unsqueeze(dim=0)

    if inv_rot_jac.dim() < 3:
        inv_rot_jac = inv_rot_jac.unsqueeze(dim=0)

    if trans.dim() < 3:
        trans = trans.unsqueeze(dim=2)

    rho = torch.bmm(inv_rot_jac, trans).squeeze()
    if rho.dim() < 2:
        rho = rho.unsqueeze(dim=0)
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    return torch.cat([rho, phi], dim=1)


def so3_log(mat):
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    phi = mat.new_empty(mat.shape[0], 3, device=mat.device)

    # The cosine of the rotation angle is related to the utils.trace of C
    # Clamp to its proper domain to avoid NaNs from rounding errors
    cos_angle = (0.5 * trace(mat) - 0.5).clamp(-1., 1.)
    angle = cos_angle.acos()

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = isclose(angle, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze(dim=1)

    if len(small_angle_inds) > 0:
        phi[small_angle_inds] = \
            vee(mat[small_angle_inds] -
                     torch.eye(3, dtype=mat.dtype, device=mat.device).expand_as(mat[small_angle_inds]))

    # Otherwise...
    large_angle_mask = ~small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze(dim=1)

    if len(large_angle_inds) > 0:
        angle = angle[large_angle_inds]
        sin_angle = angle.sin()
        phi[large_angle_inds] = \
            vee(
                (0.5 * angle / sin_angle).unsqueeze(dim=1).unsqueeze(dim=1).expand_as(mat[large_angle_inds]) *
                (mat[large_angle_inds] - mat[large_angle_inds].transpose(2, 1)))

    return phi.squeeze()


def so3_inv_left_jacobian(phi):
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    if phi.shape[1] != 3:
        raise ValueError("phi must have shape (3,) or (N,3)")

    jac = phi.new_empty(phi.shape[0], 3, 3, device=phi.device)
    angle = phi.norm(p=2, dim=1)

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = isclose(angle, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze(dim=1)
    if len(small_angle_inds) > 0:
        jac[small_angle_inds] = \
            torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(jac[small_angle_inds]) - \
            0.5 * wedge(phi[small_angle_inds])

    # Otherwise...
    large_angle_mask = ~small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze(dim=1)

    if len(large_angle_inds) > 0:
        angle = angle[large_angle_inds]
        axis = phi[large_angle_inds] / \
            angle.unsqueeze(dim=1).expand(len(angle), 3)

        ha = 0.5 * angle       # half angle
        hacha = ha / ha.tan()  # half angle * cot(half angle)

        exha = ha.unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(jac[large_angle_inds])
        exhacha = hacha.unsqueeze(dim=1).unsqueeze(
            dim=2).expand_as(jac[large_angle_inds])

        A = exhacha * \
            torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(
                dim=0).expand_as(jac[large_angle_inds])
        B = (1. - exhacha) * outer(axis, axis)
        C = -exha * wedge(axis)

        jac[large_angle_inds] = A + B + C

    return jac.squeeze()


# a = torch.tensor([[[-0.9265,  0.3242,  0.1908, -0.5296],
#          [ 0.0157, -0.4732,  0.8808, -0.3322],
#          [ 0.3759,  0.8191,  0.4334, -1.4947]]])
# a = a.repeat(5, 1, 1)
#
# b = torch.autograd.Variable(a, requires_grad=True)
# xi = se3_log(b)
#
# df = torch.tensor([[-1.0000, -1.0000, -1.0000, -0.5000, -1.5000, -2.5000]])
# df = df.view(1, 6).repeat(5, 1)
# c = torch.autograd.Variable(df, requires_grad=True)
# mat = se3_exp(c)
#
# loss = torch.sum(xi)
# torch.autograd.set_detect_anomaly(True)
# loss.backward()
# loss

