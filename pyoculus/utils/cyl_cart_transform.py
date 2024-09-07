import numpy as np

# Jacobian of the map (x, y, z) -> (r, phi, z) at (r, phi, z) and (x, y, z)

def xyz_jac(r, phi, z):
    """
    Jacobian of the map :math:`(x, y, z) \\to (r, \\phi, z)` at :math:`(r, \\phi, z)`

    .. math::
        J(r, \\phi, z) = \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
    """
    return np.array(
        [
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi) / r, np.cos(phi) / r, 0],
            [0, 0, 1],
        ]
    )

def rphiz_jac(x, y, z):
    """
    Jacobian of the map :math:`(x, y, z) \to (r, \phi, z)` at :math:`(x, y, z)`
    """
    return np.array(
        [
            [x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0],
            [-y / (x**2 + y**2), x / (x**2 + y**2), 0],
            [0, 0, 1],
        ]
    )

# Jacobian of the map (r, phi, z) -> (x, y, z) at (r, phi, z) and (x, y, z)

def xyz_inv_jac(r, phi, z):
    """
    Inverse Jacobian of the map (r, phi, z) -> (x, y, z) at (r, phi, z)
    """
    return np.array(
        [
            [np.cos(phi), -np.sin(phi) / r, 0],
            [np.sin(phi), np.cos(phi) / r, 0],
            [0, 0, 1],
        ]
    )


def rphiz_inv_jac(x, y, z):
    """
    Inverse Jacobian of the map (r, phi, z) -> (x, y, z)s at (x, y, z)
    """
    return np.array(
        [
            [x / np.sqrt(x**2 + y**2), -y / (x**2 + y**2), 0],
            [y / np.sqrt(x**2 + y**2), x / (x**2 + y**2), 0],
            [0, 0, 1],
        ]
    )

# Coordinate transformations

def xyz(r, phi, z):
    return np.array([r * np.cos(phi), r * np.sin(phi), z])

def rphiz(x, y, z):
    return np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x), z])

# Vector transformations

def vec_cart2cyl(vec, r, phi, z):
    """
    Transforms a vector in cartesian coordinates to cylindrical coordinates at (r, phi, z).

    .. math:: 
        \\begin{bmatrix}
            v^r \\\\
            v^{\\phi} \\\\
            v^z
        \\end{bmatrix} =
        \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            v^x \\\\
            v^y \\\\
            v^z
        \\end{bmatrix}
    """
    return np.matmul(xyz_jac(r, phi, z), np.atleast_2d(vec).T).T[0]

def vec_cyl2cart(vec, x, y, z):
    pass

# Matrix transformations

def mat_cart2cyl(mat, r, phi, z):
    """
    Transforms a matrix :math:`A` from cartesian coordinates to cylindrical coordinates at :math:`(r, \\phi, z)`.

    .. math:: 
        \\begin{bmatrix}
            A^r_r & A^r_\\phi & A^r_z \\\\
            A^\\phi_r & A^\\phi_\\phi & A^\\phi_z \\\\
            A^z_r & A^z_\\phi & A^z_z \\\\
        \\end{bmatrix} =
        \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            A^x_x & A^x_y & A^x_z \\\\
            A^y_x & A^y_y & A^y_z \\\\
            A^z_x & A^z_y & A^z_z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            \\partial_r x & \\partial_\\phi x & \\partial_z x \\\\
            \\partial_r y & \\partial_\\phi y & \\partial_z y \\\\
            \\partial_r z & \\partial_\\phi z & \\partial_z z \\\\
        \\end{bmatrix}
    """

    invjac = xyz_inv_jac(r, phi, z)
    jac = xyz_jac(r, phi, z)
    return np.matmul(np.matmul(jac, mat), invjac)

def mat_cyl2cart(mat, x, y, z):
    pass