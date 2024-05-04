from .cylindrical_bfield import CylindricalBfield
from functools import wraps
import numpy as np


# Decorators to convert from Cartesian to cylindrical coordinates
def mat2cyl(func):
    @wraps(func)
    def wrapper(self, RphiZ, *args, **kwargs):
        return mat2cyl(func(self, xyz(*RphiZ), *args, **kwargs), *RphiZ)

    return wrapper


def vec2cyl(func):
    @wraps(func)
    def wrapper(self, RphiZ, *args, **kwargs):
        return vec2cyl(func(self, xyz(*RphiZ), *args, **kwargs), *RphiZ)

    return wrapper


def inv_Jacobian(R, phi, _):
    return np.array(
        [
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi) / R, np.cos(phi) / R, 0],
            [0, 0, 1],
        ]
    )


def Jacobian(R, phi, Z):
    return np.linalg.inv(inv_Jacobian(R, phi, Z))


def xyz(R, phi, Z):
    return np.array([R * np.cos(phi), R * np.sin(phi), Z])


def vec2cyl(vec, R, phi, Z):
    return np.matmul(inv_Jacobian(R, phi, Z), np.atleast_2d(vec).T).T[0]


def mat2cyl(mat, R, phi, Z):
    invjac = inv_Jacobian(R, phi, Z)
    jac = np.linalg.inv(invjac)
    return np.matmul(np.matmul(invjac, mat), jac)
