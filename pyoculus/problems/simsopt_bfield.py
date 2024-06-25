from .cylindrical_bfield import CylindricalBfield
from .cartesian_bfield import vec2cyl, mat2cyl
from simsopt.field import MagneticField
import numpy as np


class SimsoptBfieldProblem(CylindricalBfield):
    """
    Class to set up a Simsopt magnetic field problem.
    """

    def __init__(self, magnetic_field: MagneticField, phi0=0., R0=None, Z0=None, Nfp=1, **kwargs):
        if not isinstance(magnetic_field, MagneticField):
            raise ValueError("The magnetic_field must be an instance of MagneticField.")
        self._mf = magnetic_field
        super().__init__(phi0=phi0, R0=R0, Z0=Z0, Nfp=Nfp, **kwargs)

    def B(self, coords, *args):
        coords = np.reshape(coords, (-1, 3))
        self._mf.set_points(coords)
        return vec2cyl(self._mf.B().flatten(), *coords.flatten())

    def dBdX(self, coords, *args):
        B = self.B(coords)
        return B, mat2cyl(self._mf.dB_by_dX().reshape(3, 3), *coords)

    def A(self, coords, *args):
        coords = np.reshape(coords, (-1, 3))
        self._mf.set_points(coords)
        return vec2cyl(self._mf.A().flatten(), *coords.flatten())

    # def B_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
    #     if input1D:
    #         xyz = np.array([x1arr, x2arr, x3arr], dtype=np.float64).T
    #     else:
    #         xyz = np.meshgrid(x1arr, x2arr, x3arr)
    #         xyz = np.array(
    #             [xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()], dtype=np.float64
    #         ).T

    #     xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    #     self._mf.set_points(xyz)

    #     return self._mf.B()

    # def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
    #     B = self.B_many(x1arr, x2arr, x3arr, input1D=input1D)
    #     return [B], self._mf.dB_by_dX()
