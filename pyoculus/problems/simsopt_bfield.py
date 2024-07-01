from .cylindrical_bfield import CylindricalBfield, CylindricalBfieldMap
from .cartesian_bfield import vec2cyl, mat2cyl
from simsopt.field import MagneticField
import numpy as np


class SimsoptBfield(CylindricalBfield):
    """
    Class to set up a Simsopt magnetic field.
    """
    def __init__(self, magnetic_field: MagneticField):
        if not isinstance(magnetic_field, MagneticField):
            raise ValueError("The magnetic_field must be an instance of MagneticField.")
        self._mf = magnetic_field

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

class SimsoptBfieldMap(CylindricalBfieldMap, SimsoptBfield):
    """
    Class to set up a Simsopt magnetic field poincare section map.
    """

    def __init__(self, magnetic_field: MagneticField, phi0=0., R0=None, Z0=None, Nfp=1, **kwargs):
        CylindricalBfield.__init__(self, phi0=phi0, R0=R0, Z0=Z0, Nfp=Nfp, **kwargs)
        SimsoptBfield.__init__(self, magnetic_field)