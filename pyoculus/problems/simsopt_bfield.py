from pyoculus.problems import CartesianBfield
from simsopt.field import MagneticField
import numpy as np

### Simsopt magnetic field problem class ###
class SimsoptBfieldProblem(CartesianBfield):
    def __init__(self, R0, Z0, Nfp, bs):
        super().__init__(R0, Z0, Nfp)

        if not isinstance(bs, MagneticField):
            raise ValueError("bs must be a MagneticField object")

        self.bs = bs

    # The return of the B field for the two following methods is not the same as the calls are :
    #   - CartesianBfield.f_RZ which does :
    #   line 37     B = np.array([self.B(xyz, *args)]).T
    #   - CartesianBfield.f_RZ_tangent which does :
    #   line 68     B, dBdX = self.dBdX(xyz, *args)
    #   line 69     B = np.array(B).T
    # and both should result in a (3,1) array
    def B(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self.bs.set_points(xyz)
        return self.bs.B().flatten()

    def dBdX(self, xyz):
        B = self.B(xyz)
        return [B], self.bs.dB_by_dX().reshape(3, 3)

    def B_many(self, x1arr, x2arr, x3arr, input1D=True):
        if input1D:
            xyz = np.array([x1arr, x2arr, x3arr], dtype=np.float64).T
        else:
            xyz = np.meshgrid(x1arr, x2arr, x3arr)
            xyz = np.array(
                [xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()], dtype=np.float64
            ).T

        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        self.bs.set_points(xyz)

        return self.bs.B()

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True):
        B = self.B_many(x1arr, x2arr, x3arr, input1D=input1D)
        return [B], self.bs.dB_by_dX()