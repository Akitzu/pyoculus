from pyoculus.problems import CartesianBfield
from simsopt.field import MagneticField, InterpolatedField, BiotSavart
from simsopt.geo import SurfaceXYZFourier, SurfaceClassifier
import numpy as np
from typing import Union

### Simsopt magnetic field problem class ###
class SimsoptBfieldProblem(CartesianBfield):
    def __init__(self, R0, Z0, Nfp, mf, interpolate: Union[bool, InterpolatedField] = False, **kwargs):
        super().__init__(R0, Z0, Nfp)

        if not isinstance(mf, MagneticField):
            raise ValueError("mf must be a MagneticField object")

        self._mf = mf

        if interpolate:
            self.interpolating = True
            if isinstance(interpolate, InterpolatedField):
                self._mf_B = interpolate
            else:
                surf = kwargs.get('surf', None)
                if surf is None:
                    surf = surf_from_coils(mf.coils, **kwargs)

                p = kwargs.get('p', 2)
                h = kwargs.get('h', 0.03)
                self.surfclassifier = SurfaceClassifier(surf, h=h, p=p)

                deltah = kwargs.get('deltah', 0.05)
                def skip(rs, phis, zs):
                    rphiz = np.asarray([rs, phis, zs]).T.copy()
                    dists = self.surfclassifier.evaluate_rphiz(rphiz)
                    skip = list((dists < -deltah).flatten())
                    return skip

                n = kwargs.get('n', 20)
                rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
                zs = surf.gamma()[:, :, 2]
                rrange = (np.min(rs), np.max(rs), n)
                phirange = (0, 2 * np.pi / Nfp, n * 2)
                zrange = (0, np.max(zs), n // 2)

                degree = kwargs.get('degree', 3)
                stellsym = kwargs.get('stellsym', True)
                skyping = kwargs.get('skyping', skip)
                
                self._mf_B = InterpolatedField(
                    mf,
                    degree,
                    rrange,
                    phirange,
                    zrange,
                    True,
                    nfp=Nfp,
                    stellsym=stellsym,
                    skip=skyping,
                )
        else:
            self.interpolating = False
            self._mf_B = mf

    @classmethod
    def from_coils(cls, R0, Z0, Nfp, coils, **kwargs):
        mf = BiotSavart(coils)
        return cls(R0, Z0, Nfp, mf, **kwargs)

    @classmethod
    def without_axis(
        cls, guess, Nfp, mf, interpolate = False, **kwargs
    ):
        instance = cls(guess[0], guess[1], Nfp, mf, **kwargs)
        R0, Z0 = instance.find_axis(guess, **kwargs)
        return cls(R0, Z0, Nfp, mf, interpolate, **kwargs)

    # The return of the B field for the two following methods is not the same as the calls are :
    #   - CartesianBfield.f_RZ which does :
    #   line 37     B = np.array([self.B(xyz, *args)]).T
    #   - CartesianBfield.f_RZ_tangent which does :
    #   line 68     B, dBdX = self.dBdX(xyz, *args)
    #   line 69     B = np.array(B).T
    # and both should result in a (3,1) array
    def B(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self._mf_B.set_points(xyz)
        return self._mf_B.B().flatten()

    def dBdX(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)
        
        return [self._mf.B().flatten()], self._mf.dB_by_dX().reshape(3, 3)

    def B_many(self, x1arr, x2arr, x3arr, input1D=True):
        if input1D:
            xyz = np.array([x1arr, x2arr, x3arr], dtype=np.float64).T
        else:
            xyz = np.meshgrid(x1arr, x2arr, x3arr)
            xyz = np.array(
                [xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()], dtype=np.float64
            ).T

        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        self._mf_B.set_points(xyz)

        return self._mf_B.B()

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True):
        B = self.B_many(x1arr, x2arr, x3arr, input1D=input1D)
        return [B], self._mf.dB_by_dX()
    
    def A(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)
        return self._mf.A().flatten()


def surf_from_coils(coils, **kwargs):
    print(kwargs)
    mpol = kwargs.get('mpol', 3)
    ntor = kwargs.get('ntor', 3)
    stellsym = kwargs.get('stellsym', False)
    nfp = kwargs.get('nfp', 1)

    ncoils = kwargs.get('ncoils', None)
    
    nphi, ntheta = len(coils), len(coils[0].curve.gamma())
    qpts_theta = np.linspace(0, 1, ntheta, endpoint=False)
    qpts_phi = np.linspace(0, 1, nphi, endpoint=False)

    surf = SurfaceXYZFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=qpts_phi,
        quadpoints_theta=qpts_theta
    )
    centroids = np.array([np.mean(coil.curve.gamma(), axis=0) for coil in coils])

    phis = np.arctan2(centroids[:, 1], centroids[:, 0])
    indices = np.argsort(phis)
    gamma_curves = [coils[i].curve.gamma() for i in indices]
    if ncoils is not None:
        gamma_curves = np.stack([gamma if (i // ncoils) % 2 != 0 else gamma[::-1] for i, gamma in enumerate(gamma_curves)])
    surf.least_squares_fit(gamma_curves)

    return surf