from .cylindrical_bfield import CylindricalBfield
import pyoculus.utils.cyl_cart_transform as cct
from typing import Union
import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    from simsopt.field import MagneticField, InterpolatedField, BiotSavart
    from simsopt.geo import SurfaceXYZFourier, SurfaceClassifier
except ImportError as e:
    MagneticField, InterpolatedField, BiotSavart = None, None, None
    SurfaceXYZFourier, SurfaceClassifier = None, None
    logger.debug(str(e))

class SimsoptBfield(CylindricalBfield):
    """
    
    """
    
    def __init__(self, Nfp : int, mf : MagneticField, interpolate: Union[bool, InterpolatedField] = False, **kwargs):
        
        super().__init__(Nfp)

        if not isinstance(mf, MagneticField):
            raise ValueError("mf must be a MagneticField object")

        self._mf = mf
        self._interpolating = False

        if interpolate:
            self._interpolating = True
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
    def from_coils(cls, coils, Nfp, **kwargs):
        mf = BiotSavart(coils)
        return cls(Nfp, mf, **kwargs)

    # Methods of the MagneticField class

    def B(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf_B.set_points(xyz)

        B_cart = self._mf_B.B().flatten()

        return cct.vec_cart2cyl(B_cart, *rphiz)

    def dBdX(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)

        B_cart = self._mf.B()
        dBdX_cart = self._mf.dB_by_dX().reshape(3, 3)
        
        return cct.vec_cart2cyl(B_cart, *rphiz), cct.mat_cart2cyl(dBdX_cart, *rphiz) + cct.dinvJ_matrix(B_cart, *rphiz)

    def A(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)

        A_cart = self._mf.A().flatten()

        return cct.vec_cart2cyl(A_cart, *rphiz)


    # def B_many(self, x1arr, x2arr, x3arr, input1D=True):
    #     if input1D:
    #         xyz = np.array([x1arr, x2arr, x3arr], dtype=np.float64).T
    #     else:
    #         xyz = np.meshgrid(x1arr, x2arr, x3arr)
    #         xyz = np.array(
    #             [xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()], dtype=np.float64
    #         ).T

    #     xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    #     self._mf_B.set_points(xyz)

    #     return self._mf_B.B()    

def surf_from_coils(coils, **kwargs):
    logger.info(f"Using surf_from_coils with parameters: {kwargs}")
    logger.warning("Using surf_from_coild can result in weird surfaces. Use with caution.")
    
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