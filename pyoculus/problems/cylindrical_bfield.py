## @file cylindrical_bfield.py
#  @brief containing a class for pyoculus ODE solver that deals with magnetic field given in Cylindrical coordinates
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .cylindrical_problem import CylindricalProblem
from .bfield_problem import BfieldProblem
from ..solvers.fixed_point import FixedPoint
import numpy as np
import copy

## Class that used to setup the cylindrical bfield problem used in ODE solver.
#
# The system of ODEs is given by
# \f[ \frac{dR}{d\phi} = \frac{B^{R}}{B^{\phi}}  \f]
# \f[ \frac{dZ}{d\phi} = \frac{B^{Z}}{B^{\phi}}  \f]
class CylindricalBfield(CylindricalProblem, BfieldProblem):
    def __init__(self, R0, Z0, Nfp=1):
        """! Set up the problem
        @param R0 the R coordinate of the magnetic axis
        @param Z0 the Z coordinate of the magnetic axis
        """

        super().__init__(R0, Z0, Nfp)

    @classmethod
    def find_axis(cls, guess, Nfp = 1, **kwargs):
        """! Find the magnetic axis
        @param guess the initial guess of the axis
        @param Nfp the number of field periods
        @param **kwargs extra parameters for the FixedPoint.find_axis method
        @returns the axis R0 and Z0
        """
        
        tmpProblem = copy.deepcopy(cls)
        tmpProblem._R0 = guess[0]
        tmpProblem._Z0 = guess[1]
        tmpProblem.Nfp = Nfp

        R0, _, Z0 = FixedPoint.find_axis(tmpProblem, guess, **kwargs)
        
        return R0, Z0        

    def f_RZ(self, phi, RZ, *args):
        """! Returns ODE RHS
        @param phi cylindrical angle in ODE
        @param RZ \f$(R, Z)\f$ in ODE
        @param *args parameter for the ODE
        @returns the RHS of the ODE
        """

        R = RZ[0]
        Z = RZ[1]

        RphiZ = np.array([R, phi, Z])

        Bfield = self.B(RphiZ, *args)

        dRdt = Bfield[0] / Bfield[1]
        dZdt = Bfield[2] / Bfield[1]

        return np.array([dRdt, dZdt])

    def f_RZ_tangent(self, phi, RZ, *args):
        """! Returns ODE RHS, with tangent
        @param zeta cylindrical angle in ODE
        @param RZ \f$(R, Z, dR_1, dZ_1, dR_2, dZ_2)\f$ in ODE
        @param *args extra parameters for the ODE
        @returns the RHS of the ODE, with tangent
        """
        R = RZ[0]
        Z = RZ[1]

        dRZ = np.array([[RZ[2], RZ[4]], [RZ[3], RZ[5]]], dtype=np.float64)
        M = np.zeros([2, 2], dtype=np.float64)

        rphiz = np.array([R, phi, Z])

        B = np.array([self.B(rphiz, *args)]).T
        dBdRphiZ = np.array(self.dBdX(rphiz, *args))

        dRphiZ = B.T[0]

        dRdt = dRphiZ[0]/dRphiZ[1]
        dZdt = dRphiZ[2]/dRphiZ[1]

        M[0,0] = dBdRphiZ[0,0] / dRphiZ[1]  - dRphiZ[0] / dRphiZ[1]**2 * dBdRphiZ[1,0]
        M[0,1] = dBdRphiZ[0,2] / dRphiZ[1]  - dRphiZ[0] / dRphiZ[1]**2 * dBdRphiZ[1,2]
        M[1,0] = dBdRphiZ[2,0] / dRphiZ[1]  - dRphiZ[2] / dRphiZ[1]**2 * dBdRphiZ[1,0]
        M[1,1] = dBdRphiZ[2,2] / dRphiZ[1]  - dRphiZ[2] / dRphiZ[1]**2 * dBdRphiZ[1,2]

        dRZ = np.matmul(M, dRZ)

        return np.array([dRdt, dZdt, dRZ[0, 0], dRZ[1, 0], dRZ[0, 1], dRZ[1, 1]])