## @file cartesian_bfield.py
#  @brief containing a class for pyoculus ODE solver that deals with magnetic field given in Cartesian
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .cylindrical_problem import CylindricalProblem
from .bfield_problem import BfieldProblem
import numpy as np

class CartesianBfield(CylindricalProblem, BfieldProblem):

    def __init__(self, R0, Z0, Nfp=1):
        """! Set up the problem
        @param R0 the R coordinate of the magnetic axis
        @param Z0 the Z coordinate of the magnetic axis
        """

        super().__init__(R0, Z0, Nfp)

    def find_axis(self, guess, Nfp = 1, params=dict(), integrator=None, integrator_params=dict(), **kwargs):
            """! Find the magnetic axis
            @param guess the initial guess of the axis
            @param Nfp the number of field periods
            @param **kwargs extra parameters for the FixedPoint.find_axis method
            @returns the axis R0 and Z0
            """
            options = {
                "Rbegin": guess[0]-1, "Rend": guess[0]+1, "niter": 100, "tol": 1e-9
            }
            options.update(kwargs)

            self._R0 = guess[0]
            self._Z0 = guess[1]
            
            from ..solvers.fixed_point import FixedPoint
            fpaxis = FixedPoint(self, params=params, integrator=integrator, integrator_params=integrator_params, evolve_axis=False)
            RZ_axis = fpaxis.find_axis(R_guess = guess[0], Z_guess = guess[1], **options)

            if RZ_axis is None:
                raise ValueError("Failed to find the axis")
            
            return RZ_axis[0], RZ_axis[1]

    def f_RZ(self, phi, RZ, *args):
        """! Returns ODE RHS
        @param phi cylindrical angle in ODE
        @param RZ \f$(R, Z)\f$ in ODE
        @param *args parameter for the ODE
        @returns the RHS of the ODE
        """

        R = RZ[0]
        Z = RZ[1]

        xyz = np.array([
            R * np.cos(phi),
            R * np.sin(phi),
            Z
        ])
        
        B = np.array([self.B(xyz, *args)]).T

        invJacobian = self._inv_Jacobian(R,phi,Z)

        dRphiZ = np.matmul(invJacobian, B).T[0]
        dRdt = dRphiZ[0]/dRphiZ[1]
        dZdt = dRphiZ[2]/dRphiZ[1]

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
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        dRZ = np.array([[RZ[2], RZ[4]], [RZ[3], RZ[5]]], dtype=np.float64)
        M = np.zeros([2, 2], dtype=np.float64)

        xyz = np.array([
            R * cosphi,
            R * sinphi,
            Z
        ])
        
        B, dBdX = self.dBdX(xyz, *args)
        B = np.array(B).T
        Bx = B[0,0]
        By = B[1,0]
        dBdX = np.array(dBdX)

        invJacobian = self._inv_Jacobian(R,phi,Z)
        Jacobian = np.linalg.inv(invJacobian)

        dRphiZ = np.matmul(invJacobian, B).T[0]

        dRdt = dRphiZ[0]/dRphiZ[1]
        dZdt = dRphiZ[2]/dRphiZ[1]

        # convert from @(Bx,By,Bz)/@(x,y,z) to @(B^R,B^phi,B^Z)/@(R,phi,Z)
        dBdRphiZ = np.matmul(np.matmul(invJacobian, dBdX), Jacobian)
        dinvJ = np.array([
            [0, -Bx * sinphi + By * cosphi, 0 ],
            [Bx * sinphi / R**2 - By * cosphi / R**2, -Bx * cosphi / R - By * sinphi / R, 0],
            [0, 0, 0]
        ]) 
        dBdRphiZ = dBdRphiZ + dinvJ

        M[0,0] = dBdRphiZ[0,0] / dRphiZ[1]  - dRphiZ[0] / dRphiZ[1]**2 * dBdRphiZ[1,0]
        M[0,1] = dBdRphiZ[0,2] / dRphiZ[1]  - dRphiZ[0] / dRphiZ[1]**2 * dBdRphiZ[1,2]
        M[1,0] = dBdRphiZ[2,0] / dRphiZ[1]  - dRphiZ[2] / dRphiZ[1]**2 * dBdRphiZ[1,0]
        M[1,1] = dBdRphiZ[2,2] / dRphiZ[1]  - dRphiZ[2] / dRphiZ[1]**2 * dBdRphiZ[1,2]

        dRZ = np.matmul(M, dRZ)

        return np.array([dRdt, dZdt, dRZ[0, 0], dRZ[1, 0], dRZ[0, 1], dRZ[1, 1]])

    def f_RZ_A(self, phi, y, *args):
        """! Returns ODE RHS, with the integral of A
        @param phi cylindrical angle in ODE
        @param y \f$(R, Z, \int\vect{A}\cdot\vect{dl})\f$ in ODE
        @param *args extra parameters for the ODE
        @returns the RHS of the ODE, with the integral of A
        """
        R = y[0]
        Z = y[1]
        
        xyz = np.array([
                    R * np.cos(phi),
                    R * np.sin(phi),
                    Z
                ])
                
        B = np.array([self.B(xyz, *args)]).T

        invJacobian = self._inv_Jacobian(R,phi,Z)

        dRphiZ = np.matmul(invJacobian, B).T[0]
        dRdt = dRphiZ[0]/dRphiZ[1]
        dZdt = dRphiZ[2]/dRphiZ[1]

        # Integral of A, step
        A = np.array([self.A(xyz, *args)]).T
        Acyl = np.matmul(invJacobian, A).T[0]

        dl = np.array([dRdt, 1, dZdt])
        dl = np.array([1, R**2, 1])*dl
        
        dintegralAdphi = np.dot(Acyl, dl)

        return np.array([dRdt, dZdt, dintegralAdphi])

    @staticmethod
    def _inv_Jacobian(R, phi, Z):
        return np.array([
            [np.cos(phi), np.sin(phi), 0], 
            [-np.sin(phi)/R, np.cos(phi)/R, 0], 
            [0,0,1]
            ])