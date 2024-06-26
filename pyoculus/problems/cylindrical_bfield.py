from .base_map import BaseMap
from .integrated_problem import IntegratedProblem
from .bfield_problem import BfieldProblem
from ..solvers.fixed_point import FixedPoint
import numpy as np


class CylindricalBfield(IntegratedProblem, BfieldProblem):
    """
    Cylindrical magnetic field problem.
    """
    pass


class CylindricalBfieldMap(BaseMap, CylindricalBfield):
    """
    Class that sets up a Map given by following the a magnetic field in cylindrical system :math:`(R, \\phi, Z)`.

    Attributes:
        phi0 (float): The cylindrical angle from which to start following the field.
        R0 (float): The major radius of the magnetic axis in the phi0 plane.
        Z0 (float): The vertical position of the magnetic axis in the phi0 plane.
        Nfp (int): The number of field periods, default is 1. Gives the periodicity of the magnetic field (:math:`T = 2*\\pi/n_\\text{fp}`).
    """

    def __init__(self, phi0=0., R0=None, Z0=None, Nfp=1, finderargs=dict(), **kwargs):
        """
        Initializes the CylindricalBfield object and calls the IntegrationMap constructor. If R0 or Z0 is not provided, the magnetic axis will be found using a FixedPoint solver.
        """
        super().__init__(dim=2, **kwargs)

        self.phi0 = phi0
        self.Nfp = Nfp

        if R0 is None or Z0 is None:
            self.find_axis(finderargs)
        else:
            self.R0 = R0
            self.Z0 = Z0

    def find_axis(self, guess = None, **kwargs):
        """
        Finds the magnetic axis using a FixedPoint solver.
        """
        axisfinder = FixedPoint(self)
        axisfinder.find(1, guess, **kwargs)
        if axisfinder.is_successful():
            self.R0, self.Z0 = axisfinder.coords[0]
        else:
            raise ValueError("The magnetic axis could not be found.")

    ## BaseMap methods

    def f(self, t, y0):
        self._integrator.set_rhs(self._rhs_RZ)
        return self._integrate(t, y0)

    def df(self, t, y0):
        ic = np.array([*y0, 1., 0., 0., 1.])
        self._integrator.set_rhs(self._rhs_RZ_tangent)
        output = self._integrate(t, ic)
        return output[2:6].reshape(2, 2).T

    def lagrangian(self, y0, t):
        ic = np.array([*y0, 0.])
        self._integrator.set_rhs(self._rhs_RZ_A)
        output = self._integrate(t, ic)
        return output[2]

    def f_winding(self, t, y0, y1 = None):
        if y1 is None:
            y1 = np.array([self.R0, self.Z0])

        theta0 = np.arctan2(y0[1] - y1[1], y0[0] - y1[0])
        self._integrator.set_rhs(self._ode_rhs)
        ic = np.array([*y0, *y1, theta0, 1., 0., 0., 1.])
        output = self._integrate(t, ic)
        theta1 = np.arctan2(output[1] - output[3], output[0] - output[2])
        rho = np.sqrt((output[0] - output[2])**2 + (output[1] - output[3])**2)

        return np.array([rho, theta1 - theta0])
    
    # def df_winding(self, t, y0, y1 = None):
    #     if y1 is None:
    #         y1 = np.array([self.R0, self.Z0])

    #     theta0 = np.arctan2(y0[1] - y1[1], y0[0] - y1[0])
    #     self._integrator.set_rhs(self._ode_rhs_tangent)
    #     ic = np.array([*y0, *y1, theta0, 1., 0., 0., 1.])
    #     output = self._integrate(t, ic)

    #     theta1 = np.arctan2(output[1] - output[3], output[0] - output[2])
    #     rho = np.sqrt((output[0] - output[2])**2 + (output[1] - output[3])**2)
        
    #     dG = np.array([
    #             [output[5], output[7]],
    #             [output[6], output[8]]
    #         ], dtype=np.float64)

    #     # dH = dH(G(R,Z))
    #     deltaRZ = RZ_evolved - RZ_Axis
    #     dH = np.array([
    #             np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
    #             np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
    #     ], dtype=np.float64)

    #     # dP = dH(R,Z)
    #     deltaRZ = RZ - RZ_Axis
    #     dP = np.array([
    #         np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
    #         np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
    #     ], dtype=np.float64)
                
    #     # Jacobian of the map F = H(G(R,Z)) - H(R,Z)
    #     return dH @ dG - dP

    ## Integration methods

    def _integrate(self, t, y0):
        """
        Integrates the ODE for a number of periods.
        """
        dphi = t * 2 * np.pi / self.Nfp
        y = np.array(y0)
        self._integrator.set_initial_value(self.phi0, y)
        return self._integrator.integrate(self.phi0 + dphi)

    def _ode_rhs(self, phi, y, *args):
        """
        Returns the right-hand side (RHS) of ODE.

        Args:
            phi (float): The cylindrical angle in the ODE.
            y (array): The cylindrical coordinates :math:`(R, Z, R_0, Z_0, \theta)` in the ODE.
            *args: Additional parameters for the magnetic field calculation.

        Returns:
            array: The RHS of the ODE.
        """
        R, Z, R0, Z0 = y[:4]
        dRZ = self._rhs_RZ(phi, np.array([R, Z]), *args)
        dRZ0 = self._rhs_RZ(phi, np.array([R0, Z0]), *args)

        # Calculating the change for the angle theta (poloidal angle with origin at the magnetic axis)
        deltaR = R - R0
        deltaZ = Z - Z0
        # dartan2(Z-Z0, R-R0)/d(R-R0) * d(R-R0)/dphi + dartan2(Z-Z0, R-R0)/d(Z-Z0) * d(Z-Z0)/dphi
        dtheta = (deltaR * (dRZ[1] - dRZ0[1]) - deltaZ * (dRZ[0] - dRZ0[0])) / (
            deltaR**2 + deltaZ**2
        )

        return np.array([*dRZ, *dRZ0, dtheta])

    def _rhs_RZ(self, phi, RZ, *args):
        """
        Calculates the right-hand side (RHS) of ODE following the magnetic field.

        Args:
            phi (float): The current cylindrical angle.
            RZ (array): The current R,Z coordinates, RZ = [R, Z].
            *args: Additional parameters for the ODE.

        Returns:
            array: An array containing the derivatives of R and Z with respect to phi, i.e., [dR/dphi, dZ/dphi].
        """
        RphiZ = np.array([RZ[0], phi, RZ[1]])

        Bfield = self.B(RphiZ, *args)

        # R, Z evolution given by following the field
        # dR/dphi = B^R / B^phi and dZ/dphi = B^Z / B^phi
        dRdphi = Bfield[0] / Bfield[1]
        dZdphi = Bfield[2] / Bfield[1]

        return np.array([dRdphi, dZdphi])

    # Tangent ODE RHS

    def _ode_rhs_tangent(self, phi, y, *args):
        """
        Args:
            phi (float): The current cylindrical angle.
            y (array): The current R, Z, R0, Z0, theta, dR1, dZ1, dR2, dZ2.
        """
        R, Z, R0, Z0, theta, *dRZ = y

        dy1 = self._rhs_RZ_tangent(phi, np.array([R, Z, dRZ]))
        dy2 = self._ode_rhs(phi, np.array([R, Z, R0, Z0, theta]))

        return np.concatenate([dy1, dy2])

    def _rhs_RZ_tangent(self, phi, y, *args):
        """
        Returns the right-hand side (RHS) of the ODE with the calculation of the differential evolution. For an explanation, one could look at [S.R. Hudson](https://w3.pppl.gov/~shudson/Oculus/ga00aa.pdf).

        Args:
            phi (float): The current cylindrical angle.
            y (array): The current R, Z and dRZ. y = [dR/dphi, dZ/dphi, dR/dR_i, dZ/dR_i, dR/dZ_i, dZ/dZ_i] where i stands for the initial point in the phi0 plane (not the axis).
            *args: Additional parameters for the magnetic field calculation.
        Returns:
            array: The RHS of the ODE, with tangent.
        """
        R, Z, *dRZ = y
        # dRZ = [[y[2], y[4]], [y[3], y[5]]]
        dRZ = np.array(dRZ, dtype=np.float64).reshape(2, 2).T
        M = np.zeros([2, 2], dtype=np.float64)

        rphiz = np.array([R, phi, Z])

        Bfield, dBdRphiZ = self.dBdX(rphiz, *args)
        # Bfield is tansformed from [array] into array
        Bfield = np.array(Bfield[0], dtype=np.float64)
        dBdRphiZ = np.array(dBdRphiZ, dtype=np.float64)

        # R, Z evolution as in _rhs_RZ
        dRdphi = Bfield[0] / Bfield[1]
        dZdphi = Bfield[2] / Bfield[1]

        # Matrix of the derivatives of (B^R/B^phi, B^Z/B^phi) with respect to (R, Z)
        M[0, 0] = (
            dBdRphiZ[0, 0] / Bfield[1] - Bfield[0] / Bfield[1] ** 2 * dBdRphiZ[1, 0]
        )
        M[0, 1] = (
            dBdRphiZ[0, 2] / Bfield[1] - Bfield[0] / Bfield[1] ** 2 * dBdRphiZ[1, 2]
        )
        M[1, 0] = (
            dBdRphiZ[2, 0] / Bfield[1] - Bfield[2] / Bfield[1] ** 2 * dBdRphiZ[1, 0]
        )
        M[1, 1] = (
            dBdRphiZ[2, 2] / Bfield[1] - Bfield[2] / Bfield[1] ** 2 * dBdRphiZ[1, 2]
        )

        dRZ = np.matmul(M, dRZ)

        return np.array([dRdphi, dZdphi, dRZ[0, 0], dRZ[1, 0], dRZ[0, 1], dRZ[1, 1]])

    # dLangrangian ODE RHS

    def _rhs_RZ_A(self, phi, y, *args):
        """
        Returns RHS of the ODE for the integral of the vector potential along the field line.

        Args:
            phi (float): The current cylindrical angle.
            y (array): The current R, Z, integral of A.
            *args: Additional parameters to calculate the magnetic field.
        """

        # R, Z evolution
        dRdphi, dZdphi = self._rhs_RZ(phi, y[:2], *args)

        # magnetic potential at the current point
        RphiZ = np.array([y[0], phi, y[1]])
        A = self.A(RphiZ, *args)

        # Integral of A, step
        dl = np.array([dRdphi, 1, dZdphi])
        dl = np.array([1, y[0] ** 2, 1]) * dl

        dintegralAdphi = np.dot(A, dl)

        return np.array([dRdphi, dZdphi, dintegralAdphi])
