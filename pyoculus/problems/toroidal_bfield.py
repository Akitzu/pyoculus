## @file toroidal_bfield.py
#  @brief containing a problem class with magnetic fields in two cyclical coordinates for pyoculus ODE solver
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from overrides import overrides
from .integration_map import IntegrationMap
from .bfield_problem import BfieldProblem
import numpy as np


class ToroidalBfield(IntegrationMap, BfieldProblem):
    """
    Class that sets up a Map given by following the a magnetic field in toroidal system :math:`(s, \\theta, \\zeta)`.
    """

    def __init__(self, phi0=0., Nfp=1, **kwargs):
        super().__init__(dim=2, **kwargs)
        self.phi0 = phi0
        self.Nfp = Nfp

    ## BaseMap methods

    @overrides
    def f(self, t, y0):
        self._integrator.set_rhs(self._ode_rhs)
        return self._integrate(t, y0)

    @overrides
    def df(self, t, y0):
        self._integrator.set_rhs(self._ode_rhs_tangent)
        return self._integrate(t, y0)

    @overrides
    def lagrangian(self, y0, t):
        self._integrator.set_rhs(self._ode_rhs_A)
        return self._integrate(t, y0)

    ## Integration methods

    def _integrate(self, t, y0):
        """
        Integrates the ODE for a number of periods.
        """
        dphi = t * 2 * np.pi / self.Nfp
        y = np.array(y0)
        self._integrator.set_initial_value(self.phi0, y)
        return self._integrator.integrate(self.phi0 + dphi)

    def _ode_rhs(self, phi, st, *args):
        """
        Calculates the right-hand side (RHS) of the ODE.

        Args:
            phi (float): The current cylindrical angle.
            st (array): The cylindrical coordinates :math:`(s, \\theta)` in the ODE.
            *args: Additional parameters for the ODE.

        Returns:
            array: The RHS of the ODE.
        """
        stz = np.array([st[0], st[1], phi])
        B = self.B(stz, *args)
        return np.array([B[0] / B[2], B[1] / B[2]])

    def _ode_rhs_tangent(self, phi, y, *args):
        """
        Calculates the right-hand side (RHS) of the ODE with differential of the dependent variables.

        Args:
            phi (float): The current cylindrical angle.
            st (array): The cylindrical coordinates :math:`(s, \\theta, ds_1, d\\theta_1, ds_2, d\\theta_2))` in the ODE.
            *args: Additional parameters for the ODE.

        Returns:
            array: The RHS of the ODE.
        """
        stz = np.array([y[0], y[1], phi])
        Bu, dBu = self.dBdX(stz, *args)

        deltax = np.reshape(y[2:], [2, 2])
        gBzeta = Bu[2]

        M = dBu[0:2, 0:2] * gBzeta - dBu[0:2, 2, np.newaxis] * Bu[0:2]

        deltax = deltax @ M / gBzeta**2

        df = np.zeros([6])
        df[0:2] = Bu[0:2] / Bu[2]
        df[2:6] = deltax.flatten()

        return df

    def _ode_rhs_A():
        raise NotImplementedError("A is not implemented.")