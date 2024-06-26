## @file base_integrator.py
#  @brief Contains base class of ODE integrator
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#
from abc import ABC, abstractmethod
## Class that used to setup the ODE integrator.
#
# This is an abstract class, should never be used as an instance.
#
# All integrators derived from BaseIntegrator should contain the following member functions
#
#   - set_initial_value -- Set up initial value for the ODE solver
#   - integrate -- Solve the ODE until a given time
#   .
#
# Optional:
#   - copy -- make a copy of the integrator as a new instance
class BaseIntegrator(ABC):
    """
    Abstract base class for ODE integrators.

    It can be a wrapper of any existing ODE solver, such as scipy.integrate.ode or scipy.integrate.solve_ivp, or a new solver implemented from scratch.
    """

    def __init__(self, params):
        """
        Set up the ODE solver.

        Args:
            params (dict): The parameters used in the ODE solver.
        """
        self._params = dict(params)

    def set_initial_value(self, t, x):
        """
        Set up the initial value for the ODE solver.

        Args:
            t: The start of time.
            x: The start of coordinates.
        """
        self.t = t
        self.x = x

    @abstractmethod
    def integrate(self, tend):
        """
        Integrate the ODE until :math:`t_\\text{end}`.

        Args:
            tend: The target end time.

        Returns:
            The new value of x.
        """
        raise NotImplementedError("ERROR: Integrator has to implement integrate")

    def get_solution(self):
        """
        Get the solution at current time.

        Returns:
            The solution.
        """
        return self.x

    def __copy__(self):
        """
        Return a copy of self, to use if want to compute in parallel.

        Returns:
            A copy of self.
        """
        raise NotImplementedError("ERROR: Integrator has to implement copy")
