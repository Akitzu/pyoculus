from .base_map import BaseMap
from pyoculus.integrators import BaseIntegrator, ScipyODEIntegrator


class IntegrationMap(BaseMap):
    """
    Defines the base class for a continous map that needs to be integrated.

    Attributes:
        _integrator (BaseIntegrator): The integrator used to integrate the map.

    Example of Methods:
        ode_rhs: Calculates the right-hand side (RHS) of the ordinary differential equation (ODE).
        ode_rhs_tangent: Calculates the right-hand side (RHS) of the ordinary differential equation (ODE) including the differential of the dependent variables.

        def ode_rhs(self, t, y, *args):
            Calculates the right-hand side (RHS) of the ordinary differential equation (ODE).

            Args:
                t (float): The independent variable in the ODE (usually time).
                y (array): The dependent variable(s) in the ODE.
                *args: Additional parameters for the ODE.

        def ode_rhs_tangent(self, t, ydy, *args):
            Calculates the right-hand side (RHS) of the ordinary differential equation (ODE) including the differential of the dependent variables.

            Args:
                t (float): The independent variable in the ODE (usually time).
                ydy (array): The dependent variable(s) in the ODE, including the differential.
                *args: Additional parameters for the ODE.
    """

    def __init__(self, dim=2, domain=None, integrator=ScipyODEIntegrator, **kwargs):
        """
        Initializes the IntegrationMap object.

        Args:
            integrator (BaseIntegrator): The integrator used to integrate the map, default is RKIntegrator.
            integrator_params (dict): Parameters for the integrator.
        """
        super().__init__(dim, True, domain)

        # Check if the integrator is a derived type of BaseIntegrator
        if not issubclass(integrator, BaseIntegrator):
            raise ValueError(
                "The Integrator is not a derived type of BaseIntegrator class."
            )

        self._integrator = integrator(**kwargs)
