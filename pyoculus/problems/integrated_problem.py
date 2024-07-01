from pyoculus.integrators import BaseIntegrator, ScipyODEIntegrator
from abc import ABC

class IntegratedProblem(ABC):
    """
    Defines the base class for a continous map that needs to be integrated.

    Attributes:
        _integrator (BaseIntegrator): The integrator used to integrate the map.
   
    Example of Methods:
        - `ode_rhs`: Calculates the right-hand side (RHS) of the ordinary differential equation (ODE).
        - `ode_rhs_tangent`: Calculates the right-hand side (RHS) of the ODE including the differential of the dependent variables.
    """

    def __init__(self, dim=2, domain=None, integrator=ScipyODEIntegrator, **kwargs):
        """
        Initializes the IntegratedMap object.

        Args:
            integrator (BaseIntegrator): The integrator used to integrate the map, default is RKIntegrator.
            **kwargs: Additional parameters to be passed to the integrator.
        """
        super().__init__(dim, True, domain)

        # Check if the integrator is a derived type of BaseIntegrator
        if not issubclass(integrator, BaseIntegrator):
            raise ValueError(
                "The Integrator is not a derived type of BaseIntegrator class."
            )

        self._integrator = integrator(**kwargs)
