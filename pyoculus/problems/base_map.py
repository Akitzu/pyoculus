from abc import ABC, abstractmethod
import numpy as np

class BaseMap(ABC):
    """
    Defines an abstract base class for the map subclasses.

    A map object is a function that transforms a point in phase space to another point. The transformation can be either continuous or discrete. A continuous transformation allows the map to be applied for a continuous time (f^t where t is a real number), while a discrete transformation allows the map to be applied only for discrete times (f^t where t is an integer).

    Attributes:
        dimension (int): The dimension of the phase space that the map operates on.
        is_continuous (bool): A flag indicating whether the map is continuous. If True, the map is continuous; if False, the map is discrete.

    Methods:
        f: This method represents the mapping function. It takes a point in phase space and returns its image under the map.
        df: This method calculates the Jacobian of the mapping function. The Jacobian is a matrix that describes the rate of change of the output of the map with respect to its input.
        lagrangian: This method calculates the Lagrangian, as defined in the paper by Meiss (https://doi.org/10.1063/1.4915831).
    
    Continuous maps should also implement the following methods:
        f_winding: This method calculates how a point in phase space winds around another point after a continuous application of the map.
        df_winding: This method calculates the Jacobian of the winding function.
    """

    def __init__(self, dim=2, continuous=True, domain=None):
        """Initializes BaseMap object.

        Args:
            dim (int): Dimension of the map.
            continuous (bool): Whether the map is continuous.
            domain (list of tuples, optional): The domain of the map. Each tuple should contain the lower and upper bounds for each dimension. If None, the domain is assumed to be (-inf, inf) for each dimension.
        """
        if domain is None:
            domain = [(-np.inf, np.inf)]*dim

        self.dimension = dim
        self.is_continuous = continuous
        self.domain = domain

    @abstractmethod
    def f(self, t, y0):
        """
        Applies the map 't' times to the initial point 'y0'.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
        """
        raise NotImplementedError("A BaseMap object should have a mapping f method.")

    @abstractmethod
    def df(self, t, y0):
        """
        Computes the Jacobian of the map at 'y0' after 't' applications.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The point in the phase space where the Jacobian is computed.
        """
        raise NotImplementedError(
            "A BaseMap object should have a jacobian mapping df method."
        )

    @abstractmethod
    def lagrangian(self, y0, t=None):
        """
        Calculates the Lagrangian at a given point or the difference in Lagrangian between two points.

        Args:
            y0 (array): The point at which to calculate the Lagrangian.
            y1 (array, optional): A second point. If provided, the function returns the difference in Lagrangian between y1 and y.

        Returns:
            float: The Lagrangian at y0, or the difference in Lagrangian between y1 and y0 (L(y1)-L(y0)) if y1 is provided.
        """
        raise NotImplementedError("A BaseMap object should have a lagrangian method.")
    
    @abstractmethod
    def f_winding(self, y0, y1=None):
        """
        Calculates how the point y0 winds around the point y1 after applying the map t times. This map should take two points in the domain and return a point into a space of same dimension where the last component is the winding number. 

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
            y1 (array): The point around which y0 winds. If None, the origin should be used.
        """
        raise NotImplementedError("A Continous BaseMap object should have a winding mapping f_winding method.")
    
    @abstractmethod
    def df_winding(self, t, y0, y1=None):
        """
        Calculates the Jacobian of the winding of y0 around y1 after applying the map t times.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
            y1 (array): The point around which y0 winds. If None, the origin should be used.
        """
        raise NotImplementedError("A Continous BaseMap object should have a jacobian winding mapping df_winding method.")