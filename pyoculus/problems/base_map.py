class BaseMap:
    """
    Defines a base class for a map object.

    A map object is a function that transforms a point in phase space to another point. The transformation can be either continuous or discrete. A continuous transformation allows the map to be applied for a continuous time (f^t where t is a real number), while a discrete transformation allows the map to be applied only for discrete times (f^t where t is an integer).

    Attributes:
        dimension (int): The dimension of the phase space that the map operates on.
        is_continuous (bool): A flag indicating whether the map is continuous. If True, the map is continuous; if False, the map is discrete.

    Methods:
        f: This method represents the mapping function. It takes a point in phase space and returns its image under the map.
        df: This method calculates the Jacobian of the mapping function. The Jacobian is a matrix that describes the rate of change of the output of the map with respect to its input.
        lagrangian: This method calculates the Lagrangian, as defined in the paper by Meiss (https://doi.org/10.1063/1.4915831).
    """

    def __init__(self, dim=2, continuous=True):
        """Initializes BaseMap object.

        Args:
            dim (int): Dimension of the map.
            continuous (bool): Whether the map is continuous.
        """
        self.dimension = dim
        self.is_continous = continuous

    def f(self, t, y0):
        """
        Applies the map 't' times to the initial point 'y0'.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
        """
        raise NotImplementedError("A BaseMap object should have a mapping f method.")

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

    def lagrangian(self, y0, t = None):
        """
        Calculates the Lagrangian at a given point or the difference in Lagrangian between two points.

        Args:
            y0 (array): The point at which to calculate the Lagrangian.
            y1 (array, optional): A second point. If provided, the function returns the difference in Lagrangian between y1 and y.

        Returns:
            float: The Lagrangian at y0, or the difference in Lagrangian between y1 and y0 (L(y1)-L(y0)) if y1 is provided.
        """
        raise NotImplementedError("A BaseMap object should have a lagrangian method.")