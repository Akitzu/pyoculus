"""
fixed_point.py
==================

Contains the class for finding fixed points of a map.

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from .base_solver import BaseSolver
import pyoculus.maps as maps
from ..utils.plot import create_canvas
from scipy.optimize import root
import numpy as np
import logging


logger = logging.getLogger(__name__)


class FixedPoint(BaseSolver):
    """
    Class to find fixed points of a map, i.e. points that satisfy :math:`f^t(x) = x`.
    """

    def __init__(self, map):
        # if constraints is None:
        #     constraints = np.NaN * np.ones(map.dimension)
        # elif len(constraints) != map.dimension:
        #     raise ValueError("The constraints should have the same dimension as the map domain.")
        # elif all([c is not np.NaN for c in constraints]):
        #     raise ValueError("Their must be at least one unconstrained dimension.")
        # self._constraints = constraints
    
        self._found_by_iota = False
        super().__init__(map)

    ## Properties

    @property
    def iotaslash(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        elif not self._found_by_iota:
            raise ValueError("Fixed point not found with winding number.")
        else:
            return self._n / self._m
        
    @property
    def n(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        elif not self._found_by_iota:
            raise ValueError("Fixed point not found with winding number.")
        else:
            return self._n
    
    @property
    def m(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        elif not self._found_by_iota:
            raise ValueError("Fixed point not found with winding number.")
        else:
            return self._m

    ## Findings fixed points methods

    def find(self, t, guess=None, nrestart=0, method="newton", **kwargs):
        """
        Tries to find a fixed point of the map applied :math:`t` times.

        Args:
            t: the number of iterations of the map
            guess: the initial guess of the fixed point
            nrestarts: the maximum number of restart with different random initial conditions
            method: the method to use to find the fixed point, default is 'newton'
            **kwargs: additional arguments for the method

        Returns:
            FixedPoint.OutputData: the output data of the fixed point search
                - coords: the coordinates of the fixed point
                - jacobians: the Jacobians of the fixed point
                - GreenesResidues: the Greene's Residue of the fixed point
        """

        # Check the iteration number is correct
        if self._map.is_discrete and not isinstance(t, int):
            raise ValueError("The iteration number should be an integer for a discrete map.")

        # Setup the search
        self.t = t
        self.history = []
        x_fp = None

        # Check the guess is correct
        if guess is None:
            guess = self.random_initial_guess()
        elif len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")
        elif not self._map.in_domain(guess):
            raise ValueError("The guess is not in the domain of the map.")

        # Setup and check the method
        if method == "newton":
            method_fun = self._newton_method
        elif method == "scipy.root":
            method_fun = self._scipy_root
        else:
            raise ValueError(f"Method {method} is not implemented.")
        self._method = method

        # run the solver, if failed, try a different random initial condition
        guess0 = guess.copy()
        for i in range(nrestart + 1):
            try:
                x_fp = method_fun(guess, **kwargs)
            except Exception as e:
                logger.info(f"Search {i} - failed: {e}")

            if x_fp is not None:
                break
            elif i < nrestart:
                logger.info(f"Search {i+1} starting from a random initial guesss.")
                guess = self.random_initial_guess(guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            logger.info(f"Found fixed point at {x_fp}. Computing additional data...")
            rdata = self.record_data(x_fp)

            # Set the successful flag
            self._successful = True
        else:
            rdata = None
            logger.info(f"Fixed point search unsuccessful for t={self.t}.")

        return rdata

    def find_with_iota(self, n, m, guess, x_axis=None, nrestart=0, method="newton", **kwargs):
        """
        Tries to find the fixed point of a map with winding number :math:`\\iota/2\\pi = q^{-1} = n/m` around x_axis.s

        Args:
            n (int): the numerator of the winding number
            m (int): the denominator of the winding number
            guess (array): the initial guess of the fixed point
            x_axis (array): the point around which the winding number is calculated
            nrestarts (int): the maximum number of restart with different random initial conditions
            method (str): the method to use to find the fixed point, default is 'newton'
            **kwargs: additional arguments for the method        

        Returns:
            FixedPoint.OutputData: the output data of the fixed point search
                - coords: the coordinates of the fixed point
                - jacobians: the Jacobians of the fixed point
                - GreenesResidues: the Greene's Residue of the fixed point
                - MeanResidues: --
        """

        # Setup the x_axis if not provided
        if x_axis is None:
            if isinstance(self._map, maps.ToroidalBfieldSection):
                x_axis = np.array([0., 0.])
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                x_axis = np.array([self._map.R0, self._map.Z0])
            else:
                logger.warning("No x_axis provided, using the zero vector.")
                x_axis = np.zeros(self._map.dimension)
        elif len(x_axis) != self._map.dimension:
            raise ValueError("The x_axis should have the same dimension as the map domain.")
        elif not self._map.in_domain(x_axis):
            raise ValueError("The x_axis is not in the domain of the map.")

        # Setup of the poloidal m and toroidal mode numbers
        if not isinstance(n, int) or not isinstance(m, int):
            raise ValueError("n and m should be integers")

        n = np.sign(n*m)*np.abs(n)
        m = int(np.abs(m))
        self._n = n
        self._m = m

        # Setup the search
        self.t = m
        self.history = []
        x_fp = None

        # Check the guess is right
        if guess is None:
            guess = self.random_initial_guess()
        elif len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")
        elif not self._map.in_domain(guess):
            raise ValueError("The guess is not in the domain of the map.")

        # Setup and check the method
        if method == "newton":
            method_fun = self._newton_method_winding
        elif method == "1D":
            method_fun = self._newton_method_1D
        else:
            raise ValueError(f"Method {method} is not implemented.")
        self._method = method

        # run the solver, if failed, try a different random initial condition
        guess0 = guess.copy()
        for i in range(nrestart + 1):
            try:
                x_fp = method_fun(guess, x_axis, **kwargs)
            except Exception as e:
                logger.info(f"Search {i} - failed: {e}")

            if x_fp is not None:
                break
            elif i < nrestart:
                logger.info(f"Search {i+1} starting from a random initial guesss.")
                guess = self.random_initial_guess(guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            logger.info(f"Found fixed point at {x_fp}. Computing additionnal data...")
            self._found_by_iota = True
            rdata = self.record_data(x_fp)

            # Set the successful flag
            self._successful = True
        else:
            logger.info(f"Fixed point search unsuccessful for iotaslash=n/m={n}/{m}.")
            rdata = None

        return rdata

    ## Utils methods

    def random_initial_guess(self, mu=None, sigma=None):
        """
        Returns a random initial point in the domain of the map using a Gaussian distribution.

        Args:
            mu (float): the mean of the Gaussian distribution
            sigma (np.array): the covariance matrix of the Gaussian distribution
        """
        domain = self._map.domain
        domain = [
            (
                low if low != -np.inf else -np.finfo(np.float64).max,
                high if high != np.inf else np.finfo(np.float64).max,
            )
            for (low, high) in domain
        ]

        if mu is None:
            mu = np.array([(low + high) / 2 for (low, high) in domain])
        if sigma is None:
            sigma = np.eye(self._map.dimension)

        return np.random.multivariate_normal(mu, sigma)

    def record_data(self, x_fp):
        """
        Record some additional data about the fixed point, such as the Jacobian, the Greene's Residue, and the Mean Residue for each iteration of the map.

        Args:
            x_fp (array): Fixed point coordinates
            is_winding (bool)
        """

        # Initialize the data arrays
        self.coords = np.zeros(
            shape=(self.t + 1, self._map.dimension), dtype=np.float64
        )
        self.jacobians = np.zeros(
            shape=(self.t + 1, self._map.dimension, self._map.dimension),
            dtype=np.float64,
        )
        self.GreenesResidues = np.zeros(self.t + 1, dtype=np.float64)
        if self._found_by_iota:
            self.MeanResidues = np.zeros(self.t + 1, dtype=np.float64)
        
        # Initial condition
        self.coords[0] = x_fp

        # Compute the rest of the data
        for i in range(0, self.t + 1):
            if i > 0:
                self.coords[i] = self._map.f(1, self.coords[i - 1])
            
            self.jacobians[i] = self._map.df(self.t, self.coords[i])
            self.GreenesResidues[i] = 0.25 * (2.0 - np.trace(self.jacobians[i]))
            if self._found_by_iota:
                self.MeanResidues[i] = np.power(
                    np.abs(self.GreenesResidues[i]) / 0.25, 1 / float(self._m)
                )

        # Create an output
        rdata = FixedPoint.OutputData()
        rdata.coords = self.coords.copy()
        rdata.jacobians = self.jacobians.copy()
        rdata.GreenesResidues = self.GreenesResidues.copy()
        if self._found_by_iota:
            rdata.MeanResidues = self.MeanResidues.copy()

        return rdata

    """
    Solver methods.

    They are private methods that are used to solve the fixed point problem. They should either return the coordinates of the fixed point if the search was successful or None if the search was not. 
    
    They can be of two types depending whether they need to be used with the winding number or not.
    """

    def _newton_method(self, guess, niter=100, tol=1e-10):
        x = np.array(guess, dtype=np.float64)
        self.history.append(x.copy())
        succeeded = False

        for i in range(niter):
            logger.info(f"Newton {i} - x : {x}")
            df = self._map.df(self.t, x)
            x_evolved = self._map.f(self.t, x)

            # Stop if the resolution is good enough
            logger.info(f"Newton {i} - delta_x : {x_evolved-x}")
            if np.linalg.norm(x_evolved - x) < tol:
                succeeded = True
                break

            # Newton's step
            delta_x = x_evolved - x
            step = np.linalg.solve(df - np.eye(self._map.dimension), -1 * delta_x)
            x_new = self._map.check_domain(x + step)

            # Update the variables
            logger.info(f"Newton {i} - step : {x_new-x}")
            x = x_new

            if not self._map.in_domain(x):
                logger.info(f"Newton {i} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None

    def _scipy_root(self, guess, **kwargs):
        """
        Wrapper around the scipy root method to find the fixed point. For more details, see the scipy documentation.
        """

        def fun(x):
            return self._map.f(self.t, x) - x

        return root(fun, guess, **kwargs).x

    def _newton_method_winding(self, guess, x_axis, niter=100, tol=1e-10):
        x = np.array(guess, dtype=np.float64)
        x_axis = np.array(x_axis, dtype=np.float64)

        self.history.append(x.copy())
        succeeded = False

        for i in range(niter):
            logger.info(f"Newton {i} - x : {x}")

            dW = self._map.dwinding(self.t, x)
            x_winding = self._map.winding(self.t, x)

            logger.info(
                f"Newton {i} - x_winding : {x_winding}"
            )

            # Stop if the resolution is good enough
            delta_x = x_winding - x
            if abs(delta_x[-1]) < tol:
                succeeded = True
                break

            # Newton's step
            step = np.linalg.solve(dW - np.eye(self._map.dimension), -1 * delta_x)
            x_new = self._map.check_domain(x + step)

            # Update the variables
            logger.info(f"Newton {i} - step: {x_new-x}")
            x = x_new
            logger.info(f"Newton {i} - x_new: {x_new}")

            if not self._map.in_domain(x):
                logger.info(f"Newton {i} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None

    def _newton_method_1D(self, guess, x_axis, theta=0, niter=100, tol=1e-10):
        """
        This will be rapidly depraciated.
        """
        if not isinstance(self._map, maps.ToroidalBfieldSection):
            raise ValueError("This method is only implemented for ToroidalBfieldSection")

        x = np.array([guess[0], theta], dtype=np.float64)
        x_axis = np.array(x_axis, dtype=np.float64)

        self.history.append(x.copy())
        succeeded = False

        for i in range(niter):
            logger.info(f"Newton {i} - x : {x}")

            dW = self._map.dwinding(self.t, x)
            x_winding = self._map.winding(self.t, x)

            logger.info(
                f"Newton {i} - x_winding : {x_winding}"
            )

            # Stop if the resolution is good enough
            delta_x = x_winding - x - np.array([0., self._n*2*np.pi/self._map._mf.Nfp])
            if abs(delta_x[-1]) < tol:
                succeeded = True
                break

            # Newton's step
            step = np.array([- delta_x[-1] / dW[1,0], 0.]) 
            x_new = x + step

            # Update the variables
            logger.info(f"Newton {i} - step: {x_new-x}")
            x = x_new
            logger.info(f"Newton {i} - x_new: {x_new}")

            if not self._map.in_domain(x):
                logger.info(f"Newton {i} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None

    # Plotting methods

    def plot(
        self, plot_all = True, **kwargs
    ):
        """
        Plot the fixed point with caracteristics of 
        """

        if not self.successful:
            raise Exception("A successful call of compute() is needed")
        elif self._map.dimension != 2:
            raise ValueError("I can only plot 2D fixed points.")

        fig, ax, kwargs = create_canvas(**kwargs)

        if kwargs.get("marker", None) is None:
            if self.GreenesResidues[0] > 1:
                # Alternating hyperbolic fixed point
                kwargs["marker"] = "s"
            elif self.GreenesResidues[0] < 0:
                # Hyperbolic fixed point
                kwargs["marker"] = "X"
            elif self.GreenesResidues[0] < 1:
                # Elliptic fixed point
                kwargs["marker"] = "o"
        
        if plot_all:
            ax.scatter(self.coords[:, 0], self.coords[:, 1], **kwargs)
        else:
            ax.scatter(self.coords[0, 0], self.coords[0, 1], **kwargs)

        return fig, ax


    def plot_history(self, **kwargs):
        """
        Plot the history of the fixed point search.
        """
        pass   
