## @file fixed_point.py
#  @brief class for finding fixed points
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .base_solver import BaseSolver
import numpy as np
import logging


logger = logging.getLogger(__name__)


class FixedPoint(BaseSolver):
    """
    Class to find fixed points of a map, i.e. points that satisfy :math:`f^t(x) = x`.
    """

    ## Findings fixed points methods

    def find(self, t, guess=None, niter=100, nrestart=0, tol=1e-10):
        """
        Finds a fixed point of a map applied 't' times. Once found, the fixed point has as attributes:
        - coords: the coordinates of the fixed point
        - jacobians: the Jacobians of the fixed point
        - GreenesResidues: the Greene's Residue of the fixed point
        - MeanResidues: the 'Average Residue' f as defined by Greene

        """
        if not self._map.is_continuous and not isinstance(t, int):
            raise ValueError(
                "The iteration number should be an integer for a discrete map."
            )

        if guess is None:
            guess = self.random_initial_guess()

        self.t = t
        self.history = []
        x_fp = None

        # arrays that save the data
        self.coords = np.zeros(
            shape=(self.t + 1, self._map.dimension), dtype=np.float64
        )
        self.jacobians = np.zeros(
            shape=(self.t + 1, self._map.dimension, self._map.dimension),
            dtype=np.float64,
        )
        self.GreenesResidues = np.zeros(self.t + 1, dtype=np.float64)
        self.MeanResidues = np.zeros(self.t + 1, dtype=np.float64)

        # set up the guess
        if len(guess) != self._map.dimension:
            raise ValueError(
                "The guess should have the same dimension as the map domain."
            )

        # run the Newton's method
        guess0 = guess.copy()
        for ii in range(nrestart + 1):
            try:  # run the solver, if failed, try a different random initial condition
                x_fp, jac = self._newton_method(guess, niter, tol)
                if self._successful:
                    break
            except Exception as e:
                logger.info(f"Search {ii} - failed: {e}")

            if ii < nrestart:
                logger.info(f"Search {ii+1} starting from a random initial guesss!")
                guess = self.random_initial_guess(guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            logger.info(f"Found fixed point at {x_fp}. Computing ...")
            self.coords[0] = x_fp
            self.jacobians[0] = jac

            for jj in range(0, self.t + 1):
                if jj > 0:
                    self.coords[jj] = self._map.f(1, self.coords[jj - 1])
                    self.jacobians[jj] = self._map.df(self.t, self.coords[jj])
                self.GreenesResidues[jj] = 0.25 * (2.0 - np.trace(self.jacobians[jj]))
                # self.MeanResidues[jj] = np.power(
                #     np.abs(self.GreenesResidues[jj]) / 0.25, 1 / float(self.qq)
                # )

            rdata = FixedPoint.OutputData()
            rdata.coords = self.coords.copy()
            rdata.jacobians = self.jacobians.copy()

            # Greene's Residue
            rdata.GreenesResidues = self.GreenesResidues.copy()
            # rdata.MeanResidues = self.MeanResidues.copy()

            # Set the successful flag
            self._successful = True
        else:
            rdata = None
            logger.info(f"Fixed point search unsuccessful for t={self.t}.")

        return rdata

    def find_with_iota(self, pp, qq, guess, x_axis, niter=100, nrestart=0, tol=1e-10):
        """
        Computes the fixed point of a continuous map with winding number iota = q^-1 = pp/qq around x_axis.

        Args:
            guess (array): the initial guess of the fixed point
            x_axis (array): the point around which the winding number is calculated
            niter (int): the maximum number of iterations
            nrestarts (int): the maximum number of restart with different initial conditions
            tol (float): the tolerance of the fixed point

        Returns:
            coords (array): the fixed point in the coordinates of the problem
            jac (array): the Jacobian the fixed point
            GreenesResidues (array): the Greene's Residue of the fixed point
            MeanResidues (array): the 'Average Residue' f as defined by Greene
        """

        if not self._map.is_continuous:
            raise ValueError(
                "The map needs to be continuous to find a fixed point with a winding number."
            )

        if not isinstance(pp, int) or not isinstance(qq, int):
            raise ValueError("pp and qq should be integers")

        if pp * qq >= 0:
            pp = int(np.abs(pp))
            qq = int(np.abs(qq))
        else:
            pp = -int(np.abs(pp))
            qq = int(np.abs(qq))

        gcd = np.gcd(pp, qq)
        self.t = qq // gcd

        self.pp = pp
        self.qq = qq

        self.history = []
        x_fp = None

        # arrays that save the data
        self.coords = np.zeros(
            shape=(self.qq + 1, self._map.dimension), dtype=np.float64
        )
        self.jacobians = np.zeros(
            shape=(self.qq + 1, self._map.dimension, self._map.dimension),
            dtype=np.float64,
        )
        self.GreenesResidues = np.zeros(self.qq + 1, dtype=np.float64)
        self.MeanResidues = np.zeros(self.qq + 1, dtype=np.float64)

        # set up the guess
        if len(guess) != self._map.dimension:
            raise ValueError(
                "The guess should have the same dimension as the map domain."
            )

        # run the Newton's method
        guess0 = guess.copy()
        for ii in range(nrestart + 1):
            try:  # run the solver, if failed, try a different random initial condition
                x_fp = self._newton_method_winding(guess, x_axis, niter, tol)
                if self._successful:
                    break
            except Exception as e:
                logger.info(f"Search {ii} - failed: {e}")

            if ii < nrestart:
                logger.info(f"Search {ii+1} starting from a random initial guesss!")
                guess = self.random_initial_guess(guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            self.coords[0] = x_fp
            self.jacobians[0] = self._map.df(1, x_fp)

            steps = 1 if self._map.is_continuous else self.pp
            for jj in range(0, self.qq + 1, steps):
                if self._map.is_continuous:
                    self.coords[jj] = self._map.f(jj / self.pp * self.t, x_fp)
                else:
                    self.coords[jj] = self._map.f(jj * self.t, x_fp)
                self.jacobians[jj] = self._map.df(jj * self.t, self.coords[jj])

            for jj in range(0, self.qq + 1, steps):
                self.GreenesResidues[jj] = 0.25 * (2.0 - np.trace(self.jacobians[jj]))
                self.MeanResidues[jj] = np.power(
                    np.abs(self.GreenesResidues[jj]) / 0.25, 1 / float(self.qq)
                )

            rdata = FixedPoint.OutputData()
            rdata.coords = self.coords.copy()
            rdata.jacobians = self.jacobians.copy()

            # Greene's Residue
            rdata.GreenesResidues = self.GreenesResidues.copy()
            rdata.MeanResidues = self.MeanResidues.copy()

            # Set the successful flag
            self._successful = True
        else:
            rdata = None
            logger.info(f"Fixed point search unsuccessful for pp/qq={self.pp}/{self.qq}.")

        return rdata

    def random_initial_guess(self, mu=None, sigma=None):
        """
        Returns a random initial guess for the fixed point inside the map domain.
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

    ## Newton's methods

    def _newton_method(self, guess, niter, tol):
        x = np.array(guess, dtype=np.float64)
        self.history.append(x.copy())
        succeeded = False

        for ii in range(niter):
            logger.info(f"Newton {ii} - x : {x}")
            df = self._map.df(self.t, x)
            x_evolved = self._map.f(self.t, x)

            # Stop if the resolution is good enough
            logger.info(f"Newton {ii} - delta_x : {x_evolved-x}")
            if np.linalg.norm(x_evolved - x) < tol:
                succeeded = True
                break

            # Newton's step
            delta_x = x_evolved - x
            step = np.linalg.solve(df - np.eye(self._map.dimension), -1 * delta_x)
            x_new = x + step

            # Update the variables
            logger.info(f"Newton {ii} - step : {x_new-x}")
            x = x_new

            if any(
                (xi < lwb) or (xi > upb) for xi, (lwb, upb) in zip(x, self._map.domain)
            ):
                logger.info(f"Newton {ii} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x, df
        else:
            return None

    def _newton_method_winding(self, guess, x_axis, niter, tol):
        x = np.array(guess, dtype=np.float64)
        x_axis = np.array(x_axis, dtype=np.float64)

        self.history.append(x.copy())
        succeeded = False

        for ii in range(niter):
            logger.info(f"Newton {ii} - RZ : {x}")

            dW = self._map.df_winding(self.t, x)
            x_evolved, _, x_winding = self._map.f_winding(self.t, x)

            # Stop if the resolution is good enough
            logger.info(
                f"Newton {ii} - [DeltaR, DeltaZ] : {x_evolved-x} - dtheta : {x_winding}"
            )
            if abs(x_winding[-1]) < tol:
                succeeded = True
                break

            # Newton's step
            step = np.linalg.solve(dW, -1 * x_winding)
            x_new = x + step

            # Update the variables
            logger.info(f"Newton {ii} - step: {x_new-x}")
            x = x_new

            if any(
                (xi < lwb) or (xi > upb) for xi, lwb, upb in zip(x, self._map.domain)
            ):
                logger.info(f"Newton {ii} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None

    # Plotting method

    def plot(
        self, plottype=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs
    ):
        """! Generates the plot for fixed points
        @param plottype which variables to plot: 'RZ' or 'yx', by default using "poincare_plot_type" in problem
        @param xlabel,ylabel what to put for the xlabel and ylabel, by default using "poincare_plot_xlabel" in problem
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """
        import matplotlib.pyplot as plt

        if not self._successful:
            raise Exception("A successful call of compute() is needed")

        # default setting
        if plottype is None:
            plottype = self._problem.poincare_plot_type
        if xlabel is None:
            xlabel = self._problem.poincare_plot_xlabel
        if ylabel is None:
            ylabel = self._problem.poincare_plot_ylabel

        if plottype == "RZ":
            xdata = self.x
            ydata = self.z
        elif plottype == "yx":
            xdata = self.y
            ydata = self.x
        elif plottype == "st":
            xdata = np.mod(self.theta, 2 * np.pi)
            ydata = self.s
        else:
            raise ValueError("Choose the correct type for plottype")

        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
            newfig = False
        else:
            fig, ax = plt.subplots()
            newfig = True

        # set default plotting parameters
        # use x
        if kwargs.get("marker") is None:
            kwargs.update({"marker": "x"})
        # use gray color
        if kwargs.get("c") is None:
            kwargs.update({"c": "black"})

        xs = ax.plot(xdata, ydata, linestyle="None", **kwargs)

        if not newfig:
            if plottype == "RZ":
                plt.axis("equal")
            if plottype == "yx":
                pass

            plt.xlabel(xlabel, fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
