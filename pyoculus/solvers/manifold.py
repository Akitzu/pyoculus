import pyoculus.maps as maps
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..utils.plot import create_canvas, clean_bigsteps
from scipy.optimize import root, minimize
from typing import Iterator, Literal

# from functools import total_ordering
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np

import logging

logger = logging.getLogger(__name__)


def eig(jacobian):
    """
    Compute stable and unstable eigenvalues/eigenvectors of a fixed point.

    This function calculates the eigenvalues and eigenvectors of a given Jacobian matrix
    and separates them into stable and unstable components based on their magnitudes.

    Args:
        jacobian (np.ndarray): A 2x2 Jacobian matrix at the fixed point.

    Returns:
        tuple:
            lambda_s (float): The stable eigenvalue (:math:`\\vert\\lambda\\vert < 1`)
            vector_s (np.ndarray): The corresponding stable eigenvector
            lambda_u (float): The unstable eigenvalue (:math:`\\vert\\lambda\\vert > 1`)
            vector_u (np.ndarray): The corresponding unstable eigenvector

    Examples:
        >>> J = np.array([[1.5, 0.5], [0.5, 2.0]])
        >>> lambda_s, v_s, lambda_u, v_u = eig(J)
    """

    eigRes = np.linalg.eig(jacobian)
    eigenvalues = np.abs(eigRes[0])

    # Eigenvectors are stored as columns of the matrix eigRes[1], transposing it to access them as np.array[i]
    eigenvectors = eigRes[1].T

    # Extract the index of the stable and unstable eigenvalues
    s_index, u_index = 0, 1
    if eigenvalues[0].real > eigenvalues[1].real:
        s_index, u_index = 1, 0

    return (
        eigenvalues[s_index],
        eigenvectors[s_index],
        eigenvalues[u_index],
        eigenvectors[u_index],
    )


class Clinic:
    """A class representing the trajectory of a homoclinic/heteroclinic point.

    This class handles the computation and storage of a heteroclinic/homoclinic trajectory, which represent intersections between stable and unstable manifolds of fixed points.

    Args:
        manifold: The :class:`Manifold` object associated to the fixed points and map analyzed.
        eps_s (float): Initial distance in the linear regime along stable manifold direction.
        eps_u (float): Initial distance in the linear regime along unstable manifold direction.
        n_s (int): Number of iterations to apply to the intersection closest to the stable fixed point.
        n_u (int): Number of iterations to apply to the intersection closest to the unstable fixed point.

    Attributes:
        eps_s (float): Distance parameter along stable manifold.
        eps_u (float): Distance parameter along unstable manifold.
        nint_s (int): Number of stable iterations.
        nint_u (int): Number of unstable iterations.
        _fundamental_segments (dict): Fundamental domain bounds.
        _trajectory (np.ndarray): Computed clinic orbit.
        _path_s (np.ndarray): Stable manifold path.
        _path_u (np.ndarray): Unstable manifold path.
        _xend_s (np.ndarray): End point on stable manifold.
        _xend_u (np.ndarray): End point on unstable manifold.
    """

    def __init__(
        self, manifold: "Manifold", eps_s: float, eps_u: float, n_s: int, n_u: int
    ) -> None:
        self._manifold = manifold
        self.eps_s = eps_s
        self.eps_u = eps_u
        self.nint_s = n_s
        self.nint_u = n_u
        self._fundamental_segments = None
        self._trajectory = None
        self._path_s = None
        self._path_u = None
        self._xend_s = None
        self._xend_u = None

    @property
    def trajectory(self):
        """Get the complete trajectory of the clinic point.

        Computes the trajectory by integrating along stable and unstable
        manifolds if not already calculated.

        Returns:
            np.ndarray: Array containing the orbit from unstable to stable fixed point.
        """

        if self._trajectory is not None:
            return self._trajectory

        path_u = self._manifold.integrate(
            self._manifold.rfp_u + self.eps_u * self._manifold.vector_u, self.nint_u, +1
        )[:, 0, :]
        path_s = self._manifold.integrate(
            self._manifold.rfp_s + self.eps_s * self._manifold.vector_s, self.nint_s, -1
        )[:, 0, :]

        # self._path_u, self._path_s = path_u.T, path_s.T
        self._trajectory = np.concatenate((path_u, path_s))

        return self._trajectory

    @property
    def x_end_s(self):
        """Get the endpoint on the stable manifold.

        Returns:
            np.ndarray: Coordinates of the end point on stable manifold.
        """
        if self._xend_s is not None:
            return self._xend_s
        elif self._path_s is not None:
            self._xend_s = self._path_s[-1, :]
        else:
            self.trajectory
            self._xend_s = self._path_s[-1, :]

        return self._xend_s

    @property
    def x_end_u(self):
        """Get the endpoint on the unstable manifold.

        Returns:
            np.ndarray: Coordinates of the end point on unstable manifold.
        """
        if self._xend_u is not None:
            return self._xend_u
        elif self._path_u is not None:
            self._xend_u = self._path_u[-1, :]
        else:
            self.trajectory
            self._xend_u = self._path_u[-1, :]

        return self._xend_u

    @property
    def fundamental_segments(self):
        """Get the fundamental domain boundaries.

        Returns:
            dict: Contains 'stable' and 'unstable' segment bounds.
        """
        if self._fundamental_segments is None:
            bnd_s, bnd_u = self._fundamental_segments_from_eps()
            self._fundamental_segments = {"stable": bnd_s, "unstable": bnd_u}
        return self._fundamental_segments

    # Private methods

    def _fundamental_segments_from_eps(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate the fundamental segment on unstable and stable manifolds.

        Computes the bounds of fundamental domains by evolving points along
        the manifolds and measuring their distances from fixed points.

        Returns:
            tuple: A tuple containing two tuples:
                - (eps_s, upperbound_s): The initial ε_s and computed upper bound
                  for the stable manifold.
                - (eps_u, upperbound_u): The initial guess and computed upper bound
                  for the unstable manifold.
        """
        # Initial points along the manifolds
        r_s = self._manifold.rfp_s + self.eps_s * self._manifold.vector_s
        r_u = self._manifold.rfp_u + self.eps_u * self._manifold.vector_u

        # Evolve the points along the manifolds
        # r_s_unevolved = self._map.f(+1*self.fixedpoint_1.m, r_s)
        # r_s_evolved   = self._map.f(-1*self.fixedpoint_1.m, r_s)
        r_s_evolved = self._manifold.integrate(r_s, 1, -1)[1, 0] 

        # r_u_unevolved = self._map.f(-1*self.fixedpoint_1.m, r_u)
        # r_u_evolved   = self._map.f(+1*self.fixedpoint_1.m, r_u)
        r_u_evolved = self._manifold.integrate(r_u, 1, +1)[1, 0]

        # Measure the distance from the fixed points with usual two norm
        upperbound_s = np.linalg.norm(r_s_evolved - self._manifold.rfp_s)
        upperbound_u = np.linalg.norm(r_u_evolved - self._manifold.rfp_u)

        return (self.eps_s, upperbound_s), (self.eps_u, upperbound_u)


class ClinicSet:
    """A collection of homoclinics/heteroclinics with unstable (:math:`>_u`) ordering.

    This class manages multiple :class:`Clinic` objects, maintaining their ordering and ensuring
    proper fundamental domain representation. It uses the first clinic added to manage the fundamental domain boundaries.

    Args:
        manifold: The :class:`Manifold` object associated to the fixed points and the map analyzed.

    Attributes:
        _clinics_list (list): List of Clinic objects.
        fundamental_segments (dict): Fundamental domain boundaries.
        nint_pair (tuple): Default iteration numbers (n_s, n_u).

    Methods:
        record_clinic: Add a new clinic point to the collection.
        reset: Clear all stored clinics.

    Examples:
        >>> a_manifold = Manifold(...)
        >>> clinic_set = ClinicSet(a_manifold)
        >>> clinic_set.record_clinic(eps_s=0.1, eps_u=0.1, n_s=10, n_u=10)
    """

    DEFAULT_TOLERANCE = 1e-2
    MAX_ITERATIONS = 20

    def __init__(self, manifold: "Manifold") -> None:
        self._manifold = manifold
        self.reset()

    def __len__(self) -> int:
        return len(self._clinics_list)

    @property
    def size(self) -> int:
        """Number of clinic points in the set."""
        return len(self._clinics_list)

    @property
    def is_empty(self) -> bool:
        """Check if the clinic set is empty."""
        return len(self._clinics_list) == 0

    # Make the class indexable
    def __getitem__(self, index: int) -> Clinic:
        return self._clinics_list[index]

    # Make the class iterable
    def __iter__(self) -> Iterator[Clinic]:
        return iter(self._clinics_list)

    # Public methods

    def record_clinic(
        self, eps_s: float, eps_u: float, n_s: int, n_u: int, **kwargs
    ) -> None:
        """Record a new clinic point in the fundamental domain.

        Creates and stores a new :class:`Clinic` object after converting the given parameters
        to their fundamental domain representation.

        Args:
            eps_s (float): Initial distance along stable manifold.
            eps_u (float): Initial distance along unstable manifold.
            n_s (int): Number of iterations for stable manifold.
            n_u (int): Number of iterations for unstable manifold.
            **kwargs: Additional keyword arguments:
                tol (float, optional): Tolerance for comparing epsilon values. Defaults to 1e-2.

        Note:
            If this is the first clinic point, it establishes the fundamental domain boundaries.
            Otherwise, parameters are converted to their fundamental domain representation.
        """
        if not self._clinics_list:
            clinic = Clinic(self._manifold, eps_s, eps_u, n_s, n_u)
            self.fundamental_segments = clinic.fundamental_segments
            self.nint_pair = (n_s, n_u)
            self._clinics_list.append(clinic)
        else:
            fundamental_eps_s, n_s_shift = self._find_fundamental_eps(eps_s, "stable")
            fundamental_eps_u, n_u_shift = self._find_fundamental_eps(eps_u, "unstable")
            clinic = Clinic(
                self._manifold,
                fundamental_eps_s,
                fundamental_eps_u,
                n_s + n_s_shift,
                n_u + n_u_shift,
            )

            tol = kwargs.get("tol", self.DEFAULT_TOLERANCE)
            if not np.any(
                [
                    np.isclose(clinic.eps_u, other.eps_u, rtol=tol)
                    for other in self._clinics_list
                ]
            ):
                self._clinics_list.append(clinic)
                self._orderize()
            else:
                logger.warning("Homo/heteroclinic already recorded, skipping...")

    def reset(self) -> None:
        """Reset the clinic set to its initial empty state.

        Clears:
            - All stored clinic points
            - Fundamental segment boundaries
            - Default iteration numbers (nint_pair)
        """
        self._clinics_list = []
        self.fundamental_segments = None
        self.nint_pair = None

    # Private methods

    def _orderize(self) -> None:
        """Sorts the internal list of clinic points based on their eps_u values."""
        self._clinics_list = [
            self._clinics_list[i]
            for i in np.argsort([x.eps_u for x in self._clinics_list])
        ]

    def _find_fundamental_eps(
        self, eps: float, which: Literal["stable", "unstable"], **kwargs
    ) -> tuple[float, int]:
        """
        Find the epsilon parameter lying within the fundamental segment (stable or unstable) of the ClinicSet for a given epsilon value.

        Args:
            eps (float): The initial epsilon value to convert.
            which (str): Either "stable" or "unstable", which manifold eps lies on.
            **kwargs: Additional keyword arguments:
                max_iters (int, optional): Maximum number of iterations to find the fundamental eps.

        Returns:
            tuple: (fundamental_eps, n_shift)
                - fundamental_eps (float): The equivalent eps in fundamental segment.
                - n_shift (int): Number of iterations needed for the shift.

        Raises:
            ValueError: If which is neither "stable" nor "unstable".
            RuntimeError: If fundamental epsilon is not found within max_iters iterations.
        """
        if which == "stable":
            rfp, eigenvector, forward_dir = (
                self._manifold.rfp_s,
                self._manifold.vector_s,
                -1,
            )
        elif which == "unstable":
            rfp, eigenvector, forward_dir = (
                self._manifold.rfp_u,
                self._manifold.vector_u,
                +1,
            )
        else:
            raise ValueError(
                f"Invalid manifold selection: {which}. Must be 'stable' or 'unstable'"
            )

        fund = self.fundamental_segments[which]
        r_cur = rfp + eps * eigenvector
        norm = eps
        n_shift = 0

        # If the eps is less then the lower bound, then the direction should be forward otherwise backward
        forward_dir *= 1 if eps < fund[0] else -1

        # Set a maximum number of iterations
        max_iterations = kwargs.get("max_iters", self.MAX_ITERATIONS)
        for _ in range(max_iterations):
            if fund[0] <= norm < fund[1]:
                return norm, n_shift
            r_cur = self._manifold._map.f(-1 * self._manifold.fixedpoint_1.m, r_cur)
            norm = np.linalg.norm(r_cur - rfp)
            n_shift -= 1 * forward_dir
            logger.debug(f"Current epsilon (from norm calculation): {norm}")

        raise RuntimeError(
            "Failed to find a solution within the maximum number of iterations"
        )

    def order(self):
        """
        Order the homo/hetero-clinic points with the induced linear ordering of the unstable manifold >_u.
        """
        self._clinics_list = [
            self._clinics_list[i]
            for i in np.argsort([x.eps_u for x in self._clinics_list])
        ]

    def reset(self):
        """
        remove all known clinics and start afresh. 
        """
        self._clinics_list = []
        self.fundamental_segments = None
        self.nint_pair = None




class Manifold(BaseSolver):
    """Class for computing and analyzing a tangle composed of one stable and one unstable manifold of fixed points.

    This class handles the computation of stable and unstable manifolds for fixed points, including finding homoclinic/heteroclinic intersections and calculating the turnstile flux of the tangle.

    Args:
        map (maps.base_map): The map defining the dynamical system.
        fixedpoint_1 (FixedPoint): First fixed point to consider.
        fixedpoint_2 (FixedPoint, optional): Second fixed point to consider if the manifolds go from one to the other.
        dir1 (str, optional): Direction ('+' or '-') for first manifold.
        dir2 (str, optional): Direction ('+' or '-') for second manifold.
        is_first_stable (bool, optional): Whether the first fixed point direction captures the stable manifold departure.

    Attributes:
        fixedpoint_1 (FixedPoint): First fixed point.
        fixedpoint_2 (FixedPoint): Second fixed point.
        rfp_s (np.ndarray): Stable fixed point coordinates.
        rfp_u (np.ndarray): Unstable fixed point coordinates.
        vector_s (np.ndarray): Stable eigenvector.
        vector_u (np.ndarray): Unstable eigenvector.
        lambda_s (float): Stable eigenvalue.
        lambda_u (float): Unstable eigenvalue.
        stable (np.array): Stable manifold points.
        unstable (np.array): Unstable manifold points.
        clinics (ClinicSet): Set of homoclinic/heteroclinic intersections.
        turnstile_areas (np.array): Turnstile areas of the tangle.

    Methods:
        choose: Choose the stable and unstable directions for the manifold.
        show_directions: Plot the fixed points and their stable/unstable directions.
        show_current_directions: Plot the current stable and unstable directions.
        error_linear_regime: Metric to evaluate if a point is in the linear regime of a fixed point.
        start_config: Compute a starting configuration for the manifold drawing.
        find_epsilon: Find the epsilon that lies in the linear regime.
        compute_manifold: Compute the stable or unstable manifold.
        compute: Compute the stable and unstable manifolds.
        plot: Plot the stable and unstable manifolds.
        find_N: Find the number of times the map needs to be applied for the stable and unstable points to cross.
        find_clinic_single: Find a single homoclinic/heteroclinic intersection.
        find_clinic: Find all homoclinic/heteroclinic intersections.
        compute_turnstile_areas: Compute the turnstile areas of the tangle.

    Raises:
        TypeError: If fixed points are not FixedPoint instances.
        ValueError: If fixed points are not successfully computed.
    """

    def __init__(
        self,
        map: maps.base_map,
        fixedpoint_1: FixedPoint,
        fixedpoint_2: FixedPoint = None,
        dir1: str = None,
        dir2: str = None,
        is_first_stable: bool = None,
    ) -> None:
        """
        Initialize the Manifold class by providing two fixed points and specifying if the first fixedpoint is stable. If only one fixed point is specified, a homoclinic connection is assumed. 
        Directions are specified with the string '+' or '-' to indicate which eigenvectors of the fixed points to follow (remember that if $v$ is an eingenvector, $-v$ is as well). 

        Args:
            map (maps.base_map): The map to use for the computation.
            fixedpoint_1 (FixedPoint): first fixed point
            fixedpoint_2 (FixedPoint, optional): second fix point if not homoclinic manifold. Defaults to None.
        """

        # Check that the fixed points are correct FixedPoint instances
        if not isinstance(fixedpoint_1, FixedPoint):
            raise TypeError("Fixed point must be an instance of FixedPoint class")
        if not fixedpoint_1.successful:
            raise ValueError("Need a successful fixed point to compute the manifold")

        if isinstance(fixedpoint_2, FixedPoint):
            if not fixedpoint_2.successful:
                raise ValueError(
                    "Need a successful fixed point to compute the manifold"
                )

        # Initialize the directions and the dictionnaries
        if fixedpoint_2 is not None:
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_2
        else:
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_1

        # Setting the fast and slow directions
        if dir1 is None or dir2 is None or is_first_stable is None:
            logger.warning("No choice of direction was made. Proceeding with default.")
            dir1, dir2, is_first_stable = "+", "+", True
        self.choose(dir1, dir2, is_first_stable)

        # Initialize the clinic set
        self.clinics = ClinicSet(self)

        # Initialize the BaseSolver
        super().__init__(map)

    def choose(
        self, dir_1: Literal["+", "-"], dir_2: Literal["+", "-"], is_first_stable: bool
    ) -> None:
        """Choose manifold stable and unstable directions to define your :class:`Manifold` problem.

        You must choose directions away from the fixed point in which the manifolds actually intersect. The good orientation is the one for which you could create the manifold by going away from the fixed point. Be carefull to this point, otherwise other manifold computations such as clinic finding will fail.

        Hint: Use :meth:`Manifold.show_directions` to help you choose here. This plot shows the fixed points and the stable eigenvector (and its negative) in green, the unstable eigenvector (and it's negative) in red.

        Args:
            dir_1 (str): '+' or '-' for the stable direction.
            dir_2 (str): '+' or '-' for the unstable direction.
            first_stable (bool): Whether the first fixed point is a stable direction.
        """
        if is_first_stable:
            fp_s, fp_u = self.fixedpoint_1, self.fixedpoint_2
        else:
            fp_s, fp_u = self.fixedpoint_2, self.fixedpoint_1

        signs = [(-1) ** int(d == "-") for d in dir_1 + dir_2]

        # Choose the fixed points and their directions
        self.rfp_s = fp_s.coords[0]
        self.lambda_s, self.vector_s, _, _ = eig(fp_s.jacobians[0])

        self.rfp_u = fp_u.coords[0]
        _, _, self.lambda_u, self.vector_u = eig(fp_u.jacobians[0])

        # Assign the fixed points and their directions
        self.vector_s *= signs[0]
        self.vector_u *= signs[1]

        # Initialize the manifolds
        self._stable_trajectory = None
        self._unstable_trajectory = None

    @classmethod
    def show_directions(
        cls, fp_1: FixedPoint, fp_2: FixedPoint, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot fixed points and their stable/unstable directions.

        Helper function to plot the fixed points and their stable and unstable direction. Usefull to look at which direction need to be considered for the inner and outer manifolds before creating a class and analyzed them.

        Args:
            fp_1 (FixedPoint): First fixed point.
            fp_2 (FixedPoint): Second fixed point.
            **kwargs: Optional visualization parameters:
                pcolors (list): Colors for fixed points.
                vcolors (list): Colors for eigenvectors.
                vscale (int): Scale for eigenvectors.
                dvtext (float): Text distance as fraction.

        Returns:
            tuple: (figure, axis) matplotlib objects.
        """
        # Defaults
        pcolors = kwargs.get("pcolors", ["tab:blue", "tab:orange"])
        vcolors = kwargs.get("vcolors", ["tab:green", "tab:red"])
        vscale = kwargs.get("vscale", 18)
        dvtext = kwargs.get("dvtext", 0.005)

        # Set the figure and ax
        fig, ax, kwargs = create_canvas(**kwargs)

        # Choose the fixed points and their directions
        rfp_1 = fp_1.coords[0]
        _, p1_vector_s, _, p1_vector_u = eig(fp_1.jacobians[0])
        rfp_2 = fp_2.coords[0]
        _, p2_vector_s, _, p2_vector_u = eig(fp_2.jacobians[0])

        # Plot the fixed points
        ax.scatter(
            *rfp_1,
            marker="X",
            s=100,
            label="Fixed point 1",
            zorder=999,
            color=pcolors[0],
            edgecolor="black",
            linewidth=1,
        )
        ax.scatter(
            *rfp_2,
            marker="X",
            s=100,
            label="Fixed point 2",
            zorder=999,
            color=pcolors[1],
            edgecolor="black",
            linewidth=1,
        )
        # ax.text(*(rfp_1), '1', zorder=999, ha='center', va='center')
        # ax.text(*(rfp_2), '2', zorder=999, ha='center', va='center')

        # Plot the eigenvectors
        def plot_eigenvectors(ax, rfp, vectors):
            for vector, color in zip(vectors, vcolors):
                # Positive and negative arrows
                p_arrow = FancyArrowPatch(
                    rfp,
                    rfp + vector / vscale,
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                )
                n_arrow = FancyArrowPatch(
                    rfp,
                    rfp - vector / vscale,
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                )

                # Add the arrows to the plot
                ax.add_patch(p_arrow)
                ax.add_patch(n_arrow)

                # Add the text
                ax.text(
                    *(rfp + vector * (dvtext + 1 / vscale)),
                    "+",
                    zorder=999,
                    color=color,
                    fontsize="large",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
                ax.text(
                    *(rfp - vector * (dvtext + 1 / vscale)),
                    "-",
                    zorder=999,
                    color=color,
                    fontsize="large",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

        plot_eigenvectors(ax, rfp_1, [p1_vector_s, p1_vector_u])
        plot_eigenvectors(ax, rfp_2, [p2_vector_s, p2_vector_u])

        return fig, ax

    def show_current_directions(self, **kwargs):
        pass

    def error_linear_regime(
        self,
        epsilon: float,
        rfp: np.ndarray,
        eigenvector: np.ndarray,
        direction: int = 1,
    ) -> float:
        """Calculate error in linear regime approximation.

        Metric to estimate if the point rfp + epsilon * eigenvector is in the linear regime of rfp point.

        Args:
            epsilon (float): Distance from fixed point.
            rfp (np.ndarray): Fixed point coordinates.
            eigenvector (np.ndarray): Eigenvector to check.
            direction (int, optional): Integration direction. Defaults to 1.

        Returns:
            float: Error metric for linear approximation.
        """
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[1, 0, :] - rz_path[0, 0, :]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

        # Use the dot product to see if: cos(angle btw eps_dir_norm and eigenvector) is close to 1
        return np.abs(1 - np.dot(eps_dir_norm, eigenvector))

    ### Computation of the manifolds

    def start_config(self, epsilon, rfp, eigenvalue, eigenvector, neps, direction=1):
        """
        Compute a starting configuration for the manifold drawing. It takes a point in the linear regime
        and devide the interval from the point to its evolution after one nfp into neps points. The interval
        is computed geometrically.

        Args:
            epsilon (float): initial epsilon
            rfp (np.array): fixed point
            eigenvalue (float): eigenvalue of the fixed point
            eigenvector (np.array): eigenvector of the fixed point
            neps (int): number of points
            direction (int): direction of the integration (1 for forward, -1 for backward)

        Returns:
            np.array: array of starting points (shape (neps, 2))
        """
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[0, 0, :] - rz_path[1, 0, :]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

        # Geometric progression from log_eigenvalue(epsilon) to log_eigenvalue(epsilon + norm_eps_dir)
        eps = np.logspace(
            np.log(epsilon) / np.log(eigenvalue),
            np.log(epsilon + norm_eps_dir) / np.log(eigenvalue),
            neps,
            base=eigenvalue,
        )

        Rs = rfp[0] + eps * eps_dir_norm[0]
        Zs = rfp[1] + eps * eps_dir_norm[1]
        return np.array([Rs, Zs]).T

    def find_epsilon(self, which: str, eps_guess=1e-3):
        """
        Find the epsilon that lies in the linear regime.
        """
        if which == "stable":
            rfp, eigenvector, direction = self.rfp_s, self.vector_s, -1
        elif which == "unstable":
            rfp, eigenvector, direction = self.rfp_u, self.vector_u, +1
        else:
            raise ValueError("Invalid manifold selection.")

        find_eps = lambda x: self.error_linear_regime(
            x, rfp, eigenvector, direction=direction
        )
        minobj = minimize(find_eps, eps_guess, bounds=[(0, 1)], tol=1e-12)

        if not minobj.success:
            raise ValueError("Search for minimum of the linear error failed.")

        esp_root = minobj.x[0]
        logger.info(
            f"Search for minimum of the linear error succeeded, epsilon = {esp_root:.5e}."
        )
        return esp_root

    def compute_manifold(self, which: str, eps=None, **kwargs):
        """
        Compute the stable or unstable manifold.

        Args:
            eps (float): epsilon in the stable or unstable direction
            compute_stable (bool): whether to compute the stable or unstable manifold

        Keyword Args:
            eps_guess (float): guess for epsilon (if eps is not given)
            neps (int): number of points in the starting configuration
            nint (int): number of intersections

        Returns:
            np.array: array of points on the manifold
        """
        # Check the manifold selection
        if which not in ["stable", "unstable"]:
            raise ValueError("Invalid manifold selection.")
        compute_stable = True if which == "stable" else False

        # Set the right parameters
        if compute_stable:
            rfp, vector, lambda_, goes = self.rfp_s, self.vector_s, self.lambda_s, -1
        else:
            rfp, vector, lambda_, goes = self.rfp_u, self.vector_u, self.lambda_u, 1

        # If the epsilon is not given, find the best one in the linear regime
        eps_guess = kwargs.get("eps_guess", 1e-3)
        if eps is None:
            eps = self.find_epsilon(which, eps_guess)

        # Compute the starting configuration and the manifold
        neps, nint = kwargs.get("neps", 40), kwargs.get("nint", 6)
        RZs = self.start_config(eps, rfp, lambda_, vector, neps, goes)
        logger.info(f"Computing {which} manifold...")
        manifoldpoints = self.integrate(RZs, nintersect=nint, direction=goes)
        orderedmanifoldpoints = np.concatenate(manifoldpoints)  # put points in order: first intersections, second intersections, etc)

        if which == "stable":
            self._stable_trajectory = orderedmanifoldpoints
        else:  
            self._unstable_trajectory = orderedmanifoldpoints

        return orderedmanifoldpoints

    @property
    def stable(self):
        if self._stable_trajectory is None:
            logger.warning(
                "Stable manifold not computed. Using the computation method."
            )
            self.compute_manifold(1e-5, True)
        return self._stable_trajectory

    @property
    def unstable(self):
        if self._unstable_trajectory is None:
            logger.warning(
                "Unstable manifold not computed. Using the computation method."
            )
            self.compute_manifold(1e-5, False)
        return self._unstable_trajectory

    def compute(self, eps_s=None, eps_u=None, **kwargs):
        """
        Computation of the stable and unstable manifolds.

        Args:
            eps_s (float): epsilon in the stable direction.
            eps_u (float): epsilon in the unstable direction

        Keyword Args:
            eps_guess_s (float): guess for epsilon in the stable direction (if eps_s is not given)
            eps_guess_u (float): guess for epsilon in the unstable direction (if eps_u is not given)
            neps_s (int): number of points in the starting configuration for the stable part
            neps_u (int): number of points in the starting configuration for the unstable part
            nint_s (int): number of evolutions of the initial segments for the stable part
            nint_u (int): number of evolutions of the initial segments for the unstable part

        Returns:
            tuple: A tuple containing two np.array of points, the first one for the stable manifold and the second one for the unstable manifold.
        """
        # Extract the keyword arguments
        kwargs_s = {
            key[:-2]: kwargs.pop(key)
            for key in ["eps_guess_s", "neps_s", "nint_s"]
            if key in kwargs
        }
        kwargs_u = {
            key[:-2]: kwargs.pop(key)
            for key in ["eps_guess_u", "neps_u", "nint_u"]
            if key in kwargs
        }
        if kwargs:
            logger.warning(f"Unused keyword arguments: {kwargs}")

        # Compute the manifolds
        self.compute_manifold("stable", eps_s, **kwargs_s)
        self.compute_manifold("unstable", eps_u, **kwargs_u)

        return self._stable_trajectory, self._unstable_trajectory

    def plot(self, which="both", stepsize_limit=None, **kwargs):
        """
        Plot the stable and/or unstable manifolds.

        kwargs:
        which (str): which manifold to plot. Can be 'stable', 'unstable' or 'both'.
        stepsize_limit = 

        Other kwargs are givent to the plot. 

        Specific extra plots: 
        *rm_points* (int): remove the last *rm_points* points of the manifold.

        """
        fig, ax, kwargs = create_canvas(**kwargs)

        colors = kwargs.pop("colors", ["green", "red"])
        markersize = kwargs.pop("markersize", 2)
        fmt = kwargs.pop("fmt", "-o")
        rm_points = kwargs.pop("rm_points", 0)
        final_index = -rm_points - 1

        for i, dir in enumerate(["stable", "unstable"]):
            if dir == which or which == "both":
                points = self.stable if dir == "stable" else self.unstable
                if stepsize_limit is not None:
                    points = clean_bigsteps(points, threshold=stepsize_limit)
                ax.plot(
                    points[:,0][:final_index],
                    points[:,1][:final_index],
                    fmt,
                    label=f"{dir} manifold",
                    color=colors[i],
                    markersize=markersize,
                    **kwargs,
                )

        return fig, ax
    
    def plot_manifold_copies(self, which='both', stepsize_limit=None, **kwargs):
        """
        plot the images of the manifolds as they appear arount the other islands
        of the chain, using the periodicity of the fixed points.
        """
        fig, ax, kwargs = create_canvas(**kwargs)

        colors = kwargs.pop("colors", ["green", "red"])
        markersize = kwargs.pop("markersize", 2)
        fmt = kwargs.pop("fmt", "-o")
        rm_points = kwargs.pop("rm_points", 0)
        final_index = -rm_points - 1

        number_of_copies = self.fixedpoint_1.m - 1
        for i, dir in enumerate(["stable", "unstable"]):
            if dir == which or which == "both":
                points = self.stable if dir == "stable" else self.unstable
                points = points[:final_index]
                for _ in range(number_of_copies):
                    points = self._map.f_many(1, points)
                    if stepsize_limit is not None: 
                        points = clean_bigsteps(points, threshold=stepsize_limit)
                    ax.plot(
                        points[:,0],
                        points[:,1],
                        fmt,
                        label=f"{dir} manifold",
                        color=colors[i],
                        markersize=markersize,
                        **kwargs,
                    )
    
    ### Homo/Hetero-clinic methods

    def find_N(self, eps_s: float, eps_u: float):
        """
        Find the number of times the map needs to be applied for the stable and unstable points to cross.

        This method evolves the initial stable :math:`x_s = x^\\star + \\varepsilon_s\\textbf{e}_s` and unstable :math:`x_u = x^\\star + \\varepsilon_u\\textbf{e}_u` points until they cross. They are alternatively evolved once and when the initial direction is reversed, the number of iterations is returned.

        Args:
            eps_s (float, optional): Initial :math:`\\varepsilon_s` along the stable manifold direction. Defaults to 1e-3.
            eps_u (float, optional): Initial :math:`\\varepsilon_u` along the unstable manifold direction. Defaults to 1e-3.

        Returns:
            tuple: A tuple containing two integers:
                - n_s (int): Number of iterations for the stable manifold.
                - n_u (int): Number of iterations for the unstable manifold.
        """
        r_s = self.rfp_s + eps_s * self.vector_s
        r_u = self.rfp_u + eps_u * self.vector_u

        first_dir = r_u - r_s
        last_norm = np.linalg.norm(first_dir)

        n_s, n_u = 0, 0
        success, stable_evol = False, True
        while not success:
            if stable_evol:
                r_s = self._map.f(-1 * self.fixedpoint_1.m, r_s)
                n_s += 1
            else:
                r_u = self._map.f(+1 * self.fixedpoint_1.m, r_u)
                n_u += 1
            stable_evol = not stable_evol

            norm = np.linalg.norm(r_u - r_s)
            # logger.debug(f"{np.dot(first_dir, r_u - r_s)} / {last_norm} / {norm}")
            if np.sign(np.dot(first_dir, r_u - r_s)) < 0:  # and last_norm < norm:
                success = True
            last_norm = norm

        # if not success:
        #     raise ValueError("Could not find N")
        return n_s, n_u

    def find_clinic_single(self, guess_eps_s, guess_eps_u, **kwargs):
        """
        Search a homo/hetero-clinic point.

        This function attempts to find the intersection point of the stable and unstable manifolds by iteratively adjusting the provided epsilon guesses using scipy root-finding algorithm.

        Args:
            guess_eps_s (float): Initial guess for the stable manifold epsilon.
            guess_eps_u (float): Initial guess for the unstable manifold epsilon.
            **kwargs: Additional keyword arguments.
                - root_args (dict): Arguments to pass to the root-finding function.
                - ERR (float): Error tolerance for verifying the linear regime (default: 1e-3).
                - n_s (int): Number of times the map needs to be applied for the stable manifold.
                - n_u (int): Number of times the map needs to be applied for the unstable manifold.

        Returns:
            tuple: A tuple containing the found epsilon values for the stable and unstable manifolds (eps_s, eps_u).
        """

        # Updating the default root finding parameters
        root_kwargs = {"jac": False}
        root_kwargs.update(kwargs.get("root_args", {}))
        use_jac = root_kwargs.get("jac")

        # Verifying that epsilons lie in linear regime
        ERR = kwargs.pop("ERR", 1e-3)

        if (
            self.error_linear_regime(
                guess_eps_s, self.rfp_s, self.vector_s, direction=-1
            )
            > ERR
        ):
            raise ValueError("Stable epsilon guess is not in the linear regime.")

        if (
            self.error_linear_regime(
                guess_eps_u, self.rfp_u, self.vector_u, direction=+1
            )
            > ERR
        ):
            raise ValueError("Unstable epsilon guess is not in the linear regime.")

        # Set the number of times the map needs to be applied (times the poloidal mode m)
        n_s, n_u = kwargs.pop("n_s", None), kwargs.pop("n_u", None)
        if n_s is None or n_u is None:
            n_s, n_u = self.find_N(guess_eps_s, guess_eps_u)
        logger.debug(f"Using n_s, n_u - {n_s}, {n_u}")

        # Logging the search initial configuration
        self._search_history = []

        # Evolution function for the root finding without and with jacobian
        def evolution_no_jac(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = self.rfp_s + eps_s * self.vector_s
            r_u = self.rfp_u + eps_u * self.vector_u

            r_s_evolved = self._map.f(-1 * n_s * self.fixedpoint_1.m, r_s)
            r_u_evolved = self._map.f(n_u * self.fixedpoint_1.m, r_u)

            return (r_s_evolved, r_u_evolved, r_s_evolved - r_u_evolved)

        def evolution_with_jac(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = self.rfp_s + eps_s * self.vector_s
            r_u = self.rfp_u + eps_u * self.vector_u

            jac_s = self._map.df(-1 * n_s * self.fixedpoint_1.m, r_s)
            r_s_evolved = self._map.f(-1 * n_s * self.fixedpoint_1.m, r_s)

            jac_u = self._map.df(n_u * self.fixedpoint_1.m, r_u)
            r_u_evolved = self._map.f(n_u * self.fixedpoint_1.m, r_u)

            return (
                r_s_evolved,
                r_u_evolved,
                r_s_evolved - r_u_evolved,
                np.array([jac_s @ self.vector_s, -jac_u @ self.vector_u]),
            )

        # Root finding
        def residual(logeps, n_s, n_u):
            eps_s, eps_u = np.exp(logeps)
            logger.debug(f"Current epsilon pair (eps_s, eps_u) : {eps_s, eps_u}")

            # if not in the fundamental segments then it should get back there, to work on maybe...

            if use_jac:
                _, _, diff, r_jac = evolution_with_jac([eps_s, eps_u], n_s, n_u)
                diff_jac = r_jac * np.array([eps_s, eps_u])
            else:
                _, _, diff = evolution_no_jac([eps_s, eps_u], n_s, n_u)

            logger.debug(f"Current difference : {diff}")
            self._search_history.append(diff)

            if use_jac:
                return diff, diff_jac
            else:
                return diff

        root_obj = root(
            residual,
            np.log([guess_eps_s, guess_eps_u]),
            args=(n_s, n_u),
            **root_kwargs,
        )

        # Checking status and logging the result
        logger.info(f"Root search status : {root_obj.message}")
        logger.debug(f"Root search object : {root_obj}")

        if not root_obj.success:
            raise ValueError("Homo/Heteroclinic search was not successful.")

        eps_s, eps_u = np.exp(root_obj.x)

        logger.info(
            f"Success! Found epsilon pair (eps_s, eps_u) : {eps_s:.3e}, {eps_u:.3e} gives a difference of {root_obj.fun}."
        )

        # Recording the homo/hetero-clinic point
        self.clinics.record_clinic(eps_s, eps_u, n_s, n_u)

        return eps_s, eps_u

    def find_clinics(
        self, first_guess_eps_s, first_guess_eps_u, n_points=None, **kwargs
    ):
        """
        Args:

        """
        shift = kwargs.pop("shift", 0)

        if n_points is None:
            n_points = 2 * self.fixedpoint_1.m

        # Reset the clinic search
        if kwargs.get("reset", True):
            self.clinics.reset()

        logger.info(
            f"Search {1}/{n_points} - initial guess for epsilon pair (eps_s, eps_u): {first_guess_eps_s, first_guess_eps_u}"
        )
        self.find_clinic_single(first_guess_eps_s, first_guess_eps_u, **kwargs)
        bounds = self.clinics.fundamental_segments

        stable_multiplicators = np.power(
            self.lambda_s, np.arange(n_points)[1:] / n_points
        )
        unstable_multiplicators = np.power(
            self.lambda_u, np.arange(n_points)[1:] / n_points
        )

        for i, (mult_s, mult_u) in enumerate(
            zip(stable_multiplicators, unstable_multiplicators)
        ):
            guess_i = [
                bounds["stable"][1] * mult_s,
                bounds["unstable"][0] * mult_u,
            ]

            logger.info(
                f"Search {i+2}/{n_points} - initial guess for epsilon pair (eps_s, eps_u): {guess_i}"
            )

            # Retrieve the
            n_s, n_u = self.clinics.nint_pair
            n_s += shift
            n_u += shift - 1

            self.find_clinic_single(*guess_i, n_s=n_s, n_u=n_u, **kwargs)

        # if len(self.onworking["clinics"]) != n_points:
        #     logger.warning("Number of clinic points is not the expected one.")

    def plot_clinics(self, **kwargs):
        """ """
        markers = kwargs.get(
            "markers", ["P", "o", "s", "p", "*", "X", "D", "d", "^", "v", "<", ">"]
        )
        color, edgecolor = kwargs.get("color", "royalblue"), kwargs.get(
            "edgecolor", "cyan"
        )

        fig, ax, kwargs = create_canvas(**kwargs)

        for i, clinic in enumerate(self.clinics):
            trajectory = clinic.trajectory
            ax.scatter(
                *trajectory.T,
                marker=markers[i],
                color=color,
                edgecolor=edgecolor,
                zorder=10,
            )

        return fig, ax

    ### Calculating Island/Turnstile Flux

    @property
    def turnstile_areas(self):
        if self._areas is None:
            self.compute_turnstile_areas()
        return self._areas

    def compute_turnstile_areas(self, **kwargs):
        if isinstance(self._map, maps.CylindricalBfieldSection):
            return self._turnstile_areas_cylbfieldsection(**kwargs)
        else:
            raise NotImplementedError(
                "Turnstile area computation only implemented for CylindricalBfieldSection for now."
            )

    def _turnstile_areas_cylbfieldsection(self, n_joining=100):
        """
        Compute the turnstile area by integrating the vector potential along the trajectory of the homo/hetero-clinics points.
        """
        lagrangian_values = np.NaN * np.zeros(len(self.clinics) + 1)

        # Could be put in the clinic trajectory directly. Open question.
        # Calculation of the lagrangian value (from the unstable to the stable fundamental segment)
        for i, clinic in enumerate([*self.clinics, self.clinics[0]]):
            x_s_0, x_u_0 = (
                clinic.trajectory[-1 if i != 0 else -2, :],
                clinic.trajectory[0 if i != len(self.clinics) else 1, :],
            )

            n_bwd = clinic.nint_s if i != 0 else clinic.nint_s - 1
            n_fwd = clinic.nint_u if i != len(self.clinics) else clinic.nint_u - 1

            # x_t_s = self._map.f(-n_bwd * self.fixedpoint_1.m, x_s_0)
            # x_t_u = self._map.f(n_fwd * self.fixedpoint_1.m, x_u_0)

            intA_s = self._map.lagrangian(x_s_0, -n_bwd * self.fixedpoint_1.m)
            intA_u = self._map.lagrangian(x_u_0, n_fwd * self.fixedpoint_1.m)

            lagrangian_values[i] = intA_u - intA_s

            logger.info(
                f"Lagrangian value obtained ({(lagrangian_values[i]):.3e}) for homo/hetero-clinic trajectory (eps_s, eps_u) : {clinic.eps_s, clinic.eps_u}"
            )

        # Computation of the turnstile area
        areas = np.empty(len(self.clinics))
        shifted_indices = [
            i for i in np.roll(np.arange(len(self.clinics), dtype=int), -1)
        ]

        # Loop on the L values : L_h current clinic point, L_m next clinic point (in term of >_u ordering)
        for i, shifted_i in enumerate(shifted_indices):
            # Area is the difference in the
            areas[i] = lagrangian_values[i] - lagrangian_values[i + 1]

            # Closure by joining integrals
            traj_h = self.clinics[i].trajectory
            traj_m = self.clinics[shifted_i].trajectory

            # Get the correct points to join
            r_h_u, r_m_u = traj_h[0, :], traj_m[0, :]
            r_h_s, r_m_s = traj_h[-1, :], traj_m[-1, :]
            if i == 0:
                r_h_s, r_m_s = traj_h[-2, :], traj_m[-1, :]
            elif i == len(shifted_indices) - 1:
                r_h_u, r_m_u = traj_h[0, :], traj_m[1, :]

            for j, (rA, rB) in enumerate(zip([r_m_u, r_h_s], [r_h_u, r_m_s])):
                # Create a segment between r1 and r2
                gamma, dl = np.linspace(rA, rB, n_joining, retstep=True)

                # Evaluate A at the middle point between (x_i, x_{i+1})
                mid_gamma = (gamma + dl / 2)[:-1]
                mid_gamma = np.vstack(
                    (
                        mid_gamma[:, 0],
                        self._map.phi0 * np.ones(mid_gamma.shape[0]),
                        mid_gamma[:, 1],
                    )
                ).T
                mid_A = np.array([self._map._mf.A(r)[0::2] for r in mid_gamma])

                # Discretize the A.dl integral and sum it
                closing_integral = np.einsum(
                    "ij,ij->i", mid_A, np.ones((mid_A.shape[0], 1)) * dl
                ).sum()
                logger.debug(f"Closing integral {i+1}, {j+1}/2 : {closing_integral}")
                areas[i] += closing_integral

        self._areas = areas
        self._lagrangian_values = lagrangian_values

        return areas

    ### Integration methods

    def integrate(self, x_many, nintersect, direction=1):
        """
        Integrate a set of points x_many for nintersect times in the direction specified.
        Robust to integration failures and has fixed return shape. 

        Returns an array of shape (nintersect, len(x_many), _map.dimension).
        """

        x_many = np.atleast_2d(x_many)
        t = self.fixedpoint_1.m * direction
        res = []
        res.append(x_many)
        for _ in range(nintersect):
            x_many = self._map.f_many(t, x_many)
            res.append(x_many)
        return np.array(res)
    

#        # used to return [dimension*len(x_many) , nintersect + 1]
#        x_path = np.full((self._map.dimension * x_many.shape[0], nintersect + 1), np.nan)
#        x_path[:, 0] = x_many.flatten()
#
#        t = self.fixedpoint_1.m * direction
#
#        for i, x in enumerate(x_many):
#            for j in range(nintersect):
#                try:
#                    x_new = self._map.f(t, x)
#                except:
#                    logger.error(f"Integration of point {x} failed.")
#                    break
#
#                x_path[2 * i : 2 * i + self._map.dimension, j + 1] = x_new
#                x = x_new
#
#        #return x_path
#
#    
#