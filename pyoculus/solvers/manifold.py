import pyoculus.maps as maps
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..utils.plot import create_canvas
from scipy.optimize import root, minimize

# from functools import total_ordering
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(__name__)

def eig(jacobian):
    """
    Compute the stable and unstable eigenvalues and eigenvectors of the fixed point.
    
    Args:
        jacobian (np.array): Jacobian matrix of the fixed point.

    Returns:
        tuple: A tuple containing four elements: 
            - lambda_s (float): The stable eigenvalue.
            - vector_s (np.array): The stable eigenvector.
            - lambda_u (float): The unstable eigenvalue.
            - vector_u (np.array): The unstable eigenvector.
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

class Manifold(BaseSolver):
    """
    
    
    """

    def __init__(
        self,
        map : maps.base_map,
        fixedpoint_1 : FixedPoint,
        fixedpoint_2 : FixedPoint = None
    ):
        """
        Initialize the Manifold class.

        Args:
            map (maps.base_map): The map to use for the computation.
            fixedpoint_1 (FixedPoint): first fixed point
            fixedpoint_2 (FixedPoint, optional): second fix point if not homoclinic manifold. Defaults to None.
        """

        # Check that the fixed points are correct FixedPoint instances
        if not isinstance(fixedpoint_1, FixedPoint):
            raise TypeError("Fixed point must be an instance of FixedPoint class")
        if not fixedpoint_1.successful:
            raise ValueError(
                "Need a successful fixed point to compute the manifold"
            )

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

        self._lfs = {"stable": None, "unstable": None}

        # Initialize the BaseSolver
        super().__init__(map)

    def choose(self, dir_1, dir_2, is_first_stable):
        """
        Choose the stable and unstable directions.
        
        Args:
            dir_1 (str): '+' or '-' for the stable direction.
            dir_2 (str): '+' or '-' for the unstable direction.
            first_stable (bool): Whether the first fixed point is a stable direction.
        """

        if is_first_stable:
            fp_1, fp_2 = self.fixedpoint_1, self.fixedpoint_2
        else:
            fp_1, fp_2 = self.fixedpoint_2, self.fixedpoint_1

        signs = [(-1)**int(d=='+') for d in dir_1 + dir_2]

        # Choose the fixed points and their directions
        rfp_1 = fp_1.coords[0]
        p1_lambda_s, p1_vector_s, _, _ = eig(
            fp_1.jacobians[0]
        )

        rfp_2 = fp_2.coords[0]
        _, _, p2_lambda_u, p2_vector_u = eig(
            fp_2.jacobians[0]
        )
        
        # Assign the fixed points and their directions
        self.rfp_s, self.lambda_s, self.vector_s = (
            rfp_1,
            p1_lambda_s,
            signs[0] * p1_vector_s,
        )
        self.rfp_u, self.lambda_u, self.vector_u = (
            rfp_2,
            p2_lambda_u,
            signs[3] * p2_vector_u,
        )

    
    @classmethod
    def show_directions(cls, fp_1, fp_2, **kwargs):
        """
        Helper function to plot the fixed points and their stable and unstable direction. Usefull to look at which direction need to be considered for the inner and outer manifolds before creating a class and analyzed them.

        Args:
            fp_1 (FixedPoint): First fixed point.
            fp_2 (FixedPoint): Second fixed point.
            **kwargs: Additional optional keyword arguments.
                - pcolors (list): Colors for the fixed points.
                - vcolors (list): Colors for the eigenvectors.
                - vscale (int): Scale for the eigenvectors (default is 18).
                - dvtext (float): Additional distance for the text as a fraction of the eigenvector (default is 0.005).

        Returns:
            tuple: A tuple containing the figure and the axis.
        """

        # Defaults 
        pcolors = kwargs.get("pcolors", ['tab:blue', 'tab:orange'])
        vcolors = kwargs.get("vcolors", ['tab:green', 'tab:red'])
        vscale = kwargs.get("scale", 18)
        dvtext = kwargs.get("dvtext", 0.005)

        # Set the figure and ax
        fig, ax, kwargs = create_canvas(**kwargs)
        
        # Choose the fixed points and their directions
        rfp_1 = fp_1.coords[0]
        _, p1_vector_s, _, p1_vector_u = eig(
            fp_1.jacobians[0]
        )
        rfp_2 = fp_2.coords[0]
        _, p2_vector_s, _, p2_vector_u = eig(
            fp_2.jacobians[0]
        )

        # Plot the fixed points
        ax.scatter(*rfp_1, marker='X', s=100, label='Fixed point 1', zorder=999, 
                   color=pcolors[0], edgecolor='black', linewidth=1)
        ax.scatter(*rfp_2, marker='X', s=100, label='Fixed point 2', zorder=999, 
                   color=pcolors[1], edgecolor='black', linewidth=1)
        # ax.text(*(rfp_1), '1', zorder=999, ha='center', va='center')
        # ax.text(*(rfp_2), '2', zorder=999, ha='center', va='center')

        # Plot the eigenvectors
        def plot_eigenvectors(ax, rfp, vectors):
            for vector, color in zip(vectors, vcolors):
                # Positive and negative arrows
                p_arrow = FancyArrowPatch(rfp, rfp + vector / vscale, arrowstyle='-|>', color=color, mutation_scale=10)
                n_arrow = FancyArrowPatch(rfp, rfp - vector / vscale, arrowstyle='-|>', color=color, mutation_scale=10)
                
                # Add the arrows to the plot
                ax.add_patch(p_arrow)
                ax.add_patch(n_arrow)

                # Add the text
                ax.text(*(rfp + vector * (dvtext + 1 / vscale)), '+', zorder=999, color=color, 
                        fontsize='large', fontweight='bold', ha='center', va='center')
                ax.text(*(rfp - vector * (dvtext + 1 / vscale)), '-', zorder=999, color=color, 
                        fontsize='large', fontweight='bold', ha='center', va='center')

        plot_eigenvectors(ax, rfp_1, [p1_vector_s, p1_vector_u])
        plot_eigenvectors(ax, rfp_2, [p2_vector_s, p2_vector_u])

        return fig, ax

    def error_linear_regime(self, epsilon, rfp, eigenvector, direction=1):
        """
        Metric to evaluate if the point rfp + epsilon * eigenvector is in the linear regime of the fixed point.
        """
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[:, 1] - rz_path[:, 0]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

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
        eps_dir = rz_path[:, 1] - rz_path[:, 0]
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
        return Rs, Zs

    def find_epsilon(self, rfp, eigenvector, eps_guess=1e-3, direction=1):
        """Find the epsilon that lies in the linear regime."""
        find_eps = lambda x: self.error_linear_regime(
            x, rfp, eigenvector, direction=direction
        )
        minobj = minimize(find_eps, eps_guess, bounds=[(0, 1)], tol=1e-12)

        if not minobj.success:
            logger.warning(
                "Search for minimum of the linear error failed, using the guess for epsilon."
            )
            return eps_guess
        else:
            esp_root = minobj.x[0]
            logger.info(
                f"Search for minimum of the linear error succeeded, epsilon = {esp_root:.5e}"
            )
            return esp_root
    
    def compute_manifold(self, eps, compute_stable, **kwargs):
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
        # Set the right parameters
        mname = "stable" if compute_stable else "unstable"
        if compute_stable:
            rfp, vector, lambda_, goes = self.rfp_s, self.vector_s, self.lambda_s, -1
        else:
            rfp, vector, lambda_, goes = self.rfp_u, self.vector_u, self.lambda_u, 1

        # Setup the epsilon such that the first point is at rfp + eps * vector
        eps = kwargs.get("eps", None)

        # If the epsilon is not given, find the best one in the linear regime
        eps_guess = kwargs.get("eps_guess", 1e-3)
        if eps is None:
            eps = self.find_epsilon(
                rfp, vector, eps_guess, goes
            )

        # Compute the starting configuration and the manifold
        neps, nint = kwargs.get("neps", 20), kwargs.get("nint", 10)
        RZs = self.start_config(
            eps, rfp, lambda_, vector, neps, goes
        )
        logger.info(f"Computing {mname} manifold...")
        self._lfs[mname] = self.integrate(
            RZs, nintersect=nint, direction=goes
        )

        return self._lfs[mname]

    @property
    def stable(self):
        if self._lfs["stable"] is None:
            logger.info("Stable manifold not computed. Using the computation method.")
            self.compute_manifold(1e-3, True)
        return self._lfs["stable"]
    
    @property
    def unstable(self):
        if self._lfs["unstable"] is None:
            logger.info("Unstable manifold not computed. Using the computation method.")
            self.compute_manifold(1e-3, False)
        return self._lfs["unstable"]

    def compute(self, eps_s, eps_u, **kwargs):
        """
        Computation of the stable and unstable manifolds.

        Args:
            eps_s (float): epsilon in the stable direction
            eps_u (float): epsilon in the unstable direction    
        
        Keyword Args:
            eps_guess_s (float): guess for epsilon in the stable direction (if eps_s is not given)
            eps_guess_u (float): guess for epsilon in the unstable direction (if eps_u is not given)
            neps (int): number of points in the starting configuration
        """
        # Extract the keyword arguments
        kwargs_s = {key: kwargs.pop(key) for key in ["eps_guess_s", "neps_s", "nint_s"] if key in kwargs}
        kwargs_u = {key: kwargs.pop(key) for key in ["eps_guess_u", "neps_u", "nint_u"] if key in kwargs}
        if kwargs:
            logger.warning(f"Unused keyword arguments: {kwargs}")

        # Compute the manifolds
        self.compute_manifold(eps_s, True, **kwargs_s)
        self.compute_manifold(eps_u, False, **kwargs_u)

        return self._lfs["stable"], self._lfs["unstable"]


    def plot(self, which="both", **kwargs):
        """
        
        """
        fig, ax, kwargs = create_canvas(**kwargs)
        
        colors = kwargs.pop("colors", ["green", "red"])
        markersize = kwargs.pop("markersize", 2)
        fmt = kwargs.pop("fmt", "-o")

        for i, dir in enumerate(["stable", "unstable"]):
            if dir == which or which == "both":
                points = self._lfs[dir]
                points = points.T.flatten()
                ax.plot(
                    points[::2],
                    points[1::2],
                    fmt,
                    label=f"{dir} manifold",
                    color=colors[i],
                    markersize=markersize,
                    **kwargs,
                )

        return fig, ax

    ### Homo/Hetero-clinic methods

    def _order(self, clinic):
        fund = self.onworking["fundamental_segment"][1]
        rfp_u = self.onworking["rfp_u"]

        norm = np.linalg.norm(clinic - rfp_u)

        r_ev = clinic
        max_iterations = 20  # Set a maximum number of iterations
        for _ in range(max_iterations):
            if fund[0] <= norm < fund[1]:
                return norm
            logger.debug(f"norm = {norm}")
            r_ev = self._map.f(-1*self.fixedpoint_1.m, r_ev)
            norm = np.linalg.norm(r_ev - rfp_u)

        raise ValueError(
            "Failed to find a solution within the maximum number of iterations"
        )

    def order(self):
        """Order the homo/hetero-clinic points with the induced linear ordering of the unstable manifold >_u."""
        self.onworking["clinics"] = [
            self.onworking["clinics"][i] for i in np.argsort([x[0] for x in self.onworking["clinics"]])
        ]


    def _fundamental_segment(self, eps_s : float, eps_u : float):
        """
        Calculate the fundamental segment on the unstable and stable manifolds.

        Args:
            eps_s (float): Initial :math:`\\varepsilon_s` along the stable manifold direction.
            eps_u (float): Initial :math:`\\varepsilon_u` along the unstable manifold direction.

        Returns:
            tuple: A tuple containing two tuples:
                - (eps_s, upperbound_s): The initial :math:`\\varepsilon_s` and the computed upper bound 
                for the stable manifold.
                - (eps_u, upperbound_u): The initial guess and the computed upper bound 
                for the unstable manifold.
        """

        r_s = self.onworking["rfp_s"] + eps_s * self.onworking["vector_s"]
        r_u = self.onworking["rfp_u"] + eps_u * self.onworking["vector_u"]

        r_s_evolved = self._map.f(-1*self.fixedpoint_1.m, r_s)
        r_u_evolved = self._map.f(1*self.fixedpoint_1.m, r_u)

        upperbound_s = np.linalg.norm(r_s_evolved - self.onworking["rfp_s"])
        upperbound_u = np.linalg.norm(r_u_evolved - self.onworking["rfp_u"])

        bounds = ((eps_s, upperbound_s), (eps_u, upperbound_u))
        return bounds

    def find_N(self, eps_s : float = 1e-3, eps_u : float = 1e-3):
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
        r_s = self.onworking["rfp_s"] + eps_s * self.onworking["vector_s"]
        r_u = self.onworking["rfp_u"] + eps_u * self.onworking["vector_u"]

        first_dir = r_u - r_s
        last_norm = np.linalg.norm(first_dir)

        n_s, n_u = 0, 0
        success, stable_evol = False, True
        while not success:
            if stable_evol:
                r_s = self._map.f(-1*self.fixedpoint_1.m, r_s)
                n_s += 1
            else:
                r_u = self._map.f(1*self.fixedpoint_1.m, r_u)
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

    def find_clinic_single(self, guess_eps_s=None, guess_eps_u=None, **kwargs):
        """Find the homo/hetero-clinic points (intersection of the stable and unstable manifold)."""
        defaults = {
            "n_s": None,
            "n_u": None,
            "bounds": None,
            "root": {"method": "hybr", "jac": False},
        }
        defaults.update(
            {key: value for key, value in kwargs.items() if key in defaults}
        )
        defaults["root"].update(
            {key: value for key, value in kwargs.items() if key not in defaults}
        )
        ERR = 1e-3

        # Initializing the lower bound / Verifying that epsilon lies in linear regime
        if guess_eps_s is None:
            eps_s_lb = self.find_epsilon(self.onworking["rfp_s"], self.onworking["vector_s"], direction=-1)
        elif (
            self.error_linear_regime(
                guess_eps_s, self.onworking["rfp_s"], self.onworking["vector_s"], direction=-1
            )
            > ERR
        ):
            raise ValueError("Guess for stable epsilon is not in the linear regime.")
        else:
            eps_s_lb = guess_eps_s

        if guess_eps_u is None:
            eps_u_lb = self.find_epsilon(self.onworking["rfp_u"], self.onworking["vector_u"])
        elif self.error_linear_regime(guess_eps_u, self.onworking["rfp_u"], self.onworking["vector_u"]) > ERR:
            raise ValueError("Guess for unstable epsilon is not in the linear regime.")
        else:
            eps_u_lb = guess_eps_u

        # Find the bounds of the search domain : lower bound epsilons are map to upper bound epsilons after one iteration
        if defaults["bounds"] is None:
            defaults["bounds"] = self._fundamental_segment(eps_s_lb, eps_u_lb)
        if guess_eps_s is None:
            guess_eps_s = (defaults["bounds"][0][1] - defaults["bounds"][0][0]) / 2
        if guess_eps_u is None:
            guess_eps_u = (defaults["bounds"][1][1] - defaults["bounds"][1][0]) / 2

        # Initialize the number of times the map needs to be applied
        if defaults["n_s"] is None or defaults["n_u"] is None:
            n_s, n_u = self.find_N(guess_eps_s, guess_eps_u)
        else:
            n_s, n_u = defaults["n_s"], defaults["n_u"]

        # Logging the search initial configuration
        logger.debug(f"Guess - {guess_eps_s}, {guess_eps_u}")
        logger.debug(f"Bounds - {defaults['bounds']}")
        logger.debug(f"n_s, n_u - {n_s}, {n_u}")

        self.onworking["history"] = []

        # Residual function for the root finding
        def evolution(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = self.onworking["rfp_s"] + eps_s * self.onworking["vector_s"]
            r_u = self.onworking["rfp_u"] + eps_u * self.onworking["vector_u"]

            try:
                if defaults['root']['jac']:
                    jac_s = self._map.df(-1*n_s*self.fixedpoint_1.m, r_s)
                r_s_evolved = self._map.f(-1*n_s*self.fixedpoint_1.m, r_s)
            except Exception as e:
                logger.error(f"Error in stable manifold integration : {e}")
                raise e
                # breakpoint()

            try:
                if defaults['root']['jac']:
                    jac_u = self._map.df(n_u*self.fixedpoint_1.m, r_u)
                r_u_evolved = self._map.f(n_u*self.fixedpoint_1.m, r_u)
            except Exception as e:
                logger.error(f"Error in unstable manifold integration : {e}")
                raise e
                # breakpoint()

            if defaults['root']['jac']:
                return (
                    r_s_evolved,
                    r_u_evolved,
                    r_s_evolved - r_u_evolved,
                    np.array([jac_s @ self.onworking["vector_s"], -jac_u @ self.onworking["vector_u"]]),
                )
            else:
                return (
                    r_s_evolved,
                    r_u_evolved,
                    r_s_evolved - r_u_evolved
                )

        def residual(logeps, n_s, n_u):
            eps_s, eps_u = np.exp(logeps)
            logger.debug(f'Inside : {eps_s, eps_u}')
            # if not defaults['bounds'][0][0] <= eps_s <= defaults['bounds'][0][1] or not defaults['bounds'][1][0] <= eps_u <= defaults['bounds'][1][1]:
            #     dist_s = min(abs(eps_s - defaults['bounds'][0][0]), abs(eps_s - defaults['bounds'][0][1]))
            #     dist_u = min(abs(eps_u - defaults['bounds'][1][0]), abs(eps_u - defaults['bounds'][1][1]))
            #     coef = 1+10**(np.log(dist_s+dist_u)/np.log(defaults['bounds'][0][1]+defaults['bounds'][1][1]))
            #     # ret = coef*evolution([min(max(eps_s, defaults['bounds'][0][0]), defaults['bounds'][0][1]), min(max(eps_u, defaults['bounds'][1][0]), defaults['bounds'][1][1])], n_s, n_u)[2]
            #     ret = (np.array([dist_s, dist_u])/defaults['bounds'][0][0])**2
            #     logger.debug(f"Outside : {eps_s, eps_u} - {ret[:3]}")
            #     return ret
            # else:
            ret = evolution([eps_s, eps_u], n_s, n_u)
            self.onworking["history"].append(np.array([[eps_s, eps_u], *ret]))
            
            if defaults['root']['jac']:
                ret[3][:, 0] *= eps_s
                ret[3][:, 1] *= eps_u
                logger.debug(f"Returns - {ret[:3]}")
            else:
                logger.debug(f"Returns - {ret}")
                
            if defaults['root']['jac']:
                return ret[2], ret[3]
            else:
                return ret[2]

        r = root(
            residual,
            np.log([guess_eps_s, guess_eps_u]),
            args=(n_s, n_u),
            **defaults["root"],
        )

        logger.info(f"Root finding status : {r.message}")
        logger.debug(f"Root finding object : {r}")

        if not r.success:
            raise ValueError("Homoclinic search not successful.")

        eps_s, eps_u = np.exp(r.x)
        # if self.error_linear_regime(eps_s, self.rfp_s, self.vector_s, direction=-1) > 1e-4:
        #     raise ValueError("Homoclinic point stable epsilon does not lie linear regime.")
        # if self.error_linear_regime(eps_u, self.rfp_u, self.vector_u) > 1e-4:
        #     raise ValueError("Homoclinic point unstable epsilon does not lie linear regime.")

        logger.info(
            f"Eps_s : {eps_s:.3e}, Eps_u : {eps_u:.3e} gives a difference in endpoint [R,Z] : {r.fun}"
        )

        # Recording the homo/hetero-clinic point
        r_s_ev, r_u_ev = evolution([eps_s, eps_u], n_s, n_u)[:2]
        if not self.onworking["clinics"]:
            self.onworking["fundamental_segment"] = self._fundamental_segment(eps_s, eps_u)
            order = eps_u
            self.onworking["find_clinic_configuration"] = {"n_s": n_s, "n_u": n_u}
        else:
            order = self._order(self.onworking["rfp_u"] + eps_u * self.onworking["vector_u"])

        if not np.any([np.isclose(order, other[0], rtol=1e-2) for other in self.onworking["clinics"]]):
            self.onworking["clinics"].append((order, eps_s, eps_u, r_s_ev, r_u_ev))
        else:
            logger.warning("Clinic already recorded, skipping...")

        return eps_s, eps_u

    def find_clinics(self, indices = None, n_points = 1, m = 0, **kwargs):
        if len(self.onworking["clinics"]) == 0:
            self.find_clinic_single(**kwargs)
        
        bounds_0 = self.onworking["fundamental_segment"]

        for key in ['n_s', 'n_u']:
            if key in kwargs:
                kwargs.pop(key)

        if indices is None:
            indices = range(1, n_points)

        for i in indices:
            bounds_i = np.array(bounds_0)
            bounds_i[0][0] = self.onworking["clinics"][-1][1]
            bounds_i[1][1] = self.onworking["clinics"][-1][2]
            bounds_i = (tuple(bounds_i[0]), tuple(bounds_i[1]))

            guess_i = [
                bounds_0[0][1] * np.power(self.onworking["lambda_s"], i / n_points),
                bounds_0[1][0] * np.power(self.onworking["lambda_u"], i / n_points),
            ]
            logger.info(f"Initial guess: {guess_i}")

            n_s = self.onworking["find_clinic_configuration"]["n_s"] + m
            n_u = self.onworking["find_clinic_configuration"]["n_u"] - m - 1
            self.find_clinic_single(
                *guess_i, bounds=bounds_i, n_s=n_s, n_u=n_u, **kwargs
            )
        
        if len(self.onworking["clinics"]) != n_points:
            logger.warning("Number of clinic points is not the expected one.")

        self.order()

    def plot_clinics(self, **kwargs):
        marker = ["P", "o", "s", "p", "P", "*", "X", "D", "d", "^", "v", "<", ">"]

        fig, ax, kwargs = create_canvas(**kwargs)
        
        for i, clinic in enumerate(self.onworking["clinics"]):
            eps_s, eps_u = clinic[1:3]
            n_s, n_u = self.onworking["find_clinic_configuration"].values()
            rfp_s, rfp_u = self.onworking["rfp_s"], self.onworking["rfp_u"]
            vec_s, vec_u = self.onworking["vector_s"], self.onworking["vector_u"]

            hs_i = self.integrate(rfp_s + eps_s * vec_s, n_s, -1)
            hu_i = self.integrate(rfp_u + eps_u * vec_u, n_u, 1)
            ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10)
            ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10)

        return fig, ax

    ### Calculating Island/Turnstile Flux

    # def resonance_area(self):
    #     pass

    def turnstile_area(self, n_joining = 100):
        """Compute the turnstile area by integrating the vector potential along the trajectory of the homo/hetero-clinics points.
        """

        if not isinstance(self._map, maps.CylindricalBfieldSection):
            raise NotImplementedError("Turnstile area computation only implemented for CylindricalBfieldSection")

        # Function for forward/backward integration for each clinic point
        def integrate_direction(rz, n, rfp, direction):
            history = []
            intA = []
            n_tmp = 0

            while n_tmp < n:
                intA_tmp = self._map.lagrangian(rz, direction)
                rz_end = self._map.f(direction, rz)

                if n_tmp > 3 and np.linalg.norm(
                    rz_end - rfp
                ) > np.linalg.norm(rz - rfp):
                    if direction == -1:
                        direction_str = "Backward"
                    else:
                        direction_str = "Forward"
                    logger.info(f"{direction_str} integration goes beyond stable saddle point.")
                    logger.debug(
                        f"rfp: {rfp}, rz_end: {rz_end}, rz: {rz}"
                    )
                    break

                rz = rz_end
                history.append(rz)
                intA.append(intA_tmp)
                n_tmp += 1
            
            return history, intA

        # Potential integration
        n_fwd, n_bwd = self.onworking["find_clinic_configuration"].values()
        potential_integrations = []
        history = []
        for i, clinic in enumerate(self.onworking["clinics"]):
            # Forward integration
            rz_forward = clinic[-2]
            history_forward, intA_forward = integrate_direction(rz_forward, n_fwd, self.onworking["rfp_s"], 1)

            # Backward integration
            # taking the point found from unstable manifold to go back to the fixedpoint
            rz_backward = clinic[-1]
            history_backward, intA_backward = integrate_direction(rz_backward, n_bwd, self.onworking["rfp_u"], -1)

            logger.info(
                f"Potential integration completed for homo/hetero-clinic point of order : {clinic[0]:.3e}"
            )

            if i == 0:
                n_bwd -= 1

            potential_integrations.append(
                [np.array(intA_forward), np.array(intA_backward)]
            )
            history.append([history_forward, history_backward])

        # Computation of the turnstile area
        areas = np.zeros(len(potential_integrations))
        err_by_diff, err_by_estim = np.zeros_like(areas), np.zeros_like(areas)

        # Loop on the intA values : intA_h current clinic point, intA_m next clinic point (in term of >_u ordering)
        for i, (intA_h, intA_m) in enumerate(zip(
            potential_integrations,
            [
                potential_integrations[i]
                for i in np.roll(np.arange(len(potential_integrations), dtype=int), -1)
            ],
        )):
            # Set up the maximum number of fwd/bwd usable iterations
            n_fwd = min(intA_h[0].size, intA_m[0].size)
            n_bwd = min(intA_h[1].size, intA_m[1].size)

            # Action integration
            intm = intA_m[0][:n_fwd].sum() - intA_m[1][:n_bwd].sum()
            inth = intA_h[0][:n_fwd].sum() - intA_h[1][:n_bwd].sum()
            areas[i] = intm - inth

            # Closure by joining integrals
            for j, n in enumerate([n_fwd, n_bwd]):
                r1 = history[i][j][n-1]
                r2 = history[(i+1)%len(potential_integrations)][j][n-1]
                
                # Create a segment between r2 and r1
                gamma, dl = np.linspace(r1, r2, n_joining, retstep=True)  

                # Evaluate A at the middle point between (x_i, x_{i+1})
                mid_gamma = (gamma + dl/2)[:-1]
                mid_gamma = np.vstack((mid_gamma[:,0], self._map.phi0*np.ones(mid_gamma.shape[0]), mid_gamma[:,1])).T
                mid_A = np.array([self._map._mf.A(r)[0::2] for r in mid_gamma])
                # else:
                #     mid_A = np.empty((mid_gamma.shape[0], 2))
                #     for k, r in enumerate(mid_gamma):
                #         xyz = np.array([
                #             r[0] * np.cos(r[1]),
                #             r[0] * np.sin(r[1]),
                #             r[2]
                #         ])
                #         invJacobian = self._problem._inv_Jacobian(r[0], r[1], r[2])
                #         mid_A[k] = np.matmul(invJacobian, np.array([self._problem.A(xyz)]).T).T[0][::2]

                # Discretize the A.dl integral and sum it
                areas[i] += np.einsum('ij,ij->i', mid_A, np.ones((mid_A.shape[0], 1)) * dl).sum()

        self.onworking["areas"] = np.vstack((areas, err_by_diff, err_by_estim)).T
        self.onworking["potential_integrations"] = potential_integrations
        self.onworking["clinic_history"] = history

        return areas

    ### Integration methods

    def integrate(self, x_many, nintersect, direction=1):
        """
        
        """
        
        x_many = np.atleast_2d(x_many)
        
        x_path = np.zeros((self._map.dimension * x_many.shape[0], nintersect + 1))
        x_path[:, 0] = x_many.flatten()

        t = self.fixedpoint_1.m * direction

        for i, x in enumerate(x_many):
            for j in range(nintersect):
                try:
                    x_new = self._map.f(t, x)
                except:
                    logger.error(f"Integration of point {x} failed.")
                    break
                
                x_path[2 * i : 2 * i + self._map.dimension, j + 1] = x_new
                x = x_new

        return x_path