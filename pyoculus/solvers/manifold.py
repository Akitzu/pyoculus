import pyoculus.maps as maps
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..utils.plot import create_canvas
from scipy.optimize import root, minimize

# from functools import total_ordering
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(__name__)


class Manifold(BaseSolver):
    def __init__(
        self,
        map : maps.base_map,
        fixedpoint_1 : FixedPoint,
        fixedpoint_2 : FixedPoint = None
    ):
        """
        
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

        self._is_self_intersection = False
        if fixedpoint_2 is not None:
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_2

            # Initialize the inner/outer dictionnaries
            self.outer = dict()
            self.inner = dict()

            # Initialize the manifold for later computations
            self.outer['lfs'] = {"stable": None, "unstable": None}
            self.inner['lfs'] = {"stable": None, "unstable": None}

            # Set the hetero/homo-clinic lists
            self.outer['clinics'] = []
            self.inner['clinics'] = []
        else:
            self._is_self_intersection = True
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_1

            # Same as for outer/inner but in the case of self intersection
            self.records = dict()
            self.records['lfs'] = {"stable": None, "unstable": None}
            self.records['clinics'] = []

        # Initialize the BaseSolver
        super().__init__(map)

    @staticmethod
    def eig(jacobian):
        """Compute the eigenvalues and eigenvectors of the jacobian and returns them in the order : stable, unstable."""
        eigRes = np.linalg.eig(jacobian)
        eigenvalues = np.abs(eigRes[0])

        # Eigenvectors are stored as columns of the matrix eigRes[1], transposing it to access them as np.array[i]
        eigenvectors = eigRes[1].T
        s_index, u_index = 0, 1
        if eigenvalues[0].real > eigenvalues[1].real:
            s_index, u_index = 1, 0

        return (
            eigenvalues[s_index],
            eigenvectors[s_index],
            eigenvalues[u_index],
            eigenvectors[u_index],
        )

    def choose(self, signs, order=True):
        """Choose the two fixed points and their stable or unstable directions."""

        # Choose the 1st/2nd fixedpoint as inner/outer stable
        if order:
            fp_1, fp_2 = self.fixedpoint_1, self.fixedpoint_2
        else:
            fp_1, fp_2 = self.fixedpoint_2, self.fixedpoint_1

        # Choose the fixed points and their directions
        rfp_1 = fp_1.coords[0]
        p1_lambda_s, p1_vector_s, p1_lambda_u, p1_vector_u = self.eig(
            fp_1.jacobians[0]
        )

        rfp_2 = fp_2.coords[0]
        p2_lambda_s, p2_vector_s, p2_lambda_u, p2_vector_u = self.eig(
            fp_2.jacobians[0]
        )

        # Inner difrection
        self.inner['rfp_s'], self.inner['lambda_s'], self.inner['vector_s'] = (
            rfp_1,
            p1_lambda_s,
            signs[0][0] * p1_vector_s,
        )
        self.inner['rfp_u'], self.inner['lambda_u'], self.inner['vector_u'] = (
            rfp_2,
            p2_lambda_u,
            signs[1][0] * p2_vector_u,
        )

        # Outter direction
        self.outer['rfp_s'], self.outer['lambda_s'], self.outer['vector_s'] = (
            rfp_2,
            p2_lambda_s,
            signs[1][1] * p2_vector_s,
        )
        self.outer['rfp_u'], self.outer['lambda_u'], self.outer['vector_u'] = (
            rfp_1,
            p1_lambda_u,
            signs[0][1] * p1_vector_u,
        )
        
    def show_directions(self, **kwargs):

        fig, ax, kwargs = create_canvas(**kwargs)
        
        # Plot the fixed points
        ax.scatter(*self.inner['rfp_s'], color='blue')
        ax.scatter(*self.inner['rfp_u'], color='blue')

        # Plot the eigenvectors
        Q_inner_stable = ax.quiver(*self.inner['rfp_s'], *self.inner['vector_s'], color='green')
        Q_inner_unstable = ax.quiver(*self.inner['rfp_u'], *self.inner['vector_u'], color='red')
        Q_outer_stable = ax.quiver(*self.outer['rfp_s'], *self.outer['vector_s'], color='green')
        Q_outer_unstable = ax.quiver(*self.outer['rfp_u'], *self.outer['vector_u'], color='red')

        # Convert the start point of the quiver from data coordinates to axes coordinates
        x_inner_stable, y_inner_stable = ax.transAxes.inverted().transform(ax.transData.transform(self.inner['rfp_s'][:2]))
        x_inner_unstable, y_inner_unstable = ax.transAxes.inverted().transform(ax.transData.transform(self.inner['rfp_u'][:2]))
        x_outer_stable, y_outer_stable = ax.transAxes.inverted().transform(ax.transData.transform(self.outer['rfp_s'][:2]))
        x_outer_unstable, y_outer_unstable = ax.transAxes.inverted().transform(ax.transData.transform(self.outer['rfp_u'][:2]))

        ax.text(x_inner_stable, y_inner_stable, 'inner stable', ha='right')
        ax.text(x_inner_unstable, y_inner_unstable, 'inner unstable', ha='left')
        ax.text(x_outer_stable, y_outer_stable, 'outer stable', ha='left')
        ax.text(x_outer_unstable, y_outer_unstable, 'outer unstable', ha='right')

        return fig, ax

    def error_linear_regime(self, epsilon, rfp, eigenvector, direction=1):
        """Metric to evaluate if the point rfp + epsilon * eigenvector is in the linear regime of the fixed point."""
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[:, 1] - rz_path[:, 0]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

        return np.abs(1 - np.dot(eps_dir_norm, eigenvector))

    ### Computation of the manifolds

    def start_config(self, epsilon, rfp, eigenvalue, eigenvector, neps=10, direction=1):
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
            float: norm of the difference between the computed eigenvector and the given one
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
        RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

        return RZs

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

    def compute(self, epsilon: float = None, **kwargs):
        """Computation of the manifold. If no fixed point number is given, the tangle is computed otherwise
        selects the fixed point associated to fp_num and compute for the directions.

        Args:
            epsilon (float, optional): epsilon for the starting configuration. If not given, it is computed for both the stable and unstable directions.
            fp_num (int, optional): fixed point number. If given, the computation is perform for manifold from this fixed point only.

        Keyword Args:
            eps_s (float): epsilon in the stable direction
            eps_u (float): epsilon in the unstable direction
            eps_guess_s (float): guess for epsilon in the stable direction (if eps_s is not given)
            eps_guess_u (float): guess for epsilon in the unstable direction (if eps_u is not given)
            nintersect (int): number of intersections
            directions (str): directions to compute the manifold
            neps (int): number of points in the starting configuration
        """
        options = {
            "eps_guess_s": 1e-3,
            "eps_guess_u": 1e-3,
            "nintersect": 10,
            "neps": 2,
            "directions": "both",
            "eps_s": None,
            "eps_u": None,
        }
        options.update({key: value for key, value in kwargs.items() if key in options})

        # Setup the fixedpoints/eigenvectors to use
        if options['directions'] == "both":
            directions = [self.inner, self.outer]
        elif options['directions'] == "inner":
            directions = [self.inner]
        elif options['directions'] == "outer":
            directions = [self.outer]
        else:
            raise ValueError("Invalid directions")
        logger.info(f"Computing manifold for directions [inner/outer/both]: {options['directions']}")

        # Computation
        for dik in directions:
            rfp_s, rfp_u = dik['rfp_s'], dik['rfp_u']
            vector_s, vector_u = dik['vector_s'], dik['vector_u']
            lambda_s, lambda_u = dik['lambda_s'], dik['lambda_u']

            # Setup the epsilon (eps_s, eps_u) of stable and unstable directions
            if epsilon is not None:
                if options["eps_s"] is not None:
                    logger.warning("Both eps_s and epsilon are given, ignoring the eps_s.")
                if options["eps_u"] is not None:
                    logger.warning("Both eps_u and epsilon are given, ignoring the eps_u.")
                options["eps_s"] = epsilon
                options["eps_u"] = epsilon
            if options["eps_s"] is None:
                options["eps_s"] = self.find_epsilon(
                    rfp_s, vector_s, options["eps_guess_s"], -1
                )
            if options["eps_u"] is None:
                options["eps_u"] = self.find_epsilon(
                    rfp_u, vector_u, options["eps_guess_u"]
                )

            # Compute the unstable starting configuration and the manifold
            RZs = self.start_config(
                options["eps_u"], rfp_u, lambda_u, vector_u, options["neps"]
            )
            logger.info("Computing unstable manifold...")
            dik["lfs"]["unstable"] = self.integrate(RZs, nintersect=options["nintersect"])

            # Compute the stable starting configuration and the manifold
            RZs = self.start_config(
                options["eps_s"], rfp_s, lambda_s, vector_s, options["neps"], -1
            )
            logger.info("Computing stable manifold...")
            dik["lfs"]["stable"] = self.integrate(
                RZs, nintersect=options["nintersect"], direction=-1
            )


    def plot(self, directions="isiuosou", color=None, end=None, **kwargs):
        default = {
            "markersize": 2,
            "fmt": "-o",
            "colors": ["green", "red", "green", "red"],
        }
        default.update({key: value for key, value in kwargs.items() if key in default})
        plotkwargs = {key: value for key, value in kwargs.items() if key not in default}

        fig, ax, kwargs = create_canvas(**kwargs)

        dirdict = {
            "is": self.inner["lfs"]["stable"],
            "iu": self.inner["lfs"]["unstable"],
            "os": self.outer["lfs"]["stable"],
            "ou": self.outer["lfs"]["unstable"]
        }

        for i, dir in enumerate(["is", "iu", "os", "ou"]):
            if dir in directions:
                out = dirdict[dir]
                if out is None:
                    logger.warning(f"Manifold {dir} not computed.")
                else:
                    # plotting each starting point trajectory as order
                    # for yr, yz in zip(out[::2], out[1::2]):
                    #     # ax.scatter(yr, yz, alpha=1, s=5)
                    #     ax.scatter(yr, yz, color=color[i], alpha=1, s=5)
                    if color is None:
                        tmpcolor = default["colors"][i]
                    else:
                        tmpcolor = color
                    if end is not None:
                        if end > out.shape[1]:
                            raise ValueError("End index out of bounds")
                        out = out[:, :end]

                    out = out.T.flatten()
                    ax.plot(
                        out[::2],
                        out[1::2],
                        default["fmt"],
                        label=f"{dir} - manifold",
                        color=tmpcolor,
                        markersize=default["markersize"],
                        **plotkwargs,
                    )
                    #     # ax.scatter(yr, yz, alpha=1, s=5)
                    #     ax.scatter(yr, yz, color=color[i], alpha=1, s=5)

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

    def turnstile_area(self, cyl_flag, n_joining = 100):
        """Compute the turnstile area by integrating the vector potential along the trajectory of the homo/hetero-clinics points.
        """

        # Function for forward/backward integration for each clinic point
        def integrate_direction(rz, n, rfp, direction):
            history = []
            intA = []
            n_tmp = 0

            while n_tmp < n:
                rz_end, intA_tmp = self.integrate_single(
                    rz, 1, direction=direction, ret_jacobian=False, integrate_A=True
                )

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
                mid_gamma = np.vstack((mid_gamma[:,0], self._params['zeta']*np.ones(mid_gamma.shape[0]), mid_gamma[:,1])).T
                
                if cyl_flag:
                    mid_A = np.array([self._problem.A(r)[0::2] for r in mid_gamma])
                else:
                    mid_A = np.empty((mid_gamma.shape[0], 2))
                    for k, r in enumerate(mid_gamma):
                        xyz = np.array([
                            r[0] * np.cos(r[1]),
                            r[0] * np.sin(r[1]),
                            r[2]
                        ])
                        invJacobian = self._problem._inv_Jacobian(r[0], r[1], r[2])
                        mid_A[k] = np.matmul(invJacobian, np.array([self._problem.A(xyz)]).T).T[0][::2]

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