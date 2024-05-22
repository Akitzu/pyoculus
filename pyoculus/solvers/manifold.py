from ..problems.bfield_problem import BfieldProblem
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from scipy.optimize import root, minimize

# from functools import total_ordering
import matplotlib.pyplot as plt
import numpy as np

import structlog

log = structlog.get_logger()


class Manifold(BaseSolver):
    def __init__(
        self,
        fixedpoint,
        bfield,
        params=dict(),
        integrator=None,
        integrator_params=dict(),
    ):
        """! Set up the manifold solver
        @param fixedpoint, computed fixed point
        @param bfield, instance of the BfieldProblem class

        """
        # Check that the fixed point is a correct FixedPoint instance
        if not isinstance(fixedpoint, FixedPoint):
            raise AssertionError("Fixed point must be an instance of FixedPoint class")
        if not fixedpoint.successful:
            raise AssertionError(
                "Need a successful fixed point to compute the manifold"
            )

        self.fixedpoint = fixedpoint
        self.fixedpoint.compute_all_jacobians()

        # Initialize the manifolds for later computation
        self.unstable = {"+": None, "-": None}
        self.stable = {"+": None, "-": None}

        # Check that the bfield is a correct BfieldProblem instance
        if not isinstance(bfield, BfieldProblem):
            raise AssertionError("Bfield must be an instance of BfieldProblem class")

        # Integrator and BaseSolver initialization
        integrator_params["ode"] = bfield.f_RZ
        default_params = {"zeta": 0}
        default_params.update(params)

        super().__init__(
            problem=bfield,
            params=default_params,
            integrator=integrator,
            integrator_params=integrator_params,
        )

    @staticmethod
    def eig(jacobian):
        """Compute the eigenvalues and eigenvectors of the jacobian and returns them in the order : stable, unstable."""
        eigRes = np.linalg.eig(jacobian)
        eigenvalues = eigRes[0]

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

    def choose(self, fp_num_1, fp_num_2, directions=["u+", "s+"], sngs=[1, 1]):
        """Choose the two fixed points and their stable or unstable directions."""

        rfp_1 = np.array([self.fixedpoint.x[fp_num_1], self.fixedpoint.z[fp_num_1]])
        p1_lambda_s, p1_vector_s, p1_lambda_u, p1_vector_u = self.eig(
            self.fixedpoint.all_jacobians[fp_num_1]
        )

        rfp_2 = np.array([self.fixedpoint.x[fp_num_2], self.fixedpoint.z[fp_num_2]])
        p2_lambda_s, p2_vector_s, p2_lambda_u, p2_vector_u = self.eig(
            self.fixedpoint.all_jacobians[fp_num_2]
        )

        # Initialize the choice
        self.directions = "".join(directions)
        if "u" in directions[0]:
            self.rfp_u, self.lambda_u, self.vector_u = (
                rfp_1,
                p1_lambda_u,
                sngs[0] * p1_vector_u,
            )
            self.rfp_s, self.lambda_s, self.vector_s = (
                rfp_2,
                p2_lambda_s,
                sngs[1] * p2_vector_s,
            )
        else:
            self.rfp_s, self.lambda_s, self.vector_s = (
                rfp_1,
                p1_lambda_u,
                sngs[0] * p1_vector_u,
            )
            self.rfp_u, self.lambda_u, self.vector_u = (
                rfp_2,
                p2_lambda_s,
                sngs[1] * p2_vector_s,
            )

        self.clinics = []

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
        """Compute a starting configuration for the manifold drawing. It takes a point in the linear regime
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

    def find_epsilon(self, rfp, eigenvector, eps_guess=1e-6, direction=1):
        """Find the epsilon that lies in the linear regime."""
        find_eps = lambda x: self.error_linear_regime(
            x, rfp, eigenvector, direction=direction
        )
        minobj = minimize(find_eps, eps_guess, bounds=[(0, 1)], tol=1e-12)

        if not minobj.success:
            log.warning(
                "Search for minimum of the linear error failed, using the guess for epsilon."
            )
            return eps_guess
        else:
            esp_root = minobj.x[0]
            log.info(
                f"Search for minimum of the linear error succeeded, epsilon = {esp_root:.5e}"
            )
            return esp_root

    def compute(self, epsilon: float = None, fp_num: int = None, **kwargs):
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
            "eps_guess_s": 2e-6,
            "eps_guess_u": 2e-6,
            "nintersect": 10,
            "neps": 2,
            "directions": "u+u-s+s-",
            "eps_s": None,
            "eps_u": None,
        }
        options.update({key: value for key, value in kwargs.items() if key in options})

        # Setup the fixedpoints/eigenvectors to use
        if fp_num is not None:
            if fp_num not in range(self.fixedpoint.x.shape[0] - 1):
                raise ValueError("Invalid fixed point number")
            lambda_s, vector_s, lambda_u, vector_u = self.eig(
                self.fixedpoint.all_jacobians[fp_num]
            )
            rfp_u = np.array([self.fixedpoint.x[fp_num], self.fixedpoint.z[fp_num]])
            rfp_s = np.array([self.fixedpoint.x[fp_num], self.fixedpoint.z[fp_num]])
        else:
            options["directions"] = self.directions
            lambda_s, vector_s = self.lambda_s, self.vector_s
            lambda_u, vector_u = self.lambda_u, self.vector_u
            rfp_s, rfp_u = self.rfp_s, self.rfp_u

        # Setup the epsilon (eps_s, eps_u) of stable and unstable directions
        if epsilon is not None:
            if options["eps_s"] is not None:
                log.warning("Both eps_s and epsilon are given, ignoring the eps_s.")
            if options["eps_u"] is not None:
                log.warning("Both eps_u and epsilon are given, ignoring the eps_u.")
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
        if "u+" in options["directions"]:
            log.info("Computing unstable manifold with postive epsilon...")
            self.unstable["+"] = self.integrate(RZs, nintersect=options["nintersect"])

        if "u-" in options["directions"]:
            RZs = 2 * rfp_u - RZs
            log.info("Computing unstable manifold with negative epsilon...")
            self.unstable["-"] = self.integrate(RZs, nintersect=options["nintersect"])

        # Compute the stable starting configuration and the manifold
        RZs = self.start_config(
            options["eps_s"], rfp_s, lambda_s, vector_s, options["neps"], -1
        )
        if "s+" in options["directions"]:
            log.info("Computing stable manifold with positive epsilon...")
            self.stable["+"] = self.integrate(
                RZs, nintersect=options["nintersect"], direction=-1
            )

        if "s-" in options["directions"]:
            log.info("Computing stable manifold with negative epsilon...")
            RZs = 2 * rfp_s - RZs
            self.stable["-"] = self.integrate(
                RZs, nintersect=options["nintersect"], direction=-1
            )

    def plot(self, ax=None, directions="u+u-s+s-", color=None, end=None, **kwargs):
        default = {
            "markersize": 2,
            "fmt": "-o",
            "colors": ["red", "blue", "green", "purple"],
        }
        default.update({key: value for key, value in kwargs.items() if key in default})

        plotkwargs = {key: value for key, value in kwargs.items() if key not in default}

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        dirdict = {
            "u+": self.unstable["+"],
            "u-": self.unstable["-"],
            "s+": self.stable["+"],
            "s-": self.stable["-"],
        }
        for i, dir in enumerate(["u+", "u-", "s+", "s-"]):
            if dir in directions:
                out = dirdict[dir]
                if out is None:
                    log.warning(f"Manifold {dir} not computed.")
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
        fund = self.fundamental_segment[1]
        norm = np.linalg.norm(clinic - self.rfp_u)

        r_ev = clinic
        max_iterations = 20  # Set a maximum number of iterations
        for _ in range(max_iterations):
            if fund[0] <= norm < fund[1]:
                return norm
            log.debug(f"norm = {norm}")
            r_ev = self.integrate_single(r_ev, 1, -1, ret_jacobian=False)
            norm = np.linalg.norm(r_ev - self.rfp_u)

        raise ValueError(
            "Failed to find a solution within the maximum number of iterations"
        )

    def find_bounds(self, guess_eps_s, guess_eps_u):
        """Find the bounds on the unstable and stable manifolds"""
        r_s = self.rfp_s + guess_eps_s * self.vector_s
        r_u = self.rfp_u + guess_eps_u * self.vector_u
        r_s_evolved = self.integrate_single(r_s, 1, -1, ret_jacobian=False)
        r_u_evolved = self.integrate_single(r_u, 1, 1, ret_jacobian=False)

        upperbound_s = np.linalg.norm(r_s_evolved - self.rfp_s)
        upperbound_u = np.linalg.norm(r_u_evolved - self.rfp_u)

        bounds = ((guess_eps_s, upperbound_s), (guess_eps_u, upperbound_u))
        return bounds

    def find_N(self, guess_eps_s=1e-3, guess_eps_u=1e-3):
        """Finding the number of times the map needs to be applied for the stable and unstable points to cross."""

        r_s = self.rfp_s + guess_eps_s * self.vector_s
        r_u = self.rfp_u + guess_eps_u * self.vector_u

        first_dir = r_u - r_s
        last_norm = np.linalg.norm(first_dir)

        n_s, n_u = 0, 0
        success, stable_evol = False, True
        while not success:
            if stable_evol:
                r_s = self.integrate_single(r_s, 1, -1, ret_jacobian=False)
                n_s += 1
            else:
                r_u = self.integrate_single(r_u, 1, 1, ret_jacobian=False)
                n_u += 1
            stable_evol = not stable_evol

            norm = np.linalg.norm(r_u - r_s)
            # log.debug(f"{np.dot(first_dir, r_u - r_s)} / {last_norm} / {norm}")
            if np.sign(np.dot(first_dir, r_u - r_s)) < 0:  # and last_norm < norm:
                success = True
            last_norm = norm

        if not success:
            raise ValueError("Could not find N")
        else:
            return n_s, n_u

    def find_homoclinic(self, guess_eps_s=None, guess_eps_u=None, **kwargs):
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
            eps_s_lb = self.find_epsilon(self.rfp_s, self.vector_s, direction=-1)
        elif (
            self.error_linear_regime(
                guess_eps_s, self.rfp_s, self.vector_s, direction=-1
            )
            > ERR
        ):
            raise ValueError("Guess for stable epsilon is not in the linear regime.")
        else:
            eps_s_lb = guess_eps_s

        if guess_eps_u is None:
            eps_u_lb = self.find_epsilon(self.rfp_u, self.vector_u)
        elif self.error_linear_regime(guess_eps_u, self.rfp_u, self.vector_u) > ERR:
            raise ValueError("Guess for unstable epsilon is not in the linear regime.")
        else:
            eps_u_lb = guess_eps_u

        # Find the bounds of the search domain : lower bound epsilons are map to upper bound epsilons after one iteration
        if defaults["bounds"] is None:
            defaults["bounds"] = self.find_bounds(eps_s_lb, eps_u_lb)
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
        log.debug(f"Guess - {guess_eps_s}, {guess_eps_u}")
        log.debug(f"Bounds - {defaults['bounds']}")
        log.debug(f"n_s, n_u - {n_s}, {n_u}")

        self.history = []

        # Residual function for the root finding
        def evolution(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = self.rfp_s + eps_s * self.vector_s
            r_u = self.rfp_u + eps_u * self.vector_u

            try:
                if defaults['root']['jac']:
                    r_s_evolved, jac_s = self.integrate_single(r_s, n_s, -1)
                else:    
                    r_s_evolved = self.integrate_single(r_s, n_s, -1, ret_jacobian=False)
            except Exception as e:
                log.error(f"Error in stable manifold integration : {e}")
                # breakpoint()

            try:
                if defaults['root']['jac']:
                    r_u_evolved, jac_u = self.integrate_single(r_u, n_u, 1)
                else:    
                    r_u_evolved = self.integrate_single(r_u, n_u, 1, ret_jacobian=False)
            except Exception as e:
                log.error(f"Error in unstable manifold integration : {e}")
                # breakpoint()

            if defaults['root']['jac']:
                return (
                    r_s_evolved,
                    r_u_evolved,
                    r_s_evolved - r_u_evolved,
                    np.array([jac_s @ self.vector_s, -jac_u @ self.vector_u]),
                )
            else:
                return (
                    r_s_evolved,
                    r_u_evolved,
                    r_s_evolved - r_u_evolved
                )

        def residual(logeps, n_s, n_u):
            eps_s, eps_u = np.exp(logeps)
            log.debug(f'Inside : {eps_s, eps_u}')
            # if not defaults['bounds'][0][0] <= eps_s <= defaults['bounds'][0][1] or not defaults['bounds'][1][0] <= eps_u <= defaults['bounds'][1][1]:
            #     dist_s = min(abs(eps_s - defaults['bounds'][0][0]), abs(eps_s - defaults['bounds'][0][1]))
            #     dist_u = min(abs(eps_u - defaults['bounds'][1][0]), abs(eps_u - defaults['bounds'][1][1]))
            #     coef = 1+10**(np.log(dist_s+dist_u)/np.log(defaults['bounds'][0][1]+defaults['bounds'][1][1]))
            #     # ret = coef*evolution([min(max(eps_s, defaults['bounds'][0][0]), defaults['bounds'][0][1]), min(max(eps_u, defaults['bounds'][1][0]), defaults['bounds'][1][1])], n_s, n_u)[2]
            #     ret = (np.array([dist_s, dist_u])/defaults['bounds'][0][0])**2
            #     log.debug(f"Outside : {eps_s, eps_u} - {ret[:3]}")
            #     return ret
            # else:
            ret = evolution([eps_s, eps_u], n_s, n_u)
            self.history.append(ret)
            
            if defaults['root']['jac']:
                ret[3][:, 0] *= eps_s
                ret[3][:, 1] *= eps_u
                log.debug(f"Returns - {ret[:3]}")
            else:
                log.debug(f"Returns - {ret}")
                
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

        log.info(f"Root finding status : {r.message}")
        log.debug(f"Root finding object : {r}")

        if not r.success:
            raise ValueError("Homoclinic search not successful.")

        eps_s, eps_u = np.exp(r.x)
        # if self.error_linear_regime(eps_s, self.rfp_s, self.vector_s, direction=-1) > 1e-4:
        #     raise ValueError("Homoclinic point stable epsilon does not lie linear regime.")
        # if self.error_linear_regime(eps_u, self.rfp_u, self.vector_u) > 1e-4:
        #     raise ValueError("Homoclinic point unstable epsilon does not lie linear regime.")

        log.info(
            f"Eps_s : {eps_s:.3e}, Eps_u : {eps_u:.3e} gives a difference in endpoint [R,Z] : {r.fun}"
        )

        # Recording the homo/hetero-clinic point
        r_s_ev, r_u_ev = evolution([eps_s, eps_u], n_s, n_u)[:2]
        if not self.clinics:
            self.fundamental_segment = self.find_bounds(eps_s, eps_u)
            order = eps_u
            self.find_clinic_configuration = {"n_s": n_s, "n_u": n_u}
        else:
            order = self._order(self.rfp_u + eps_u * self.vector_u)

        if not np.any([np.isclose(other[0], order) for other in self.clinics]):
            self.clinics.append((order, eps_s, eps_u, r_s_ev, r_u_ev))
        else:
            log.warning("Clinic already recorded, skipping...")

        return eps_s, eps_u

    def find_clinics(self, indices = None, n_points = 1, m = 0, **kwargs):
        if len(self.clinics) == 0:
            self.find_homoclinic(**kwargs)
            
        bounds_0 = self.fundamental_segment

        for key in ['n_s', 'n_u']:
            if key in kwargs:
                kwargs.pop(key)

        if indices is None:
            indices = range(1, n_points)

        for i in indices:
            bounds_i = np.array(bounds_0)
            bounds_i[0][0] = self.clinics[-1][1]
            bounds_i[1][1] = self.clinics[-1][2]
            bounds_i = (tuple(bounds_i[0]), tuple(bounds_i[1]))

            guess_i = [
                bounds_0[0][1] * np.power(self.lambda_s, i / n_points),
                bounds_0[1][0] * np.power(self.lambda_u, i / n_points),
            ]
            log.info(f"Initial guess: {guess_i}")

            n_s = self.find_clinic_configuration["n_s"] + m
            n_u = self.find_clinic_configuration["n_u"] - m - 1
            self.find_homoclinic(
                *guess_i, bounds=bounds_i, n_s=n_s, n_u=n_u, **kwargs
            )
        
        if len(self.clinics) != n_points:
            log.warning("Number of clinic points is not the expected one.")

        self.order()

    # def clinic_bijection(self, guess_eps_s, guess_eps_u, **kwargs):
    #     defaults = {"tol": 1e-10, "n_s": None, "n_u": None}
    #     defaults.update(
    #         {key: value for key, value in kwargs.items() if key in defaults}
    #     )

    #     # Find the first homo/hetero-clinic point
    #     eps_s_1, eps_u_1 = self.find_homoclinic(guess_eps_s, guess_eps_u, **defaults)

    #     bounds_1 = self.find_bounds(eps_s_1, eps_u_1)

    #     guess_2 = [eps_s_1*np.power(self.lambda_s, 1/2), eps_u_1*np.power(self.lambda_u, 1/2)]
    #     eps_s_2, eps_u_2 = self.find_homoclinic(guess_2[0], guess_2[1], **defaults)

    #     # considering the >_u ordering of the homoclinic points

    ### Calculating Turnstile Area

    def order(self):
        """Order the homo/hetero-clinic points with the induced linear ordering of the unstable manifold >_u."""
        self.clinics = [
            self.clinics[i] for i in np.argsort([x[0] for x in self.clinics])
        ]

    def resonance_area(self, n_f_max=40, n_b_max=40, n_transit=3):
        """Compute the turnstile area by integrating the vector potential along the trajectory of the homo/hetero-clinics points.

        Args:
            n_f_max (int): Maximum number of forward iteration.
            n_b_max (int): Maximum number of backward iteration.
            n_transit (int): Number of iteration consider as transition before being close to the fixedpoints.
        
        Returns:
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

                if n_tmp > n_transit and np.linalg.norm(
                    rz_end - rfp
                ) > np.linalg.norm(rz - rfp):
                    if direction == -1:
                        direction_str = "Backward"
                    else:
                        direction_str = "Forward"
                    log.info(f"{direction_str} integration goes beyond stable saddle point.")
                    log.debug(
                        f"rfp: {rfp}, rz_end: {rz_end}, rz: {rz}"
                    )
                    break

                rz = rz_end
                history.append(rz)
                intA.append(intA_tmp)
                n_tmp += 1
            
            return history, intA

        # Potential integration
        potential_integrations = []
        history = []
        for clinic in self.clinics:
            # Forward integration
            rz_forward = clinic[-2]
            history_forward, intA_forward = integrate_direction(rz_forward, n_f_max, self.rfp_s, 1)

            # Backward integration
            # taking the point found from unstable manifold to go back to the fixedpoint
            rz_backward = clinic[-1]
            history_backward, intA_backward = integrate_direction(rz_backward, n_b_max, self.rfp_u, -1)

            log.info(
                f"Potential integration completed for homo/hetero-clinic point of order : {clinic[0]:.3e}"
            )

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

            # Area calculation
            intm = intA_m[0][:n_fwd].sum() - intA_m[1][:n_bwd].sum()
            inth = intA_h[0][:n_fwd].sum() - intA_h[1][:n_bwd].sum()
            areas[i] = intm - inth

            # Error by difference
            errf = np.abs(intA_m[0][-1] - intA_h[0][-1])
            errb = np.abs(intA_m[1][-1] - intA_h[1][-1])
            err_by_diff[i] = errf + errb

            # Error by estimation
            for j, n in enumerate([n_fwd, n_bwd]):
                r1 = history[i][j][n-1]
                r2 = history[(i+1)%len(potential_integrations)][j][n-1]
                rmid = (r2 + r1)/2
                Amid = self._problem.A([rmid[0], self._params['zeta'], rmid[1]])[0::2]
                err_by_estim[i] += np.abs(np.dot(Amid, r2 - r1))

        self.areas = np.vstack((areas, err_by_diff, err_by_estim)).T
        
        return areas, potential_integrations, history

    ### Integration methods

    def integrate(self, RZstart, nintersect, direction=1):
        RZstart = np.atleast_2d(RZstart)
        rz_path = np.zeros((2 * RZstart.shape[0], nintersect + 1))
        rz_path[:, 0] = RZstart.flatten()

        t0 = self._params["zeta"]
        dt = self.fixedpoint.qq * direction * 2 * np.pi / self._problem.Nfp

        for i, rz in enumerate(RZstart):
            t = t0
            ic = rz

            for j in range(nintersect):
                try:
                    self._integrator.set_initial_value(t, ic)
                except:
                    log.error(f"Integration of point {ic} failed.")
                    break
                output = self._integrator.integrate(t + dt)

                t = t + dt
                ic = output

                rz_path[2 * i, j + 1] = output[0]
                rz_path[2 * i + 1, j + 1] = output[1]

        return rz_path

    def integrate_single(
        self,
        RZstart,
        nintersect,
        direction=1,
        ret_jacobian=True,
        integrate_A=False,
        dt=None,
    ):
        r, z = RZstart
        t0 = self._params["zeta"]
        if dt is None:
            dt = self.fixedpoint.qq * direction * 2 * np.pi / self._problem.Nfp

        t = t0
        if ret_jacobian:
            ic = np.array([r, z, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self._integrator.change_rhs(self._problem.f_RZ_tangent)
        elif integrate_A:
            ic = np.array([r, z, 0.0], dtype=np.float64)
            self._integrator.change_rhs(self._problem.f_RZ_A)
        else:
            ic = np.array([r, z], dtype=np.float64)

        self._integrator.set_initial_value(t, ic)

        try:
            for _ in range(nintersect):
                output = self._integrator.integrate(t + dt)
                t = t + dt
        except Exception as e:
            self._integrator.change_rhs(self._problem.f_RZ)
            raise e

        if ret_jacobian:
            self._integrator.change_rhs(self._problem.f_RZ)
            jacobian = output[2:].reshape((2, 2)).T
            return output[:2], jacobian
        elif integrate_A:
            self._integrator.change_rhs(self._problem.f_RZ)
            return output[:2], output[2]
        else:
            return output[:2]
