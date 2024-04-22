from ..problems.bfield_problem import BfieldProblem
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from scipy.optimize import newton_krylov, fsolve
import matplotlib.pyplot as plt
import numpy as np

class Manifold(BaseSolver):
    def __init__(self, fixedpoint, bfield, params=dict(), integrator=None, integrator_params=dict()):
        
        # Check that the fixed point is a correct FixedPoint instance
        if not isinstance(fixedpoint, FixedPoint):
            raise AssertionError("Fixed point must be an instance of FixedPoint class")
        if not fixedpoint.successful:
            raise AssertionError("Need a successful fixed point to compute the manifold")
        
        self.fixedpoint = fixedpoint
        self.fixedpoint.compute_all_jacobians()
        
        # Initialize the manifolds for later computation
        self.unstable = {'+': None, '-': None}
        self.stable = {'+': None, '-': None}

        # Check that the bfield is a correct BfieldProblem instance
        if not isinstance(bfield, BfieldProblem):
            raise AssertionError("Bfield must be an instance of BfieldProblem class")
        
        self.bfield = bfield

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

        return eigenvalues[s_index], eigenvectors[s_index], \
               eigenvalues[u_index], eigenvectors[u_index] 

    def choose(self):
        """Choose the fixed points and their stable or unstable directions."""

        # For now just take twice the same fp
        self.rfp_s = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]])
        self.rfp_u = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]])
        self.lambda_s, self.vector_s, self.lambda_u, self.vector_u = self.eig(self.fixedpoint.all_jacobians[0])
        self.directions = "u+s+"

    def find_N(self, guess_eps_s = 1e-3, guess_eps_u = 1e-3):
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
            #sprint(f"{np.dot(first_dir, r_u - r_s)} / {last_norm} / {norm}")
            if np.sign(np.dot(first_dir, r_u - r_s)) < 0 and last_norm < norm:
                success = True
            last_norm = norm

        if not success:
            raise ValueError("Could not find N")    
        else:
            return n_s, n_u

    def find_homoclinic(self, guess_eps_s = 1e-3, guess_eps_u = 1e-3, **kwargs):
        defaults = {"maxiter": 100, "n_s": None, "n_u": None}
        defaults.update({key: value for key, value in kwargs.items() if key in defaults})

        if defaults['n_s'] is None or defaults['n_u'] is None:
            n_s, n_u = self.find_N(guess_eps_s, guess_eps_u)
            print(f"Found [n_s, n_u] : [{n_s}, {n_u}]")
        else:
            n_s, n_u = defaults['n_s'], defaults['n_u']

        def evolution(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = self.rfp_s + eps_s * self.vector_s
            r_u = self.rfp_u + eps_u * self.vector_u

            r_s_evolved, jac_s = self.integrate_single(r_s, n_s, -1)
            r_u_evolved, jac_u = self.integrate_single(r_u, n_u, 1)

            return r_s_evolved - r_u_evolved, np.array([jac_s @ self.vector_s, -jac_u @ self.vector_u])
        
        def residual(eps, n_s, n_u):
            return evolution(eps, n_s, n_u)[0]

        def jacobian(eps, n_s, n_u):
            return evolution(eps, n_s, n_u)[1]
        
        eps_s_1, eps_u_1 = fsolve(residual, [guess_eps_s, guess_eps_u], args=(n_s, n_u), fprime=jacobian)
        print(f"Eps 1: {eps_s_1}, {eps_u_1}")
        if eps_s_1 < 0 or eps_u_1 < 0:
            raise ValueError("Homoclinic point epsilon cannot be negative.")
        # if self.error_linear_regime(self.rfp_s, self.lambda_s, self.vector_s) > 1e-4 or self.error_linear_regime(self.rfp_u, self.lambda_u, self.vector_u) > 1e-4:
        #     raise ValueError("Homoclinic point epsilon was be found in linear regime.")    
        
        return eps_s_1, eps_u_1

    def compute(self, epsilon = None, fp_num = None, **kwargs):
        """Compute the manifolds of the fixed point. If no fixed point number is given, the two manifold coicinding with the tangle plot is computed otherwise 
        just select the fixed point and compute the manifold in directions."""
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
        
        if fp_num is not None:
            if fp_num not in range(self.fixedpoint.qq):
                raise ValueError("Invalid fixed point number")
            lambda_s, vector_s, lambda_u, vector_u = self.eig(self.fixedpoint.all_jacobians[fp_num])
            rfp_u = np.array([self.fixedpoint.x[fp_num], self.fixedpoint.z[fp_num]])
            rfp_s = np.array([self.fixedpoint.x[fp_num], self.fixedpoint.z[fp_num]])
        else:
            options['directions'] = self.directions
            lambda_s, vector_s = self.lambda_s, self.vector_s
            lambda_u, vector_u = self.lambda_u, self.vector_u
            rfp_s, rfp_u = self.rfp_s, self.rfp_u

        if epsilon is not None:
            if options['eps_s'] is not None:
                print("Warning: both eps_s and epsilon are given, ignoring the eps_s.")
            if options['eps_u'] is not None:
                print("Warning: both eps_u and epsilon are given, ignoring the eps_u.")
            options['eps_s'] = epsilon
            options['eps_u'] = epsilon
        if options['eps_s'] is None:
            options['eps_s'] = self.find_epsilon(options['eps_guess_s'], rfp_s, vector_s, -1)
        if options['eps_u'] is None:
            options['eps_u'] = self.find_epsilon(options['eps_guess_u'], rfp_u, vector_u)

        # Compute the unstable starting configuration and the manifold 
        RZs = self.start_config(options['eps_u'], rfp_u, vector_u, lambda_u, options['neps'])[0]
        if 'u+' in options['directions']:
            print("Computing unstable manifold with postive epsilon...")
            self.unstable['+'] = self.integrate(RZs, nintersect=options['nintersect'])

        if 'u-' in options['directions']:
            RZs  = 2*rfp_u - RZs
            print("Computing unstable manifold with negative epsilon...")
            self.unstable['-'] = self.integrate(RZs, nintersect=options['nintersect'])
        
        # Compute the stable starting configuration and the manifold 
        RZs = self.start_config(options['eps_s'], rfp_s, vector_s, lambda_s, options['neps'], -1)[0]
        if 's+' in options['directions']:
            print("Computing stable manifold with positive epsilon...")
            self.stable['+'] = self.integrate(RZs, nintersect=options['nintersect'], direction=-1)
        
        if 's-' in options['directions']:
            print("Computing stable manifold with negative epsilon...")
            RZs = 2*rfp_s - RZs
            self.stable['-'] = self.integrate(RZs, nintersect=options['nintersect'], direction=-1)


    def start_config(self, epsilon, rfp, eigenvector, eigenvalue = None, neps=10, direction=1, intervtype = "logspace"):
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)
        
        eps_dir = rz_path[:,1]-rz_path[:,0]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir
        
        if intervtype == "logspace":
            eps = np.logspace(np.log(epsilon), np.log(epsilon+norm_eps_dir), neps, base = np.exp(1))
            Rs = rfp[0] + eps * eps_dir_norm[0]
            Zs = rfp[1] + eps * eps_dir_norm[1]
        elif intervtype == "linspace":
            Rs = np.linspace(rz_path[0,0], rz_path[0,1], neps)
            Zs = np.linspace(rz_path[1,0], rz_path[1,1], neps)
        else:
            raise ValueError("Invalid interval type")
        
        RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])
        return RZs, np.linalg.norm(eps_dir_norm - eigenvector)
    

    def find_epsilon(self, eps_guess, rfp, eigenvector, direction = 1, iter = 4):
        find_eps = lambda x: self.start_config(x, rfp, eigenvector, neps = 1, direction = direction)[1]

        # eps_root = fsolve(find_eps, eps_guess, xtol=options['xtol'])
        try:
            eps_root = newton_krylov(find_eps, eps_guess, inner_maxiter=iter)[0]
            print(f"Newton-Krylov succeeded, epsilon = {eps_root}")
        except:
            print("Newton-Krylov failed, using the guess for epsilon.")
            return eps_guess

        return eps_root

    def integrate(self, RZstart, nintersect, direction = 1):
        RZstart = np.atleast_2d(RZstart)
        rz_path = np.zeros((2*RZstart.shape[0], nintersect+1))
        rz_path[:,0] = RZstart.flatten()
        
        t0 = self._params["zeta"]
        dt = direction*2*np.pi/self.bfield.Nfp

        for i, rz in enumerate(RZstart):
            t = t0
            ic = rz

            for j in range(nintersect):
                try:
                    self._integrator.set_initial_value(t, ic)
                except:
                    print(f"error for integration of point {ic}")
                    break
                output = self._integrator.integrate(t + dt)

                t = t + dt
                ic = output

                rz_path[2*i, j+1] = output[0]
                rz_path[2*i+1, j+1] = output[1]

        return rz_path
    
    def integrate_single(self, RZstart, nintersect, direction = 1, ret_jacobian = True):
        r, z = RZstart
        t0 = self._params["zeta"]
        dt = direction*2*np.pi/self.bfield.Nfp

        t = t0
        if ret_jacobian:
            ic = np.array([r, z, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self._integrator.change_rhs(self.bfield.f_RZ_tangent)
        else:
            ic = np.array([r, z], dtype=np.float64)

        self._integrator.set_initial_value(t, ic)
        for _ in range(nintersect):
            output = self._integrator.integrate(t + dt)
            t = t + dt

        if ret_jacobian:    
            self._integrator.change_rhs(self.bfield.f_RZ)
            jacobian = output[2:].reshape((2, 2)).T
            return output[:2], jacobian
        else:
            return output[:2]


    def plot(self, ax = None, directions = "u+u-s+s-", color = None, end = None, **kwargs):
        default = {"markersize": 2, "fmt": "-o", "colors": ['red', 'blue', 'green', 'purple']}
        default.update({key: value for key, value in kwargs.items() if key in default})

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        dirdict = {'u+': self.unstable['+'], 'u-': self.unstable['-'], 's+': self.stable['+'], 's-': self.stable['-']}
        for i, dir in enumerate(["u+", "u-", "s+", "s-"]):
            if dir in directions:
                out = dirdict[dir]
                if out is None:
                    print(f"Manifold {dir} not computed.")
                else:
                    # plotting each starting point trajectory as order 
                    # for yr, yz in zip(out[::2], out[1::2]):
                    #     # ax.scatter(yr, yz, alpha=1, s=5)
                    #     ax.scatter(yr, yz, color=color[i], alpha=1, s=5)
                    if color is None:
                        tmpcolor = default["colors"][i]
                    if end is not None:
                        if end > out.shape[1]:
                            raise ValueError("End index out of bounds")
                        out = out[:, :end]

                    out = out.T.flatten()
                    ax.plot(out[::2], out[1::2], default['fmt'], label=f'{dir} - manifold', color=tmpcolor, markersize=default["markersize"])
                    #     # ax.scatter(yr, yz, alpha=1, s=5)
                    #     ax.scatter(yr, yz, color=color[i], alpha=1, s=5)

        return fig, ax