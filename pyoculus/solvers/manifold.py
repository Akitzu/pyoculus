from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..problems.bfield_problem import BfieldProblem
from ..problems.cartesian_bfield import CartesianBfield
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov
import numpy as np

class Manifold(BaseSolver):
    def __init__(self, fixedpoint, bfield, params=dict(), integrator=None, integrator_params=dict()):
        
        # Check that the fixed point is a correct FixedPoint instance
        assert isinstance(fixedpoint, FixedPoint), "Fixed point must be an instance of FixedPoint class"
        assert fixedpoint.successful, "Need a successful fixed point to compute the manifold"

        self.fixedpoint = fixedpoint

        # Compute the eigenvalues and eigenvectors of the fixed point
        eigRes = fixedpoint.compute_eig()
        eigenvalues = eigRes[0]
        # Eigenvectors are stored as columns of the matrix eigRes[1], transposing it to access them as np.array[i]
        eigenvectors = eigRes[1].T
        s_index, u_index = 0, 1
        if eigenvalues[0].real > eigenvalues[1].real:
            s_index, u_index = 1, 0

        self.vector_u = eigenvectors[u_index]
        self.lambda_u = eigenvalues[u_index]
        self.vector_s = eigenvectors[s_index]
        self.lambda_s = eigenvalues[s_index]

        # Check that the bfield is a correct BfieldProblem instance
        assert isinstance(bfield, BfieldProblem), "Bfield must be an instance of BfieldProblem class"
        self.bfield = bfield

        # Integrator and BaseSolver initialization
        integrator_params["ode"] = bfield.f_tangent

        # Setting the needed parameters
        if "solve_ivp" not in params:
            params["solve_ivp"] = True

        self._start_config_params = {
            "phi": 0,
            "integrate_ivp_kwargs": {
                "atol": 1e-22, "rtol": 3e-14, "nintersect": 1, "method": "DOP853"
            },
        }
        self._start_config_params.update({key: value for key, value in params.items() if key in self._start_config_params})

        params = {key: value for key, value in params.items() if key not in self._start_config_params}

        self.unstable = {'+': None, '-': None}
        self.stable = {'+': None, '-': None}

        super().__init__(
            problem=bfield,
            params=params,
            integrator=integrator,
            integrator_params=integrator_params,
        )

    def compute(self, **kwargs):
        if self._params["solve_ivp"] == True:
            return self.compute_ivp(**kwargs)
        else:
            pass

    def compute_ivp(self, **kwargs):
        options = {
            "eps_guess": 2e-6,
            "nintersect": 10,
            "atol": 1e-9,
            "rtol": 1e-5,
            "neps": 10,
            "directions": "u+u-s+s-"
        }
        options.update({key: value for key, value in kwargs.items() if key in options})
        
        rz_fixedpoint = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]])

        epsilon = self.find_epsilon(options['eps_guess'], self.vector_u)
        RZs = self.start_config(epsilon, self.vector_u, options['neps'])[0]

        if 'u+' in options['directions']:
            self.unstable['+'] = self.integrate_ivp(RZs, [0], nintersect=options['nintersect'], atol = options['atol'], rtol = options['rtol'])

        RZs  = 2*rz_fixedpoint - RZs
        if 'u-' in options['directions']:
            self.unstable['-'] = self.integrate_ivp(RZs, [0], nintersect=options['nintersect'], atol = options['atol'], rtol = options['rtol'])
        
        RZs = self.start_config(epsilon, self.vector_s, options['neps'])[0]
        if 's+' in options['directions']:
            self.stable['+'] = self.integrate_ivp(RZs, [0], nintersect=options['nintersect'], atol = options['atol'], rtol = options['rtol'], direction=-1)
        
        RZs = 2*rz_fixedpoint - RZs
        if 's-' in options['directions']:
            self.stable['-'] = self.integrate_ivp(RZs, [0], nintersect=options['nintersect'], atol = options['atol'], rtol = options['rtol'], direction=-1)


    def start_config(self, epsilon, eigenvector, neps=10):
        options = self._start_config_params
        rEps = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]]) + epsilon * eigenvector
        out = self.integrate_ivp(np.atleast_2d(rEps), [options['phi']], **options['integrate_ivp_kwargs'])
        
        eps_dir = out.y[:,1]-out.y[:,0]
        eps_dir = eps_dir / np.linalg.norm(eps_dir)
        
        print(epsilon, eps_dir, eigenvector, np.dot(eps_dir, eigenvector) - 1)

        Rs = np.linspace(out.y[0,0], out.y[0,1], neps, endpoint=False)
        Zs = np.linspace(out.y[1,0], out.y[1,1], neps, endpoint=False)
        RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

        return RZs, np.dot(eps_dir, eigenvector) - 1
    

    def find_epsilon(self, eps_guess, eigenvector, iter = 4):
        find_eps = lambda x: self.start_config(x, eigenvector, 1)[1]

        # eps_root = fsolve(find_eps, eps_guess, xtol=options['xtol'])
        try :
            eps_root = newton_krylov(find_eps, eps_guess, inner_maxiter=iter)[0]
        except:
            print("Newton-Krylov failed, using the guess for epsilon.")
            return eps_guess

        return eps_root

    def integrate_ivp(self, RZstart, phis, **kwargs):
        options = {
            "rtol": 1e-7,
            "atol": 1e-8,
            "nintersect": 10,
            "method": "DOP853",
            "direction": 1,
        }
        options.update(kwargs)

        assert RZstart.shape[1] == 2, "RZstart must be a 2D array with shape (n, 2)"
        assert len(phis) > 0, "phis must be a list of floats with at least one element"
        assert isinstance(options["nintersect"], int) and options["nintersect"] > 0, "nintersect must be a positive integer"
        assert options["direction"] in [-1, 1], "direction must be either -1 or 1"

        def Bfield_2D(t, rzs):
            rzs = rzs.reshape((-1, 2))
            phis = (t % (2 * np.pi)) * np.ones(rzs.shape[0])
            bs_Bs = self.bfield.B_many(rzs[:, 0]*np.cos(phis), rzs[:, 0]*np.sin(phis), rzs[:, 1])
            rphizs = np.array([rzs[:, 0], phis, rzs[:, 1]]).T

            Bs = list()
            for position, B in zip(rphizs, bs_Bs):
                B = CartesianBfield._inv_Jacobian(*position) @ B.reshape(3, -1)
                Bs.append(np.array([B[0, 0] / B[1, 0], B[2, 0] / B[1, 0]]))

            return options['direction']*np.array(Bs).flatten()
        
        # setup the phis of the poincare sections
        phis = np.unique(np.mod(phis, 2 * np.pi / self.bfield.Nfp))
        phis.sort()

        # setup the evaluation points for those sections
        phi_evals = np.array(
            [
                phis + self.fixedpoint.qq * 2 * np.pi * i / self.bfield.Nfp
                for i in range(options["nintersect"] + 1)
            ]
        )

        out = solve_ivp(
            Bfield_2D,
            [0, phi_evals[-1, -1]],
            RZstart.flatten(),
            t_eval=phi_evals.flatten(),
            method=options["method"],
            atol=options["atol"],
            rtol=options["rtol"],
        )

        return out