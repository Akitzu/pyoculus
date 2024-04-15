from ..problems.bfield_problem import BfieldProblem
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from scipy.optimize import newton_krylov
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

        # Compute the eigenvalues and eigenvectors of the fixed point
        eigRes = np.linalg.eig(fixedpoint.jacobian)
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

    def compute(self, epsilon = None, **kwargs):
        options = {
            "eps_guess": 2e-6,
            "nintersect": 10,
            "neps": 2,
            "directions": "u+u-s+s-"
        }
        options.update({key: value for key, value in kwargs.items() if key in options})
        
        rz_fixedpoint = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]])

        if epsilon is None:
            epsilon = self.find_epsilon(options['eps_guess'], self.vector_u)
        
        RZs = self.start_config(epsilon, self.vector_u, options['neps'])[0]
        if 'u+' in options['directions']:
            print("Computing unstable manifold with postive epsilon...")
            self.unstable['+'] = self.integrate(RZs, nintersect=options['nintersect'])

        if 'u-' in options['directions']:
            print("Computing unstable manifold with negative epsilon...")
            RZs  = 2*rz_fixedpoint - RZs
            self.unstable['-'] = self.integrate(RZs, nintersect=options['nintersect'])
        
        RZs = self.start_config(epsilon, self.vector_s, options['neps'], -1)[0]
        if 's+' in options['directions']:
            print("Computing stable manifold with positive epsilon...")
            self.stable['+'] = self.integrate(RZs, nintersect=options['nintersect'], direction=-1)
        
        if 's-' in options['directions']:
            print("Computing stable manifold with negative epsilon...")
            RZs = 2*rz_fixedpoint - RZs
            self.stable['-'] = self.integrate(RZs, nintersect=options['nintersect'], direction=-1)


    def start_config(self, epsilon, eigenvector, neps=10, direction=1, intervtype = "logspace"):
        rEps = np.array([self.fixedpoint.x[0], self.fixedpoint.z[0]]) + epsilon * eigenvector

        rz_path = self.integrate(np.atleast_2d(rEps), 1, direction)
        
        eps_dir = rz_path[:,1]-rz_path[:,0]
        eps_dir_norm = eps_dir / np.linalg.norm(eps_dir)
        
        if intervtype == "logspace":
            eps = np.logspace(-6, 0, neps)
            Rs = eps * eps_dir[0] + rEps[0]
            Zs = eps * eps_dir[1] + rEps[1]
        elif intervtype == "linspace":
            Rs = np.linspace(rz_path[0,0], rz_path[0,1], neps)
            Zs = np.linspace(rz_path[1,0], rz_path[1,1], neps)
        else:
            raise ValueError("Invalid interval type")
        
        RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])
        return RZs, np.linalg.norm(eps_dir_norm - eigenvector)
    

    def find_epsilon(self, eps_guess, eigenvector, iter = 4):
        find_eps = lambda x: self.start_config(x, eigenvector, 1)[1]

        # eps_root = fsolve(find_eps, eps_guess, xtol=options['xtol'])
        try:
            eps_root = newton_krylov(find_eps, eps_guess, inner_maxiter=iter)[0]
            print(f"Newton-Krylov succeeded, epsilon = {eps_root}")
        except:
            print("Newton-Krylov failed, using the guess for epsilon.")
            return eps_guess

        return eps_root

    def integrate(self, RZstart, nintersect, direction = 1):
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

    def plot(self, ax = None, directions = "u+u-s+s-"):
        color = ['red', 'blue', 'green', 'purple']
        
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
                    for yr, yz in zip(out[::2], out[1::2]):
                            # ax.scatter(yr, yz, alpha=1, s=5)
                            ax.scatter(yr, yz, color=color[i], alpha=1, s=5)

        return fig, ax