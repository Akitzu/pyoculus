from pyoculus.integrators import RKIntegrator
import numpy as np

def linearized_error(fun, rtol = 1e-10, **kwargs):
    iparams = dict()
    iparams["rtol"] = rtol
    iparams["ode"] = fun

    integrator = RKIntegrator(iparams)

    initpoint = np.array([2.2, -1.6])
    v = 2e-5*np.random.random(2)

    ic = np.array([initpoint[0], initpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    M = integrator.integrate(2*np.pi)
    endpoint1 = M[0:2]
    M = M[2:6].reshape((2,2)).T
    print(M)

    inputpoint = initpoint + v
    ic = np.array([inputpoint[0], inputpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    endpoint2 = integrator.integrate(2*np.pi)[0:2]

    return np.linalg.norm(((M @ v) - (endpoint2 - endpoint1))/np.linalg.norm(endpoint2 - endpoint1))

# if __name__ == "__main__":
#     pass