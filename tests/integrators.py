import numpy as np
import unittest
from pyoculus.integrators.scipyode_integrator import ScipyODEIntegrator

class TestODEIntegrator(unittest.TestCase):
    def test_exponential_growth(self):
        # Define the differential equation
        def dydt(t, y):
            return y  # This corresponds to k=1

        # Initial condition
        yi = 1.0
        # Time interval for integration
        ti, tf = 0, 1
        # Create an instance of the ODEIntegrator
        integrator = ScipyODEIntegrator(ode=dydt)
        
        # Solve the differential equation
        integrator.set_initial_value(ti, yi)
        yf = integrator.integrate(tf)
        
        # Expected solution at t = tf
        expected_solution = yi * np.exp(tf)

        # Check if the numerical solution at tf is close to the expected solution
        self.assertAlmostEqual(yf[0], expected_solution, places=3, msg="Numerical solution does not match expected solution within tolerance.")

if __name__ == '__main__':
    unittest.main()