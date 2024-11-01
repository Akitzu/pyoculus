# Toy-Tokamak equilibirum

import numpy as np
from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold

import logging
logging.basicConfig(level=logging.INFO)

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
bfield = AnalyticCylindricalBfield(6, 0, 0.91, 0.6, perturbations_args=[separatrix])
section = CylindricalBfieldSection.without_axis(bfield, guess=[6.41, -0.7], rtol=1e-10)

## Adding the maxwellian perturbation
maxwellboltzmann = {"R": section.R0, "Z": section.Z0, "m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-1}
bfield.add_perturbation(maxwellboltzmann)

section.clear_cache()
section.find_axis()

fp_x = FixedPoint(section)
guess = [6.21560891, -4.46981856]
fp_x.find(1, guess)
# fp01.find_with_iota(0, 1, guess)
fp_x._found_by_iota = True
fp_x._m = 1
xpoint = fp_x.coords[0]
opoint = np.array([section.R0, section.Z0])
coilpoint = np.array(
    [bfield.perturbations_args[0]["R"], bfield.perturbations_args[0]["Z"]]
)
A = opoint + 1e-4 * (xpoint - opoint) / np.linalg.norm(xpoint - opoint)
C = coilpoint - 1e-1 * (coilpoint - xpoint) / np.linalg.norm(coilpoint - xpoint)
pplot = PoincarePlot.with_segments(section, [A, xpoint, C], [20, 5])
pplot.compute(nprocess=1, compute_iota=False)

fig, ax = pplot.plot()
ax.scatter(*opoint)
fp_x.plot(ax=ax)
ax.scatter(*coilpoint)

Manifold.show_directions(fp_x, fp_x)
manif = Manifold(section, fp_x, fp_x, '+', '+', is_first_stable=True)
manif.compute(eps_s=1e-6, eps_u=1e-6, nint_s=9, nint_u=9)
manif.plot(ax=ax, rm_points=10)

eps_guess_s = manif.find_epsilon("stable")
eps_guess_u = manif.find_epsilon("unstable")

manif.find_clinics(eps_guess_s, eps_guess_u)

# manif.turnstile_area()