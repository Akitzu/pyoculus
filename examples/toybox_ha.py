import numpy as np
from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

bfield = AnalyticCylindricalBfield(6, 0, 1, 0.5)
section = CylindricalBfieldSection(bfield, R0=6, Z0=0, rtol=1e-13)

## Adding the maxwellian perturbation
maxwellboltzmann = {"R": section.R0, "Z": section.Z0,
                    "m": 3, "n": -2, "d": 1.75/np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-1}
bfield.add_perturbation(maxwellboltzmann)
section.clear_cache()
section.find_axis()

## Finding the fixed points
fp_x1 = FixedPoint(section)
guess = [5.05, 0]
fp_x1.find(3, guess)
# fp01.find_with_iota(0, 1, guess)
fp_x1._found_by_iota = True
fp_x1._m = 3

## Computing the poincare plot
pplot = PoincarePlot.with_horizontal(section, 1, 10)
pplot.compute(compute_iota=False)
fig, ax = pplot.plot()
ax.scatter(section.R0, section.Z0)
fp_x1.plot(ax=ax)

# pplot.compute_iota()
# import matplotlib.pyplot as plt
# plt.plot(np.linalg.norm(pplot.xs, axis=1), 1 / pplot.iota)

# you need this for now to get the two fixed points and use them in the manifold class
fp_x2 = FixedPoint(section)
fp_x2.find(3, fp_x1.coords[1])
# fp01.find_with_iota(0, 1, guess)
fp_x2._found_by_iota = True
fp_x2._m = 3


##########################
## Manifold computation ##
##########################

from pyoculus.solvers import Manifold

Manifold.show_directions(fp_x1, fp_x2)
plt.show()

# Outer manifold
outer_manifold = Manifold(section, fp_x1, fp_x2, '+', '-', False)

eps_guess_s = outer_manifold.find_epsilon("stable")
eps_guess_u = outer_manifold.find_epsilon("unstable")
# outer_manifold.find_clinics(eps_guess_s, eps_guess_u, n_points=4, shift=2)

# outer_manifold.compute()
# manif.compute(directions="both", nintersect=6, neps=50)

inner_manifold = Manifold(section, fp_x1, fp_x2, '-', '-', True)

# eps_guess_s = inner_manifold.find_epsilon("stable")
# eps_guess_u = inner_manifold.find_epsilon("unstable")
# inner_manifold.find_clinics(eps_guess_s, eps_guess_u, n_points=4, shift=0)

# manif.plot(ax=ax)
# fig
# ### Inner turnstile
# # Finding the clinics
# i, s_shift = 6, 2
# n_s, n_u = i+s_shift, i-s_shift
# manif.onworking = manif.inner
# manif.find_clinic_single(n_s=n_s, n_u=n_u)
# manif.find_clinics(n_points=4)
# manif.plot_clinics(ax=ax)
# fig
# manif.turnstile_area()
# ### Outer turnstile
# # Finding the clinics
# s_shift = 1
# n_s, n_u = 6+s_shift, 6-s_shift
# manif.onworking = manif.outer
# manif.find_clinic_single(n_s=n_s, n_u=n_u)
# manif.find_clinics(n_points=4)
# manif.plot_clinics(ax=ax)
# fig
# manif.turnstile_area()