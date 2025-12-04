#!/usr/bin/env python

import numpy as np
from simsopt.configs import get_data
from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, PoincarePlot, Manifold
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

###############################################################################
# Define the NCSX cpnfiguration and set up the pyoculus problem
###############################################################################

curves, currents, ma, nfp, bs = get_data('ncsx')

mf = SimsoptBfield(nfp, bs)
ncsx_map = CylindricalBfieldSection.without_axis(mf, guess=[1.6, 0], method='scipy.root')

# find the o-point of the 3/7 island chain
islando = FixedPoint(ncsx_map)
islando.find(7, [1.70, 0.0], method='scipy.root',  options={'factor':1e-2})

# find the x-point of this chain
islandx = FixedPoint(ncsx_map)
islandx.find(7, [1.66, 0.21], method='scipy.root')

#get starting points for a Poincare section calculation
linepoints = np.linspace(ncsx_map.axis, islando.coords[0], 5)
extra  = islando.coords[0]+np.array([0.05, 0])
linepoints = np.vstack((linepoints, extra))

#define and compute the section
pplot = PoincarePlot(ncsx_map, linepoints)
pplot.compute(npts=200)

# get a second fixed point for defining the manifold
islandx2 = FixedPoint(ncsx_map)
islandx2.find(7, islandx.coords[-2], method='scipy.root')
manifold = Manifold(ncsx_map, islandx, islandx2, dir1=np.array([1, -1.]), dir2=np.array([1, 1.]), first_stable=True)
manifold.compute(eps_s=1e-4, eps_u=1e-4, neps_s=4, neps_u=4, nint_s=50, nint_u=50)
manifold.plot(markersize=0)


fig, ax = pplot.plot(s=0.7, linewidths=0)
ax.set_aspect('equal')
fig.show()




