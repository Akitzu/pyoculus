from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold
from simsopt._core import load
import logging
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import sys

################
# set up
################

logging.basicConfig(level=logging.INFO)
surfaces, ma, coils = load(f'coils/serial0229079.json')

# ##
# # Poincare plot
# nfp = 3
# phis = [i * (2 * np.pi / nfp) for i in range(nfp)]

# nfieldlines = 40
# Rs = np.linspace(0.869, 1.05, nfieldlines)
# Zs = np.zeros_like(Rs)
# RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# nfieldlines = 20
# Rs = np.linspace(1.05, 1.2, nfieldlines)
# Zs = np.zeros_like(Rs)
# RZs2 = np.array([[r, z] for r, z in zip(Rs, Zs)])

# nfieldlines = 20
# Rs = np.linspace(1.05, 1.2, nfieldlines)
# Zs = np.linspace(0, 0.05, nfieldlines)
# RZs3 = np.array([[r, z] for r, z in zip(Rs, Zs)])

# RZs = np.concatenate((RZs, RZs2, RZs3), axis=0)

# ax.set_xlim(0.6, 1.2)
# ax.set_ylim(-0.25, 0.25)

# Setting the problem
simsoptfield = SimsoptBfield.from_coils(coils, Nfp=3, interpolate=True, ncoils=3, mpol=5, ntor=5, n=40)
pyoproblem = CylindricalBfieldSection.without_axis(simsoptfield, guess=ma.gamma()[0,::2], rtol=1e-12)

# Fixed points Search
fp_x1 = FixedPoint(pyoproblem)
fp_x1.find(8, guess=[1.13535758, 0.07687874])
fp_x2 = FixedPoint(pyoproblem)
fp_x2.find(8, guess=[1.14374773, 0.0203871])

# for fp in enumerate([fp_x1, fp_x2]):
#     fp.plot(ax=ax, marker="X", edgecolors="black", linewidths=1)


# quick fix
fp_x1._m = 8
fp_x2._m = 8
fp_x1._found_by_iota = True
fp_x2._found_by_iota = True

# Working on manifold
mp = Manifold(pyoproblem, fp_x1, fp_x2)
mp.choose(signs=[[1, -1], [1, -1]], order=False)

# mp.compute(nintersect = 6, epsilon= 1e-3, neps = 30)
# ax.set_xlim(1.11, 1.17)
# ax.set_ylim(0.01, 0.09)
# mp.plot(ax=ax, directions="isiu")
# mp.plot(ax=ax, directions="osou")

# Inner manifold
print("Working on Inner manifold")
mp.onworking = mp.inner
mp.find_clinic_single(0.0004935714362365777, 0.0009447855326874471, n_s=5, n_u=5, jac=False)
mp.find_clinic_single(0.0007316896189876429, 0.001378277537890097, n_s=4, n_u=5, jac=False)
mp.turnstile_area()

# Outer manifold
print("Working on Outer manifold")
mp.onworking = mp.outer
mp.find_clinic_single(0.003534624002967668, 0.0006573605725165735, n_s=4, n_u=4, jac=False)
mp.find_clinics(n_points=2)
mp.turnstile_area()

# inner_areas = mp.inner["areas"]
# np.save("inner_areas_0229079.npy", inner_areas)
# outer_areas = mp.outer["areas"]
# np.save("outer_areas_0229079.npy", outer_areas)