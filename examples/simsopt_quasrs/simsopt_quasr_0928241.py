from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold
from simsopt.geo import SurfaceRZFourier
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
surfaces, ma, coils = load(f'coils/serial0928241.json')

s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=5,
    ntor=5,
    stellsym=True,
    nfp=3,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(ma, 0.7, flip_theta=False)

## 
# # Poincare plot
# nfp = 3
# phis = [i* (2 * np.pi / 3) for i in range(nfp)]

# nfieldlines = 10
# Rs = np.linspace(0.884, 1.2, nfieldlines)
# Zs = np.zeros_like(Rs)
# RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# nfieldlines = 60
# p1 = np.array([1.09955, 0.0712])
# p2 = np.array([1.4016, 0.1072])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# # Rs, Zs = np.meshgrid(Rs, Zs)
# RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
# RZs = np.concatenate((RZs, RZs2))

# nfieldlines = 10
# p1 = np.array([1.385, 0.])
# p2 = np.array([1.526, 0.])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
# RZs = np.concatenate((RZs, RZs2))

# nfieldlines = 10
# p1 = np.array([1.4446, 0.])
# p2 = np.array([1.4822, 0.])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
# RZs = np.concatenate((RZs, RZs2))

# pplane = poincare(ps._mf_B, RZs, phis, ps.surfclassifier, tmax = 1000, tol = 1e-11, plot=False)
# pplane.save("poincare_0928241.pkl")

# tys, phi_hits = pickle.load(open("poincare_0928241.pkl", "rb"))
# fig, ax = plt.subplots()
# plot_poincare_simsopt(phi_hits, ax, color=None)
# ax.set_xlim(0.3, 1.6)
# ax.set_ylim(-0.35, 0.35)

# fig.savefig(saving_folder / "poincare_0928241.png", dpi=600, bbox_inches="tight", pad_inches=0.1)

# Setting the problem
simsoptfield = SimsoptBfield.from_coils(coils, Nfp=3, interpolate=True, surf=s)
pyoproblem = CylindricalBfieldSection.without_axis(simsoptfield, guess=ma.gamma()[0,::2], rtol=1e-13)

# Finding all fixedpoints
fp11_o1 = FixedPoint(pyoproblem)
fp11_o1.find(6, guess=[1.4446355574662593, 0.0], tol=1e-15)
fp11_o2 = FixedPoint(pyoproblem)
fp11_o2.find(6, guess=[1.346295615988142, 0.2133036397909969], tol=1e-15)
fp11_o3 = FixedPoint(pyoproblem)
fp11_o3.find(6, guess=[1.40150403, 0.10815878], tol=1e-15)
fp11_o4 = FixedPoint(pyoproblem)
fp11_o4.find(6, guess=[1.40150403, -0.10815878], tol=1e-15)
fp11_x1 = FixedPoint(pyoproblem)
fp11_x1.find(6, guess=[1.43378117, 0.05140443], tol=1e-15)
fp11_x2 = FixedPoint(pyoproblem)
fp11_x2.find(6, guess=[1.43378117, -0.05140443], tol=1e-15)

# for i, fp in enumerate([fp11_o1, fp11_o2, fp11_o3, fp11_o4]):
#     results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
#     for rr in results11:
#         ax.scatter(rr[0], rr[2], marker="o", edgecolors="black", linewidths=1, zorder=20)
#     fig.savefig(saving_folder / f"fixedpoint_o_0928241_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# for i, fp in enumerate([fp11_x1, fp11_x2]):
#     results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
#     for rr in results11:
#         ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1, zorder=20)
#     fig.savefig(saving_folder / f"fixedpoint_ox_0928241_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# data = [
#     {'r': fp11_x1.x[0], 'z': fp11_x1.z[0], 'GreenesResidue': fp11_x1.GreenesResidue},
#     {'r': fp11_x2.x[0], 'z': fp11_x2.z[0], 'GreenesResidue': fp11_x2.GreenesResidue},
#     {'r': fp11_o1.x[0], 'z': fp11_o1.z[0], 'GreenesResidue': fp11_o1.GreenesResidue},
#     {'r': fp11_o2.x[0], 'z': fp11_o2.z[0], 'GreenesResidue': fp11_o2.GreenesResidue},
#     {'r': fp11_o3.x[0], 'z': fp11_o3.z[0], 'GreenesResidue': fp11_o3.GreenesResidue},
#     {'r': fp11_o4.x[0], 'z': fp11_o4.z[0], 'GreenesResidue': fp11_o4.GreenesResidue},
# ]
# df = pd.DataFrame(data)

# Working on manifold
mp = Manifold(pyoproblem, fp11_x2, fp11_x1)
mp.choose(signs=[[1, 1],[1, -1]])
# mp.compute(nintersect = 4, epsilon=1e-6, neps = 20)

# ax.set_xlim(1.3, 1.6)
# ax.set_ylim(-0.1, 0.1)
# mp.plot(ax=ax, directions="isiu")
# fig.savefig(saving_folder / "manifold_inner_0928241.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
# mp.plot(ax=ax, directions="osou")
# fig.savefig(saving_folder / "manifold_outer_0928241.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Inner manifold
print("Working on Inner manifold")
mp.onworking = mp.inner
mp.find_clinic_single(0.001276810579762792, 0.0012768113453997163, n_s=2, n_u=2)
mp.find_clinic_single(0.005129109370459298, 0.0051291087795083574, n_s=2, n_u = 1, tol=1e-8)
mp.turnstile_area()

# Outer manifold
print("Working on Outer manifold")
mp.onworking = mp.outer
mp.find_clinic_single(0.0015488037705831256, 0.0015488037607238807, n_s=2, n_u=2)
mp.find_clinic_single(0.0006060200774938109, 0.0006060193763593331, n_s=3, n_u=2)
mp.turnstile_area(False)

# inner_areas = mp.inner["areas"]
# np.save("inner_areas_0928241.npy", inner_areas)
# outer_areas = mp.outer["areas"]
# np.save("outer_areas_0928241.npy", outer_areas)