from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt
from simsopt._core import load
from horus import poincare
# import pandas as pd
import numpy as np
from pathlib import Path
import sys

latexplot_folder = Path("../../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()

sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_simsopt

surfaces, ma, coils = load(f'serial0229079.json')

# Setting the problem
R0, _, Z0 = ma.gamma()[0,:]
ps = SimsoptBfieldProblem.from_coils(R0=R0, Z0=Z0, Nfp=3, coils=coils, interpolate=True, ncoils=3, mpol=5, ntor=5, n=40)

# Poincare plot
nfp = 3
phis = [i * (2 * np.pi / nfp) for i in range(nfp)]

nfieldlines = 40
Rs = np.linspace(0.869, 1.05, nfieldlines)
Zs = np.zeros_like(Rs)
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

nfieldlines = 20
Rs = np.linspace(1.05, 1.2, nfieldlines)
Zs = np.zeros_like(Rs)
RZs2 = np.array([[r, z] for r, z in zip(Rs, Zs)])

nfieldlines = 20
Rs = np.linspace(1.05, 1.2, nfieldlines)
Zs = np.linspace(0, 0.05, nfieldlines)
RZs3 = np.array([[r, z] for r, z in zip(Rs, Zs)])

RZs = np.concatenate((RZs, RZs2, RZs3), axis=0)

pplane = poincare(ps._mf_B, RZs, phis, ps.surfclassifier, tmax = 1000, tol = 1e-11, plot=False)
pplane.save("poincare_0229079.pkl")

fig, ax = plt.subplots()
plot_poincare_simsopt(pplane.phi_hits, ax, color=None)
ax.set_xlim(0.6, 1.2)
ax.set_ylim(-0.25, 0.25)

fig.savefig(saving_folder / "poincare_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Finding all fixedpoints

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-15
pparams['niter'] = 100
# pparams["Z"] = 0 

fp_x1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp_x1.compute(guess=[1.13535758, 0.07687874], pp=3, qq=8, sbegin=0.62, send=1.2, checkonly=True)
fp_x2 = FixedPoint(ps, pparams, integrator_params=iparams)
fp_x2.compute(guess=[1.14374773, 0.0203871], pp=3, qq=8, sbegin=0.62, send=1.2, checkonly=True)

for ii, fp in enumerate([fp_x1, fp_x2]):
    results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    for rr in results11:
        ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1)
    fig.savefig(saving_folder / f"fixedpoint_0229079_{ii}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
# for fp in [fp11_o1, fp11_o2, fp11_o3, fp11_o4]:
#     results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
#     for rr in results11:
#         ax.scatter(rr[0], rr[2], marker="o", edgecolors="black", linewidths=1)


# Working on manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp_x1, fp_x2, integrator_params=iparam)
mp.choose(signs=[[1, -1], [1, -1]], order=False)

mp.compute(nintersect = 6, epsilon= 1e-3, neps = 30)

ax.set_xlim(1.11, 1.17)
ax.set_ylim(0.01, 0.09)

mp.plot(ax=ax, directions="isiu")
fig.savefig(saving_folder / "manifold_inner_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

mp.plot(ax=ax, directions="osou")
fig.savefig(saving_folder / "manifold_outer_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Inner manifold
print("Working on Inner manifold")
mp.onworking = mp.inner
mp.find_clinic_single(0.0004935714362365777, 0.0009447855326874471, n_s=5, n_u=5, jac=False)
mp.find_clinic_single(0.0007316896189876429, 0.001378277537890097, n_s=4, n_u=5, jac=False)
mp.turnstile_area()

marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
confns = mp.onworking["find_clinic_configuration"]
n_u = confns["n_u"]+confns["n_s"]+2

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=20, label=f'$h_{i+1}$')

fig.savefig(saving_folder / "clinic_inner_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Outer manifold
print("Working on Outer manifold")
mp.onworking = mp.outer
mp.find_clinic_single(0.0005138103059726661, 0.00045678777236184576, n_s=6, n_u=5, jac=False)
mp.find_clinic_single(0.0007397342403824656, 0.0006577310819452291, n_s=5, n_u=5, jac=False)
mp.turnstile_area()

inner_areas = mp.inner["areas"]
np.save("inner_areas_0229079.npy", inner_areas)
outer_areas = mp.outer["areas"]
np.save("outer_areas_0229079.npy", outer_areas)
# ps.B([ps._R0, 0., ps._Z0])[1]*ps._R0

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="red", edgecolor='cyan', zorder=20, label=f'$h_{i+1}$')

fig.savefig(saving_folder / "clinic_outer_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Convergence figure

fig_conv, ax_conv = plt.subplots()

# ar = np.zeros((2, 3))
for ii, pot in enumerate(mp.inner["potential_integrations"]):
    ns = min(len(pot[0]), len(pot[1]))
    # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
    ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=20)

for ii, pot in enumerate(mp.outer["potential_integrations"]):
    ns = min(len(pot[0]), len(pot[1]))
    # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
    ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=20)

ax_conv.set_xlabel('Iteration')
ax_conv.set_ylabel('Potential integration')

fig_conv.savefig(saving_folder / "convergence_0229079.png", dpi=300, bbox_inches="tight", pad_inches=0.1)