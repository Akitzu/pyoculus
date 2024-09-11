from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt
from simsopt._core import load
from horus import poincare
import pandas as pd
import numpy as np
from pathlib import Path
from horus import plot_poincare_simsopt
import pickle

saving_folder = Path("figs").absolute()

surfaces, ma, coils = load(f'serial1329594.json')

nfp = 3
s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=5,
    ntor=5,
    stellsym=True,
    nfp=3,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(ma, 0.4, flip_theta=False)

# Setting the problem
R0, _, Z0 = ma.gamma()[0,:]
ps = SimsoptBfieldProblem.from_coils(R0=R0, Z0=Z0, Nfp=3, coils=coils, interpolate=True, surf=s)

# Poincare plot
phis = [i * (2 * np.pi / nfp) for i in range(nfp)]

nfieldlines = 40
Rs = np.linspace(0.72, 0.967, nfieldlines)
Zs = np.zeros_like(Rs)
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# pplane = poincare(ps._mf_B, RZs, phis, ps.surfclassifier, tmax = 15000, tol = 1e-10, plot=False)
# pplane.save("data/poincare_1329594.pkl")

tys, phi_hits = pickle.load(open("data/poincare_1329594.pkl", "rb"))

fig, ax = plt.subplots()
plot_poincare_simsopt(phi_hits, ax)

# Finding all fixedpoints

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-18
pparams['niter'] = 100
# pparams["Z"] = 0 

fp_x1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp_x1.compute(guess=[0.815, 0.016], pp=3, qq=5, sbegin=0.62, send=1.2, checkonly=True)
fp_x2 = FixedPoint(ps, pparams, integrator_params=iparams)
fp_x2.compute(guess=[0.815, -0.016], pp=3, qq=5, sbegin=0.62, send=1.2, checkonly=True)

for fp in [fp_x1]:
    results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    for rr in results11:
        ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1)

fig.savefig(saving_folder / "poincare_1329594.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
fig.savefig(saving_folder / "poincare_1329594.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)

# for fp in [fp11_o1, fp11_o2, fp11_o3, fp11_o4]:
#     results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
#     for rr in results11:
#         ax.scatter(rr[0], rr[2], marker="o", edgecolors="black", linewidths=1)

# Working on manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp_x1, fp_x2, integrator_params=iparam)
mp.choose(signs=[[-1, -1], [-1, 1]])

mp.compute(nintersect = 13, epsilon= 1e-3, neps = 30, directions="outer")
mp.compute(nintersect = 7, epsilon= 1e-3, neps = 30, directions="inner")

mp.plot(ax=ax)
# fig.savefig(saving_folder / "manifold_.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Inner manifold
print("Working on Inner manifold")
mp.onworking = mp.inner
mp.find_clinic_single(0.0005513288491534374, 0.000551269621851158, n_s=9, n_u=2)
mp.find_clinics(n_points=2)
mp.turnstile_area(False)

marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
confns = mp.onworking["find_clinic_configuration"]
n_u = confns["n_u"]+confns["n_s"]+2

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

# Outer manifold
print("Working on Outer manifold")
mp.onworking = mp.outer
mp.find_clinic_single(0.0009706704534637185, 0.0009706469632955946, n_s=3, n_u=13)
mp.find_clinic_single(0.0014129766700563878, 0.0014129861303559278, n_s=3, n_u=12)
mp.turnstile_area(False)

confns = mp.onworking["find_clinic_configuration"]
n_u = confns["n_u"]+confns["n_s"]+2

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="red", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

fig.savefig(saving_folder / "homoclinics_1329594.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

inner_areas = mp.inner["areas"]
np.save("data/inner_areas_1329594.npy", inner_areas)
outer_areas = mp.outer["areas"]
np.save("data/outer_areas_1329594.npy", outer_areas)

# Convergence figure

# fig_conv, ax_conv = plt.subplots()

# # ar = np.zeros((2, 3))
# for ii, pot in enumerate(mp.inner["potential_integrations"]):
#     ns = min(len(pot[0]), len(pot[1]))
#     # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
#     ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=10)

# for ii, pot in enumerate(mp.outer["potential_integrations"]):
#     ns = min(len(pot[0]), len(pot[1]))
#     # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
#     ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=10)

# ax_conv.set_xlabel('Iteration')
# ax_conv.set_ylabel('Potential integration')