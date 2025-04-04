from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field.tracing import compute_fieldlines
from simsopt.util import proc0_print, comm_world
from simsopt.field import (
    particles_to_vtk,
    LevelsetStoppingCriterion,
    MinRStoppingCriterion,
    MaxRStoppingCriterion,
    MinZStoppingCriterion,
    MaxZStoppingCriterion,
)
import numpy as np
import logging
import time

######################################################################
# Define the configuration from the datafile of the 0229079 coilset. #
######################################################################

logging.basicConfig(level=logging.INFO)

surfaces, ma, coils = load(f'coils/serial1329594.json')
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

simsoptfield = SimsoptBfield.from_coils(coils, Nfp=3, interpolate=True, surf=s)

##########################################
# Setup the poincare plotting functions. #
##########################################

surf_classifier = simsoptfield.surfclassifier
bsh = simsoptfield._mf_B if simsoptfield._interpolating else None

tmax_fl = 1000  # "Time" for field line tracing
tol = 1e-15

stopping_criteria = [
    MaxZStoppingCriterion(0.6),
    MinZStoppingCriterion(-0.6),
    MaxRStoppingCriterion(1.4),
    MinRStoppingCriterion(0.4),
    # LevelsetStoppingCriterion(surf_classifier.dist),
]

logger = logging.getLogger("simsopt.field.tracing")
logger.setLevel(1)

def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    s=2,
    marker="o",
):
    fig, ax = plt.subplots()
    ax.set_aspect(aspect)
    color = None
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax.grid(True, linewidth=0.5)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    for j in range(len(fieldlines_phi_hits)):
        if fieldlines_phi_hits[j].size == 0:
            continue
        if mark_lost:
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            color = "r" if lost else "g"
        phi_hit_codes = fieldlines_phi_hits[j][:, 1]
        indices = np.where(phi_hit_codes >= 0)[0]
        data_this_phi = fieldlines_phi_hits[j][indices, :]
        if data_this_phi.size == 0:
            continue
        r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
        color = colors[j % len(colors)]
        ax.scatter(
            r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
        )

    return fig, ax


def trace_fieldlines(bfield, phi, RZs, label):
    phis = [phi + i * (2 * np.pi / nfp) for i in range(nfp)]

    t1 = time.time()
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        RZs[:, 0],
        RZs[:, 1],
        tmax=tmax_fl,
        tol=tol,
        comm=comm_world,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )
    t2 = time.time()
    proc0_print(
        f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}",
        flush=True,
    )
    if comm_world is None or comm_world.rank == 0:
        # particles_to_vtk(fieldlines_tys, __file__ + f'fieldlines_{label}')
        return fieldlines_tys, fieldlines_phi_hits

################################################################
# Setup and compute the poincare plot using simsopt fonctions. #
################################################################

phi = 0

nfieldlines = 40
Rs = np.linspace(0.72, 0.967, nfieldlines)
Zs = np.zeros_like(Rs)
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# Compute the fieldlines and plot
label = "1329594"
fieldlines_tys, fieldlines_phi_hits = trace_fieldlines(bsh, phi, RZs, label)
fig, ax = plot_poincare_data(
    fieldlines_phi_hits,
    phi,
    __file__ + f"_{label}",
    dpi=300,
    s=1,
)
ax.set_xlim(0.6, 1.2)
ax.set_ylim(-0.1, 0.1)

##########################################################################
# Setup the pyoculus map and solve for fixedpoints and turnstile fluxes. #
##########################################################################
if comm_world is not None and comm_world.rank != 0:
    comm_world.abort()

fieldmap = CylindricalBfieldSection.without_axis(simsoptfield, guess=ma.gamma()[0,::2], rtol=1e-13)

# Finding fixedpoints
fp_x1 = FixedPoint(fieldmap)
fp_x1.find(5, guess=[0.815, 0.016])
fp_x2 = FixedPoint(fieldmap)
fp_x2.find(5, guess=[0.815, -0.016])

for fp in [fp_x1]:
    fp.plot(ax=ax, linewidth=1)

# Provisional fix to set the poloidal mode number of the fixed points
fp_x1._m = 5
fp_x2._m = 5
fp_x1._found_by_iota = True
fp_x2._found_by_iota = True

# Inner manifold
inner_manifold = Manifold(fieldmap, fp_x1, fp_x2, '-', '-', True)
inner_manifold.compute(eps_s = 1e-3, eps_u = 1e-3, nint_s = 7, nint_u = 7, neps_s = 30, neps_u = 30)
inner_manifold.plot(ax=ax)

inner_manifold.find_clinics(0.0005513288491534374, 0.000551269621851158, 2)
inner_manifold.plot_clinics(ax=ax)

inner_manifold.compute_turnstile_areas()

# Outer manifold
outer_manifold = Manifold(fieldmap, fp_x1, fp_x2, '+', '-', False)
outer_manifold.compute(eps_s = 1e-3, eps_u = 1e-3, nint_s = 13, nint_u = 13, neps_s = 30, neps_u = 30)
outer_manifold.plot(ax=ax)

outer_manifold.find_clinic_single(0.0009706704534637185, 0.0009706469632955946, n_s=3, n_u=13)
outer_manifold.find_clinic_single(0.0014129766700563878, 0.0014129861303559278, n_s=3, n_u=12)
outer_manifold.plot_clinics(ax=ax)

outer_manifold.compute_turnstile_areas()

# Print the results
B_phi_axis = simsoptfield.B([fieldmap.R0, 0., fieldmap.Z0])[1]
print("Inner area (flux devided by B^\phi_axis in Tesla): ", inner_manifold.turnstile_areas / B_phi_axis / fieldmap.R0)
print("Outer area (flux devided by B^\phi_axis in Tesla): ", outer_manifold.turnstile_areas / B_phi_axis / fieldmap.R0)

# # Save the data
# np.save("inner_areas_1329594.npy", inner_manifold.turnstile_areas)
# np.save("outer_areas_1329594.npy", outer_manifold.turnstile_areas)

# Save the figure
fig.savefig("figs/simsopt_quasr_" + label + ".png", dpi=300)