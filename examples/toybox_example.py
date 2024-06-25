from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, PoincarePlot
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    ### Creating the pyoculus problem object

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-2}

    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem = AnalyticCylindricalBfield.without_axis(6, 0, 0.91, 0.6, perturbations_args = [separatrix], guess=[6.41, -0.7], finderargs={"niter": 100, "tol": 1e-7})

    # Adding perturbation after the object is created uses the found axis as center point
    pyoproblem.add_perturbation(maxwellboltzmann, find_axis=False)
    pyoproblem.find_axis(guess=[pyoproblem.R0, pyoproblem.Z0], niter=100, tol=1e-7)

    ### Finding the X-point
    print("\nFinding the X-point\n")

    fp = FixedPoint(pyoproblem)
    fp.find(1, guess=[6.21560891, -4.46981856], niter=300 , tol=1e-10)
    xpoint = fp.coords[0]

    ### Compute the Poincare plot
    print("\nComputing the Poincare plot\n")

    # Set RZs for the tweaked (R-Z) computation
    frac_nf_0 = 0.96
    nfieldlines = 10
    nfieldlines, nfieldlines_3 = int(np.ceil(frac_nf_0*nfieldlines)), int(np.floor((1-frac_nf_0)*nfieldlines))+1
    
    frac_nf_1 = 2/3
    nfieldlines_1, nfieldlines_2 = int(np.ceil(frac_nf_1*nfieldlines)), int(np.floor((1-frac_nf_1)*nfieldlines))

    # Two interval computation opoint to xpoint then xpoint to coilpoint
    frac_n1 = 3/4
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_1)), int(np.floor((1 - frac_n1) * nfieldlines_1))
    opoint = np.array([pyoproblem.R0, pyoproblem.Z0])
    coilpoint = np.array(
        [pyoproblem.perturbations_args[0]["R"], pyoproblem.perturbations_args[0]["Z"]]
    )

    # Simple way from opoint to xpoint then to coilpoint
    Rs = np.concatenate((np.linspace(opoint[0]+1e-4, xpoint[0], n1), np.linspace(xpoint[0], coilpoint[0]-1e-4, n2)))
    Zs = np.concatenate((np.linspace(opoint[1]+1e-4, xpoint[1], n1), np.linspace(xpoint[1], coilpoint[1]-1e-4, n2)))
    RZs_1 = np.array([[r, z] for r, z in zip(Rs, Zs)])

    # Sophisticated way more around the xpoint
    frac_n1 = 1/2
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_2)), int(np.floor((1 - frac_n1) * nfieldlines_2))
    deps = 0.05
    RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
        opoint - xpoint
    ).reshape((1, 2))
    RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
        coilpoint - xpoint
    ).reshape((1, 2))
    RZs_2 = np.concatenate((RZ1, RZ2))

    # Third interval
    Rs = np.linspace(xpoint[0]+0.1, 8, nfieldlines_3)
    Zs = np.linspace(xpoint[1]-0.1, -5, nfieldlines_3)
    RZs_3 = np.array([[r, z] for r, z in zip(Rs, Zs)])

    RZs = np.concatenate((RZs_1, RZs_2, RZs_3))

    # Set up the Poincare plot object
    pplot = PoincarePlot(pyoproblem)
    pplot.compute(RZs, nPpts=100)

    ### Plotting the results
    fig, ax = pplot.plot(marker=".", s=1, xlim=[2.3, 4], ylim=[-2.9, 0.8])
    ax.scatter(pyoproblem.R0, pyoproblem.Z0, marker="o", edgecolors="black", linewidths=1)
    ax.scatter(*xpoint, marker="X", edgecolors="black", linewidths=1)
    fig.show()

    ### Manifold computation
