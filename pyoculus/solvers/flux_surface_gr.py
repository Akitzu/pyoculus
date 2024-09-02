## @file flux_surface_gr.py
#  @brief A class for finding flux surfaces using Greene's residue method
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .base_solver import BaseSolver
from .fixed_point import FixedPoint
import pyoculus.irrationals as ir
import numpy as np

## Class that used to set up the flux surface finder.
class FluxSurfaceGR(BaseSolver):
    def __init__(
        self, problem, params=dict(), integrator=None, integrator_params=dict()
    ):
        """
        Set up the class of the flux surface point finder using Greene's method
        """

        if "theta" not in params.keys():
            raise ValueError(
                "We only support located fixed points for a fixed theta at the moment"
            )

        super().__init__(
            problem=problem,
            params=params,
            integrator=integrator,
            integrator_params=integrator_params,
        )

    def compute(
        self,
        iota,
        n_expand=10,
        nstart=5,
        sguess=0.0,
        fixed_point_left=None,
        fixed_point_right=None,
        tol=None,
    ):
        """
        Look for the flux surface with a given rotation number using Greene's residue method.

        Args:
            iota: the irrational! rotation number of the flux surface
            fixed_point_left: a sucessfully found FixPoint to mark the left bound of the flux surface,
                                its rotation number needs to be in the convergent sequence of iota
            fixed_point_right a sucessfully found FixPoint to mark the right bound of the flux surface,
                                its rotation number needs to be in the convergent sequence of iota and next to fixed_point_left
            n_expand=10 the number of terms in the continued fraction expansion of iota, used to approximate the flux surface

        @returns  a class that contains the results
            `fdata.MackayResidue` -- the Mackay Residue of the fixed points
            `fdata.fixed_points` -- all the fixed point located
            `fdata.rmnc`, fdata.rmns`, `fdata.zmnc`, `fdata.zmns` -- the Fourier harmonics
        """

        # iota will be divided by Nfp
        iota = iota / self.Nfp

        # continued fraction expansion of the input irrational
        ai = ir.expandcf(iota, n_expand + 1)

        fpleft = None
        fpright = None

        # determine how the input fixed points fit into the sequence of expansion
        for ii in range(n_expand - 1):
            ppqq1 = ir.fromcf(ai[0 : ii + 1])
            pp1 = ppqq1[0]
            qq1 = ppqq1[1]

            ppqq2 = ir.fromcf(ai[0 : ii + 2])
            pp2 = ppqq2[0]
            qq2 = ppqq2[1]

            # put the lower order fixed point as fpleft and higher order fpright
            if (
                pp1 == fixed_point_left.pp
                and qq1 == fixed_point_left.qq
                and pp2 == fixed_point_right.pp
                and qq2 == fixed_point_right.qq
            ):
                fpleft = fixed_point_left
                fpright = fixed_point_right
                nstart = ii
            elif (
                pp2 == fixed_point_left.pp
                and qq2 == fixed_point_left.qq
                and pp1 == fixed_point_right.pp
                and qq1 == fixed_point_right.qq
            ):
                fpleft = fixed_point_right
                fpright = fixed_point_left
                nstart = ii

        if fpleft is None or fpright is None:
            raise ValueError("The input fixed points are illegal")

        fixedpoints = [fpleft, fpright]
        self.nstart = nstart

        for ii in range(nstart + 2, n_expand):

            ppqq = ir.fromcf(ai[: ii + 1])
            pp = ppqq[0]
            qq = ppqq[1]
            iotatarget = float(pp) / float(qq)

            nextfixedpoint = FixedPoint(
                self._problem,
                params=self._params,
                integrator=self._integrator_type,
                integrator_params=self._integrator_params,
            )

            sleft = fixedpoints[-2].s[0]
            iotaleft = float(fixedpoints[-2].pp) / float(fixedpoints[-2].qq)
            sright = fixedpoints[-1].s[0]
            iotaright = float(fixedpoints[-1].pp) / float(fixedpoints[-1].qq)

            # interpolate between sleft and sright to get the next guess of s
            sguess = sleft + (sright - sleft) / (iotaright - iotaleft) * (
                iotatarget - iotaleft
            )

            # the lower and upper bound of s range
            ssmall = np.min([sleft, sright])
            sbig = np.max([sleft, sright])

            fp = nextfixedpoint.compute(
                sguess, pp, qq, sbegin=ssmall, send=sbig, tol=tol
            )

            if not nextfixedpoint.successful:
                raise Exception("Fixed point not found")

            fixedpoints.append(nextfixedpoint)

        # save the fixed points found
        self.fixedpoints = fixedpoints

        # assemble the output data
        fdata = FluxSurfaceGR.OutputData()
        fdata.fixedpoints = fixedpoints

        # put the flag as successful
        self.successful = True

        return fdata

    def plot(
        self, plottype=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs
    ):
        """! Generates the plot for flux surface
        @param plottype which variables to plot: 'RZ' or 'yx', by default using "poincare_plot_type" in problem
        @param xlabel,ylabel what to put for the xlabel and ylabel, by default using "poincare_plot_xlabel" in problem
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """
        import matplotlib.pyplot as plt

        if not self.successful:
            raise Exception("A successful call of compute() is needed")

        self.fixedpoints[-1].plot(plottype, xlabel, ylabel, xlim, ylim, **kwargs)

    def plot_residue(self):
        """! Generate the plot for residue"""
        import matplotlib.pyplot as plt

        gamma = (np.sqrt(5) + 1) / 2

        xlist_greene = np.arange(
            self.nstart + 1, self.nstart + 1 + len(self.fixedpoints)
        )
        greenes_list = np.zeros(len(self.fixedpoints), dtype=np.float64)

        for ii, fp in enumerate(self.fixedpoints):
            greenes_list[ii] = fp.GreenesResidue

        xlist_Mackay = xlist_greene[1:]
        Mackay_list = np.zeros(len(self.fixedpoints) - 1, dtype=np.float64)

        for ii in range(len(self.fixedpoints) - 1):
            Mackay_list[ii] = (
                self.fixedpoints[ii].GreenesResidue
                + gamma * self.fixedpoints[ii + 1].GreenesResidue
            ) / (1.0 + gamma)

        fig, ax = plt.subplots()

        geplot = ax.plot(xlist_greene, greenes_list, '-x', label="Greene")
        #mcplot = ax.plot(xlist_Mackay, Mackay_list, '-o', label="Mackay")
        mcplot = ax.plot(
            xlist_greene, 0.25 * np.ones_like(greenes_list), label="Stable bound"
        )

        ax.legend()

        plt.xlabel("Order of fixed point", fontsize=20)
        plt.ylabel("Residue", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
