## @file poincare_plot.py: class for generating the Poincare Plot
#  @brief class for generating the Poincare Plot
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

import numpy as np
from .base_solver import BaseSolver
from ..problems import BfieldProblem

class PoincarePlot(BaseSolver):
    """
        Class that used to setup the Poincare plot.
    """

    def __init__(
        self, bfield_problem, zeta=0., nPpts=500, nPtrj=10, nthreads=1, **kwargs 
    ):
        """
        Sets up the Poincare plot
        
        Args:
            zeta (float, optional): The toroidal section (zeta) for the Poincare plot. Defaults to 0.
            nPpts (int, optional): The number of iterations for the plot. Defaults to 500.
            nPtrj (int, optional): The number of equidistant points in the s coordinates for the plot. Defaults to 10.
            nthreads (int, optional): The number of threads to be used for computation. Defaults to 1.
        """

        if not isinstance(bfield_problem, BfieldProblem):
            raise TypeError("bfied_problem should be a BfieldProblem.")

        super().__init__(bfield_problem)

        self.zeta = zeta
        self.nPpts = nPpts
        self.nPtrj = nPtrj
        self.nthreads = nthreads
    
        self.iota_successful = False

        # set up the result array
        self.x = np.zeros(
            [nPtrj + 1, nPpts + 1], dtype=np.float64
        )

    def compute(self, xs):
        """
        Computes the Poincare plot
        
        Returns:
            class that contains the results
        """

        self.successful = False

        swindow = self._end - self._begin
        ds = swindow / float(self.nPtrj)

        for ii in range(self.nPtrj + 1):
            # find which s to start with
            if self._is_cylindrical_problem:
                if RZs is None:
                    self.x[ii, 0] = self._begin + ds * float(ii)
                    self.z[ii, 0] = self._params["Z"]
                else:
                    self.x[ii, 0] = RZs[ii, 0]
                    self.z[ii, 0] = RZs[ii, 1]
                
                self.theta[ii, 0] = np.arctan2(
                    (self.z[ii, 0] - self._problem._Z0),
                    (self.x[ii, 0] - self._problem._R0),
                )
            else:
                self.s[ii, 0] = self._begin + ds * float(ii)
                # put in the initial conditions
                self.theta[ii, 0] = self._params["theta"]

            self.zeta[ii, 0] = self._params["zeta"]

        if self.nthreads == 1:  # single thread, do it straight away

            for ii in range(self._params["nPtrj"] + 1):

                # set up the initial value
                t0 = self.zeta[ii, 0]
                if self._is_cylindrical_problem:
                    icr = self.x[ii, 0]
                    icz = self.z[ii, 0]
                    ictheta = self.theta[ii, 0]
                    ic = [icr, icz, self._problem._R0, self._problem._Z0, ictheta]
                else:
                    ic = [self.s[ii, 0], self.theta[ii, 0]]

                # initialize the integrator
                self._integrator.set_initial_value(t0, ic)

                t = t0
                dt = self.dt

                for jj in range(1, self._params["nPpts"] + 1):
                    
                    # run the integrator
                    try:
                        st = self._integrator.integrate(t + dt)
                    except Exception:
                        # integration failed, abort for this orbit
                        print("Integration failed for s=", ic[0])
                        break

                    # extract the result to s theta zeta
                    if self._is_cylindrical_problem:
                        self.x[ii,jj] = st[0]
                        self.z[ii,jj] = st[1]
                        self.theta[ii,jj] = st[4]
                    else:
                        self.s[ii, jj] = st[0]
                        self.theta[ii, jj] = st[1]
                    self.zeta[ii, jj] = t + dt

                    # advance in time
                    t = t + dt

        self.successful = True

        pdata = PoincarePlot.OutputData()
        pdata.x = self.x.copy()
        pdata.z = self.z.copy()
        pdata.theta = self.theta.copy()
        pdata.zeta = self.zeta.copy()

        return pdata

    def compute_iota(self):
        """! Compute the iota profile"""

        if not self.successful:
            raise Exception("A successful call of compute() is needed")

        self.iota_successful = False

        # fit iota
        if self._is_cylindrical_problem:
            self.siota = self.x[:,0]
        else:
            self.siota = self.s[:, 0]

        self.iota = np.zeros_like(self.siota)
        for ii in range(self._params["nPtrj"] + 1):
            nlist = np.arange(self._params["nPpts"] + 1, dtype=np.float64)
            dzeta = 2.0 * np.pi / self.Nfp
            leastfit = np.zeros(6, dtype=np.float64)
            leastfit[1] = np.sum((nlist * dzeta) ** 2)
            leastfit[2] = np.sum((nlist * dzeta))
            leastfit[3] = np.sum((nlist * dzeta) * self.theta[ii, :])
            leastfit[4] = np.sum(self.theta[ii, :])
            leastfit[5] = 1.0

            self.iota[ii] = (leastfit[5] * leastfit[3] - leastfit[2] * leastfit[4]) / (
                leastfit[5] * leastfit[1] - leastfit[2] * leastfit[2]
            )

        self.iota_successful = True

        return self.iota.copy()

    def compute_q(self):
        """! Compute the q profile"""
        return 1/self.compute_iota()

    def plot(
        self, plottype=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs
    ):
        import matplotlib.pyplot as plt

        """! Generates the Poincare plot
        @param plottype which variables to plot: 'RZ' or 'yx', by default using "poincare_plot_type" in problem
        @param xlabel,ylabel what to put for the xlabel and ylabel, by default using "poincare_plot_xlabel" in problem
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """

        if not self.successful:
            raise Exception("A successful call of compute() is needed")

        # default setting
        if plottype is None:
            plottype = self._problem.poincare_plot_type
        if xlabel is None:
            xlabel = self._problem.poincare_plot_xlabel
        if ylabel is None:
            ylabel = self._problem.poincare_plot_ylabel

        if plottype == "RZ":
            xdata = self.x
            ydata = self.z
        elif plottype == "yx":
            xdata = self.y
            ydata = self.x
        elif plottype == "st":
            xdata = np.mod(self.theta, 2 * np.pi)
            ydata = self.s
        else:
            raise ValueError("Choose the correct type for plottype")

        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        elif "fig" in kwargs.keys():
            fig = kwargs["fig"]
            ax = fig.gca()
        elif "ax" in kwargs.keys():
            ax = kwargs["ax"]
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        # set default plotting parameters
        # use dots
        if kwargs.get("marker") is None:
            kwargs.update({"marker": "."})
        # use gray color
        # if kwargs.get('c') is None:
        #     kwargs.update({'c': 'gray'})
        # make plot depending on the 'range'

        for ii in range(self._params["nPtrj"] + 1):
            dots = ax.scatter(xdata[ii, :], ydata[ii, :], **kwargs)

        # adjust figure properties
        if plottype == "RZ":
            ax.set_aspect("equal")
        if plottype == "yx":
            pass

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # plt.tight_layout()
        return fig, ax

    def plot_iota(self, xlim=None, ylim=None, **kwargs):
        """! Generates the iota plot
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """
        if not self.iota_successful:
            raise Exception("A successful call of compute_iota() is needed")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.siota, self.iota, **kwargs)

        if self._is_cylindrical_problem:
            plt.xlabel("R", fontsize=20)
        else:
            plt.xlabel("s", fontsize=20)
        plt.ylabel(r"$\iota\!\!$-", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # plt.tight_layout()

    def plot_q(self, xlim=None, ylim=None, **kwargs):
        """! Generates the q plot
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """
        if not self.iota_successful:
            raise Exception("A successful call of compute_q or compute_iota is needed")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.siota, 1/self.iota, **kwargs)

        if self._is_cylindrical_problem:
            plt.xlabel("R", fontsize=20)
        else:
            plt.xlabel("s", fontsize=20)
        plt.ylabel("q", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

    @staticmethod
    def _run_poincare(params):
        """A function called in parallel to generate the Poincare plot for one starting point
        Called in PoincarePlot.compute, do not call otherwise
        """

        # copy the input to local
        integrator = params["integrator"].copy()
        nPpts = params["nPpts"]
        t0 = params["t0"]
        ic = params["ic"]
        dt = params["dt"]

        s = np.zeros([nPpts + 1], dtype=np.float64)
        theta = np.zeros_like(s)
        zeta = np.zeros_like(s)

        integrator.set_initial_value(t0, ic)
        t = t0
        for jj in range(1, nPpts + 1):

            # run the integrator
            st = integrator.integrate(t + dt)

            # extract the result to s theta zeta
            s[jj] = st[0]
            theta[jj] = st[1]
            zeta[jj] = t + dt

            # put st as the new ic
            ic = st

            # advance in time
            t = t + dt

        output = dict()
        output["s"] = s
        output["theta"] = theta
        output["zeta"] = zeta
        output["id"] = params["id"]

        return output
