## @file poincare_plot.py: class for generating the Poincare Plot
#  @brief class for generating the Poincare Plot
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

import numpy as np
import matplotlib.pyplot as plt
from .base_solver import BaseSolver
import logging

logger = logging.getLogger(__name__)

class PoincarePlot(BaseSolver):
    """
        Class that used to setup the Poincare plot.
    """

    def compute(self, xs, nPpts=400, nthreads=1):
        """
        Computes the Poincare plot
        
        Returns:
            class that contains the results
        """

        if xs.shape[1] != self._map.dimension:
            raise ValueError("The initial points should have the correct dimension.")
        for ii in range(self._map.dimension):
            if xs[:, ii].any() < self._map.domain[ii][0] or xs[:, ii].any() > self._map.domain[ii][1]:
                raise ValueError("The initial points should be in the domain of the map.")

        # Initialize the hits
        hits = np.nan * np.ones((xs.shape[0], nPpts + 1, self._map.dimension), dtype=np.float64)
        hits[:, 0, :] = xs

        if nthreads == 1:  # single thread, do it straight away
            for i, x in enumerate(xs):
                current_x = x.copy()
                for j in range(nPpts):
                    try:
                        current_x = self._map.f(1, current_x)
                    except:
                        logger.warning("The map failed to compute at point %s", current_x)
                        break
                    hits[i, j + 1, :] = current_x

        self._successful = True
        self._hits = hits

        return hits

    def compute_iota(self):
        """! Compute the iota profile"""

        if not self._successful:
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
        self, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs
    ):
        
        if not self._successful:
            raise Exception("A successful call of compute() is needed")

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

        for x_mapped in self._hits:
            ax.scatter(x_mapped[:, 0], x_mapped[:, 1], **kwargs)

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
