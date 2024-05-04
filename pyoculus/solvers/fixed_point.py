## @file fixed_point.py
#  @brief class for finding fixed points
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .base_solver import BaseSolver
from ..problems import BaseMap
import numpy as np

import structlog
log = structlog.get_logger()

class FixedPoint(BaseSolver):
    """
    Fixed point class to find points that satisfy f^t(x) = x.
    """

    def find(self, t, guess, niter=100, nrestart=0, tol=1e-10):
        """
        Finds a fixed point of a map applied 't' times.    
        """
        if not self._map.is_continuous and not isinstance(t, int):
            raise ValueError("The iteration number should be an integer for a discrete map.")

        self.t = t
        self.history = []
        x_fp = None

        # set up the guess
        if len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")

        # run the Newton's method
        for ii in range(nrestart+1):
            try:  # run the solver, if failed, try a different random initial condition
                x_fp, jac = self._newton_method(guess, niter, tol)
                if self.successful:
                    break
            except Exception as e:
                log.info(f"Search {ii} - failed: {e}")
            
            if ii < nrestart:
                log.info(f"Search {ii+1} starting from a random initial guesss!")
                guess = self.random_initial_guess()

        if x_fp is not None:
            self.coords = x_fp
            self.jacobians = jac

            rdata = FixedPoint.OutputData()
            rdata.coords = self.coords.copy()
            rdata.jacobians = self.jacobians.copy()

        else:
            rdata = None
            log.info(f"Fixed point search unsuccessful for t={self.t}.")

        return rdata

    def find_with_iota(self, pp, qq, guess, x_axis = None, niter=100, nrestart=0, tol=1e-10):
        """
        Computes the fixed point of a continuous map with winding number iota = q^-1 = pp/qq around x_axis.

        Args:
            guess (array): the initial guess of the fixed point
            x_axis (array): the point around which the winding number is calculated 
            niter (int): the maximum number of iterations
            nrestarts (int): the maximum number of restart with different initial conditions
            tol (float): the tolerance of the fixed point

        Returns:
            coords (array): the fixed point in the coordinates of the problem
            jac (array): the Jacobian the fixed point
            GreenesResidues (array): the Greene's Residue of the fixed point
            MeanResidues (array): the 'Average Residue' f as defined by Greene
        """

        if not self._map.is_continuous:
            raise ValueError("The map needs to be continuous to find a fixed point with a winding number.")
        
        if not isinstance(pp, int) or not isinstance(qq, int):
            raise ValueError("pp and qq should be integers")

        if pp * qq >= 0:
            pp = int(np.abs(pp))
            qq = int(np.abs(qq))
        else:
            pp = -int(np.abs(pp))
            qq = int(np.abs(qq))

        gcd = np.gcd(pp, qq)
        self.t = qq // gcd

        self.pp = pp
        self.qq = qq

        # arrays that save the data
        self.coords = np.zeros(shape=(self.qq + 1, self._map.dimension), dtype=np.float64)
        self.jacobians = np.zeros(shape=(self.qq + 1, self._map.dimension, self._map.dimension), dtype=np.float64)
        self.GreenesResidues = np.zeros(self.qq + 1, dtype=np.float64)    
        self.MeanResidues = np.zeros(self.qq + 1, dtype=np.float64)    
        
        self.history = []
        x_fp = None

        # set up the guess
        if len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")

        # run the Newton's method
        for ii in range(nrestart+1):
            try:  # run the solver, if failed, try a different random initial condition
                x_fp, jac = self._newton_method_winding(guess, x_axis, niter, tol)
                if self.successful:
                    break
            except Exception as e:
                log.info(f"Search {ii} - failed: {e}")
            
            if ii < nrestart:
                log.info(f"Search {ii+1} starting from a random initial guesss!")
                guess = self.random_initial_guess()

        # now we go and get all the fixed points by iterating the map                
        if x_fp is not None:
            self.coords[0] = x_fp
            self.jacobians[0] = jac

            steps = 1 if self._map.is_continuous else self.pp
            for jj in range(0, self.qq + 1, steps):
                if self._map.is_continuous:
                    self.coords[jj] = self._map.f(jj/self.pp * self.t, x_fp)
                else:
                    self.coords[jj] = self._map.f(jj*self.t, x_fp)
                self.jacobians[jj] = self._map.df(jj*self.t, self.coords[jj])
                
            for jj in range(0, self.qq + 1, steps):
                self.GreenesResidues[jj] = 0.25 * (2.0 - np.trace(self.jacobians[jj]))
                self.MeanResidues[jj] = np.power(
                    np.abs(self.GreenesResidues[jj]) / 0.25, 1 / float(self.qq)
                )

            rdata = FixedPoint.OutputData()
            rdata.coords = self.coords.copy()
            rdata.jacobians = self.jacobians.copy() 
        
            # Greene's Residue
            rdata.GreenesResidues = self.GreenesResidues.copy()
            rdata.MeanResidues = self.MeanResidues.copy()

        else:
            rdata = None
            log.info(f"Fixed point search unsuccessful for pp/qq={self.pp}/{self.qq}.")

        return rdata

    def random_initial_guess(self):
        """
        Returns a random initial guess for the fixed point inside the map domain.
        """
        domain = self._map.domain
        domain = [(low if low is not -np.inf else -1e100, high if high is not np.inf else 1e100) for (low, high) in domain]
        return np.array([np.random.uniform(low, high) for (low, high) in domain])

    def _newton_method_winding(
            self, guess, niter, tol
        ):

        tmp = np.array(guess)
        self.history.append(tmp.copy())

        for ii in range(niter):
            

            dtheta = output[1] - theta - dzeta * pp
            jacobian = output[3]

            # if the resolution is good enough
            if abs(dtheta) < tol:
                succeeded = True
                break
            s_new = s - dtheta / jacobian
            s = s_new

            if s > send or s < sbegin:  # search failed, return None
                return None

            ic = np.array([s, theta, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self.history.append(ic[0:1].copy())

        if succeeded:
            return np.array([s, theta, zeta], dtype=np.float64)
        else:
            return None

    def plot(
        self, plottype=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs
    ):
        """! Generates the plot for fixed points
        @param plottype which variables to plot: 'RZ' or 'yx', by default using "poincare_plot_type" in problem
        @param xlabel,ylabel what to put for the xlabel and ylabel, by default using "poincare_plot_xlabel" in problem
        @param xlim, ylim the range of plotting, by default plotting the range of all data
        @param **kwargs passed to the plotting routine "plot"
        """
        import matplotlib.pyplot as plt

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
            newfig = False
        else:
            fig, ax = plt.subplots()
            newfig = True

        # set default plotting parameters
        # use x
        if kwargs.get("marker") is None:
            kwargs.update({"marker": "x"})
        # use gray color
        if kwargs.get("c") is None:
            kwargs.update({"c": "black"})

        xs = ax.plot(xdata, ydata, linestyle="None", **kwargs)

        if not newfig:
            if plottype == "RZ":
                plt.axis("equal")
            if plottype == "yx":
                pass

            plt.xlabel(xlabel, fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

    def _newton_method_2(
        self, pp, qq, s_guess, sbegin, send, theta_guess, zeta, dzeta, niter, tol
    ):
        """driver to run Newton's method for two variable (s,theta)
        pp,qq -- integers, the numerator and denominator of the rotation number
        s_guess -- the guess of s
        sbegin -- the allowed minimum s
        send -- the allowed maximum s
        theta_guess -- the guess of theta
        zeta -- the toroidal plain to investigate
        dzeta -- period in zeta
        niter -- the maximum number of iterations
        tol -- the tolerance of finding a fixed point
        """

        self.successful = False

        s = s_guess
        theta = theta_guess

        # set up the initial condition
        ic = np.array([s, theta, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.history.append(ic[0:1].copy())

        t0 = zeta
        dt = dzeta

        succeeded = False

        st = np.array([s, theta], dtype=np.float64)

        for ii in range(niter):
            t = t0
            self._integrator.set_initial_value(t0, ic)
            for jj in range(qq):
                output = self._integrator.integrate(t + dt)
                t = t + dt

            dtheta = output[1] - theta - dzeta * pp
            ds = output[0] - s
            dst = np.array([ds, dtheta], dtype=np.float64)
            jacobian = np.array(
                [[output[2], output[4]], [output[3], output[5]]], dtype=np.float64
            )

            # if the resolution is good enough
            if np.sqrt(dtheta ** 2 + ds ** 2) < tol:
                succeeded = True
                break

            # Newton's step
            st_new = st - np.matmul(np.linalg.inv(jacobian - np.eye(2)), dst)
            s = st_new[0]
            theta = st_new[1]
            st = st_new

            if s > send or s < sbegin:  # search failed, return None
                return None

            ic = np.array([s, theta, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self.history.append(ic[0:1].copy())

        if succeeded:
            self.successful = True
            return np.array([s, theta, zeta], dtype=np.float64)
        else:
            return None    

    def _newton_method_3(
        self, pp, qq, R_guess, Rbegin, Rend, Z, zeta, dzeta, niter, tol
    ):
        """driver to run Newton's method for one variable R, for cylindrical problem
        pp,qq -- integers, the numerator and denominator of the rotation number
        R_guess -- the guess of R
        Rbegin -- the allowed minimum R
        Rend -- the allowed maximum R
        Z -- the Z value (fixed)
        zeta -- the toroidal plain to investigate
        dzeta -- period in zeta
        niter -- the maximum number of iterations
        tol -- the tolerance of finding a fixed point
        """

        R = R_guess
        R0 = self._problem._R0
        Z0 = self._problem._Z0
        theta = np.arctan2(Z-Z0, R-R0)

        # set up the initial condition
        ic = np.array([R, Z, R0, Z0, theta, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.history.append(ic[0:1].copy())

        t0 = zeta
        dt = dzeta

        succeeded = False

        for ii in range(niter):
            t = t0
            self._integrator.set_initial_value(t0, ic)

            for jj in range(qq):
                output = self._integrator.integrate(t + dt)
                t = t + dt

            dtheta = output[4] - theta - dzeta * pp
            print(f"[R,Z] : {[output[0], output[1]]} - dtheta : {dtheta}")

            dR = output[5]
            dZ = output[6]
            
            deltaR = output[0] - R0
            deltaZ = output[1] - Z0

            jacobian = (deltaR * dZ - deltaZ * dR) / (deltaR**2 + deltaZ**2)

            # if the resolution is good enough
            if abs(dtheta) < tol:
                succeeded = True
                break
            R_new = R - dtheta / jacobian
            R = R_new
            print(f"R : {R}")
            theta = np.arctan2(Z-Z0, R-R0)

            if R > Rend or R < Rbegin:  # search failed, return None
                return None

            ic = np.array([R, Z, R0, Z0, theta, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self.history.append(ic[0:1].copy())

        if succeeded:
            return np.array([R, Z, zeta], dtype=np.float64)
        else:
            return None
    
    def _newton_method_RZ(
        self, pp, qq, x_guess, niter, tol, checkonly
    ):
        # Set up the initial guess
        RZ = np.array([R_guess, Z_guess], dtype=np.float64)

        # Set up the initial condition  
        RZ_Axis = np.array([self._problem._R0, self._problem._Z0], dtype=np.float64)
        rhotheta = np.array([np.linalg.norm(RZ-RZ_Axis), np.arctan2(RZ[1]-RZ_Axis[1], RZ[0]-RZ_Axis[0])], dtype=np.float64)
        
        ic = np.array([RZ[0], RZ[1], RZ_Axis[0], RZ_Axis[1], rhotheta[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        self.history_Revolved = []
        self.history.append(ic[0:2].copy())
        
        t0 = zeta
        dt = dzeta

        succeeded = False

        for ii in range(niter):

            t = t0
            self._integrator.set_initial_value(t0, ic)

            for jj in range(qq):
                output = self._integrator.integrate(t + dt)
                t = t + dt

            RZ_evolved = np.array([output[0],output[1]])
            rhotheta_evolved = np.array([np.linalg.norm(RZ_evolved-RZ_Axis), 
                                         output[4] - dzeta * pp], dtype=np.float64)
            
            # Stop if the resolution is good enough
            condA = checkonly and np.linalg.norm(RZ_evolved-RZ) < tol
            condB = (not checkonly) and abs(rhotheta_evolved[1]-rhotheta[1]) < tol
            print(f"{ii} - [DeltaR, DeltaZ] : {RZ_evolved-RZ} - dtheta : {abs(rhotheta_evolved[1]-rhotheta[1])}")
            if condA or condB:
                succeeded = True
                break
            
            # dG switch to the convention of 
            # df = [[dG^R/dR, dG^r/dZ]
            #       [dG^Z/dR, df^Z/dZ]]
            dG = np.array([
                [output[5], output[7]],
                [output[6], output[8]]
            ], dtype=np.float64)

            if not checkonly:
                # dH = dH(G(R,Z))
                deltaRZ = RZ_evolved - RZ_Axis
                dH = np.array([
                    np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
                    np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
                ], dtype=np.float64)

                # dP = dH(R,Z)
                deltaRZ = RZ - RZ_Axis
                dP = np.array([
                    np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
                    np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
                ], dtype=np.float64)
                
                # Jacobian of the map F = H(G(R,Z)) - H(R,Z) 
                jacobian = dH @ dG - dP
                
                # Map F = H(G(R,Z)) - H(R,Z)
                F_evolved = rhotheta_evolved-rhotheta

            else:
                # Jacobian of the map F = G(R,Z) - (R,Z)
                jacobian = dG - np.eye(2)

                # Map F = G(R,Z) - (R,Z)
                F_evolved = RZ_evolved-RZ

            # Newton's step
            step = np.linalg.solve(jacobian, -1*F_evolved)
            RZ_new = RZ + step
            
            # Update the variables
            print(f"{ii} - [StepR, StepZ]: {RZ_new-RZ}")
            RZ = RZ_new
            rhotheta = np.array([np.linalg.norm(RZ-RZ_Axis), 
                                 np.arctan2(RZ[1]-RZ_Axis[1], RZ[0]-RZ_Axis[0])], dtype=np.float64)

            print(f"{ii+1} - RZ : {RZ} - rhotheta : {rhotheta}")
            # Check if the search is out of the provided R domain
            if RZ[0] > Rend or RZ[0] < Rbegin:
                return None
            
            ic = np.array([RZ[0], RZ[1], RZ_Axis[0], RZ_Axis[1], rhotheta[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

            self.history.append(ic[0:2].copy())
            self.history_Revolved.append(RZ_evolved)

        if succeeded:
            #assert abs(rhotheta_evolved[1]-rhotheta[1]) < 1e-3, "Found fixed-point as not the right poloidal number (pp)"
            return np.array([RZ[0], RZ[1], zeta], dtype=np.float64)
        else:
            return None
    
    def find_axis(
        self, R_guess, Z_guess, niter=100
    ):
        
        # Set up the initial guess
        RZ = np.array([R_guess, Z_guess], dtype=np.float64)
        
        # Set up the
        self.history = []
        self.history.append(ic[0:2].copy())
        
        succeeded = False

        for ii in range(self.niter):
                
            # Integrate to the next periodic plane
            dG = self._problem.df(0.0, ic)
            RZ_evolved = self._problem.f(0.0, ic)
            

            # Stop if the resolution is good enough
            print(f"{ii} - dr : {np.linalg.norm(RZ_evolved-RZ)}")
            if np.linalg.norm(RZ_evolved-RZ) < tol:
                succeeded = True
                break
            
            # dG switch to the convention of 
            # df = [[dG^R/dR, dG^r/dZ]
            #       [dG^Z/dR, df^Z/dZ]]
            dG = np.array([
                [output[2], output[4]],
                [output[3], output[5]]
            ], dtype=np.float64)

            # Jacobian of the map F = G(R,Z) - (R,Z)
            jacobian = dG - np.eye(2)

            # Map F = G(R,Z) - (R,Z)
            F_evolved = RZ_evolved-RZ

            # Newton's step
            step = np.linalg.solve(jacobian, -1*F_evolved)
            RZ_new = RZ + step
            
            # Update the variables
            RZ = RZ_new
           
            print(f"{ii+1} - RZ : {RZ}")
            # Check if the search is out of the provided R domain
            if RZ[0] > Rend or RZ[0] < Rbegin:
                return None
            
            ic = np.array([RZ[0], RZ[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

            self.history.append(ic[0:2].copy())

        if succeeded:
            return np.array([RZ[0], RZ[1], t0], dtype=np.float64)
        else:
            return None