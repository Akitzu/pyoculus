## @file spec_pjh.py
#  @brief Setup the SPEC Pressure Jump Hamiltonian system for ODE solver
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#
from .spec_problem import SPECProblem
import numpy as np


## Class that used to setup the SPEC Pressure Jump Hamiltonian system for interfacing Fortran.
# For more detail, see the SPEC fortran module \ref specpjh
#
# Please note that in the Fortran module, \f$\zeta\f$ is equivalent to \f$\varphi\f$.
# ## Pressure Jump Hamiltonian (PJH) Python module
# ### Usage
#
#     pjh_problem = pyoculus.problems.SPECPJH(spec_output, lvol, dp, innout, plusminus)
#
# @param spec_data the SPEC data generated by py_spec.SPECout
# @param lvol which volume we are interested in, from 1 to spec_data.input.Mvol
# @param dp the \f$\delta p\f$ for PJH
# @param innout 0 for inner surface, 1 for outer surface of the volume specified by lvol
# @param plusminus -- the sign for computing p_zeta, +1 or -1
#
# For example, a SPEC equilibirum has 4 volumes. We are interested in computing PJH for the interface between the 2nd and the 3rd volume.
#     - If we want to set the magnetic field on the inner side of this __interface__ known, we need to choose `lvol=2` and `innout=1` (outer side of 2nd __volume__ known)
#     - On the other hand, if the field on the outer side of this __interface__ is known, we choose `lvol=3` and `innout=0` (inner side of 3rd __volume__ known)
#     - The pressure jump \f$\delta p\f$ should be choosen based on \f$\delta p =  2 ( p_1 - p_2 ) = B_2^2 - B_1^2\f$, where \f$B_1^2\f$ is the known field
#     .
#
# ### Hint for choosing the initial p_theta and plusminus
#
# Knowing the covariant components of the magnetic field on the known side of the interface could help. To do so, run
#
#     Bco = get_covariant_field(theta, zeta)
#
# @returns a numpy array contains (\f$ B_\theta, B_\zeta \f$).
#
# ### Physics
# The pressure-jump Hamiltonian is derived from the force-balance condition at the ideal interfaces.
# Let \f$p_1\f$ and \f${\bf B}_1\f$ be the pressure and field immediately inside the interface, and \f$p_1\f$ and \f${\bf B}_1\f$ be the pressure and field immediately outside,
# then the force balance condition requires that
# \f[ H \equiv 2 \, \delta p =  2 ( p_1 - p_2 ) = B_2^2 - B_1^2 \f]
#
# For Beltrami fields, which satisfy \f$\nabla \times {\bf B}=\mu {\bf B}\f$, the magnitude of the field, \f$B\f$, on the interface (where we assume that \f$B^s=0\f$) may be written
# \f[
# B^2 = \frac{g_{\zeta\zeta} f_\theta f_\theta - 2 g_{\theta\zeta}f_\theta f_\zeta + g_{\theta\theta} f_\zeta f_\zeta}{g_{\theta\theta}g_{\zeta\zeta}-g_{\theta\zeta}g_{\theta\zeta}}
# \f]
# where \f$f\f$ is a surface potential and \f$g_{\theta\theta}\f$, \f$g_{\theta\zeta}\f$ and \f$g_{\zeta\zeta}\f$ are metric elements local to the interface.
# \item Assuming that the field \f$B_1\f$ is known on the `inside' of the interface, ie. \f$B_{1\theta}=f_\theta\f$, \f$B_{1\zeta}=f_\zeta\f$ and \f$f\f$ is known,
# it is required to determine the tangential field, \f$p_\theta = B_\theta\f$ and \f$p_\zeta = B_\zeta\f$, on the `outside' of the interface.
# \item The o.d.e.'s are given by Hamilton's equations
# \f[
# \dot \theta   =  \frac{\partial H}{\partial p_\theta}\Big|_{\theta,\zeta,p_\zeta}, \;\;
# \dot p_\theta = -\frac{\partial H}{\partial \theta}\Big|_{p_\theta,\zeta,p_\zeta}, \;\;
# \dot \zeta     =  \frac{\partial H}{\partial p_\zeta}\Big|_{\theta,p_\theta,\zeta}, \;\;
# \dot p_\zeta   = -\frac{\partial H}{\partial \zeta}\Big|_{\theta,p_\theta,p_\zeta},
# \f]
# where the `dot' denotes derivative with respect to `time'.
#
# This is reduced to a \f$1\frac{1}{2}\f$ dimensional system by using \f$\zeta\f$ as the time-like integration parameter, and replacing the equation for \f$\dot p_\zeta\f$ with
# \f[ p_\zeta= P(\theta,p_\theta,\zeta; \delta p) = \frac{-b\pm\sqrt{b^2-4ac}}{2a} \f]
# where \f$a=g_{\theta\theta}\f$, \f$b=-2 g_{\theta\zeta}p_\theta\f$
# and \f$c=g_{\zeta\zeta} p_{\theta}^2 - (B_1^2 + 2 \, \delta p \,) G\f$ (see below for definition of \f$G\f$).
# The o.d.e.'s that then need to be followed are (see below for definition of \f$A\f$ and \f$b_2\f$)
# \f[
# \frac{d   \theta}{d\zeta}= \frac{g_{\zeta\zeta} p_{\theta} - g_{\theta\zeta} p_{\zeta}}{-g_{\theta\zeta}p_{\theta}+g_{\theta\theta}p_{\zeta}}
# \f]
# \f[
# \frac{d p_\theta}{d\zeta}= \frac{g_{\zeta\zeta,\theta} (-p_{\theta}^2)-2g_{\theta\zeta,\theta}(-p_{\theta}p_{\zeta})
# +g_{\theta\theta,\theta}(-p_{\zeta}^2) + (B_1^2)_{\theta} G+ b_2 G_{\theta} / G }
# {-2g_{\theta\zeta}p_\theta+g_{\theta\theta}2p_{\zeta}}.
# \f]
#
# \f[ G = g_{\theta\theta} g_{\zeta\zeta} - g_{\theta\zeta} g_{\theta\zeta} \f]
# \f[ b_2 = g_{\zeta\zeta} p_{\theta}^2 - 2 g_{\theta\zeta} p_{\theta} p_{\zeta} + g_{\theta\theta} p_{\zeta}^2 \f]
#
# Note that \f$d\theta / d \zeta = B^\theta / B^\zeta \f$; there is a fundamental relation between the pressure-jump Hamiltonian and the field-line Hamiltonian.
# (Furthermore, in many cases the surface will be given in straight field line coordinates, so \f$d \theta / d\zeta = const.\f$.)
#
class SPECPJH(SPECProblem):
    def __init__(self, spec_data, lvol, dp=0.0, innout=0, plusminus=+1):
        """!Set up the equilibrium for use of the fortran module
        @param spec_data the SPEC data generated by py_spec.SPECout
        @param lvol which volume we are interested in, from 1 to `spec_data.input.Mvol`
        @param dp the \f$\delta p\f$ for PJH
        @param innout 0 for inner surface, 1 for outer surface of the volume specified by `lvol`
        @param plusminus -- the sign for computing \f$p_\zeta\f$, +1 or -1
        Only support SPEC version >=3.0
        """
        super().__init__(spec_data, lvol)

        self.fortran_module.specpjh.init_pjh(dp, innout, plusminus)
        self.dp = dp
        self.innout = innout
        self.plusminus = plusminus
        self.initialized = True

        ## the size of the problem, 2 for 1.5 or 2D system
        self.problem_size = 2
        ## choose the variable for Poincare plot
        self.poincare_plot_type = "yx"
        ## the x label of Poincare plot
        self.poincare_plot_xlabel = "theta"
        ## the y label of Poincare plot
        self.poincare_plot_ylabel = "p_theta"

    def set_PJH_parameters(self, dp=0.0, innout=0, plusminus=+1):
        """!Set up the parameters for the pjh fortran module
        @param dp the \f$\delta p\f$ for PJH
        @param innout 0 for inner surface, 1 for outer surface for the volume specified by `lvol`
        @param plusminus the sign for computing \f$p_\zeta\f$, +1 or -1
        """
        self.fortran_module.specpjh.init_pjh(dp, innout, plusminus)

    def f(self, zeta, st, arg1=None):
        """! Python wrapper for pjh ODE RHS
        @param zeta the \f$\zeta\f$ coordinate
        @param st array size 2, \f$(p_\theta, \theta)\f$
        @param arg1 -- parameter for the ODE, not used here
        @returns array size 2, the RHS of the ODE
        """
        return self.fortran_module.specpjh.get_pjhfield(zeta, st)

    def f_tangent(self, zeta, st, arg1=None):
        """! Python wrapper for pjh ODE RHS, with tangent
        @param zeta the \f$\zeta\f$ coordinate
        @param st array size 6, the \f$(p_\theta, \theta, \Delta p_{\theta,1}, \Delta \theta_1, \Delta p_{\theta,2}, \Delta \theta_2)\f$
        @param arg1 -- parameter for the ODE, not used here
        @returns array size 6, the RHS of the ODE
        """

        return self.fortran_module.specpjh.get_pjhfield_tangent(zeta, st)

    def convert_coords(self, stz):
        """! Python wrapper for getting the xyz coordinates from stz (identity for PJH)
        @param stz the stz coordinate
        @returns the xyz coordinates
        """
        return np.array(
            [stz[0], np.mod(stz[1], 2.0 * np.pi), np.mod(stz[2], 2.0 * np.pi)],
            dtype=np.float64,
        )

    def get_covariant_field(self, theta, zeta):
        """! Get the value of \f$B_\theta\f$ and \f$B_\zeta\f$ on the known side of the interface
        @param theta  the \f$\theta \f$ coordinate
        @param zeta   the \f$\zeta  \f$ coordinate
        @returns (\f$ B_\theta, B_\zeta \f$)
        """
        return self.fortran_module.specpjh.get_covariant_field(theta, zeta)
