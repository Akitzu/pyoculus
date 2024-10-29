from .cylindrical_bfield import CylindricalBfield
from scipy.interpolate import RegularGridInterpolator
import numpy as np


class AxisymmetricCylindricalGridField(CylindricalBfield):
    """ 
    Axisymmetric magnetic field provided by interpolating a grid of points given in the R-Z plane. 

    Tokamak equilibrium solvers often provide the relevant data in a grid of points in the R-Z plane. 
    """

    def __init__(self, R, Z, B_R, B_Z, B_phi, F_psi, pertfield: CylindricalBfield = None):
        """
        R: numpy array specifying the R coordinates of the grid points
        Z: numpy array specifying the Z coordinates of the grid points
        B_R: numpy array specifying the R component of the magnetic field at each grid point
        B_Z: numpy array specifying the Z component of the magnetic field at each grid point
        B_phi: numpy array specifying the phi component of the magnetic field at each grid point
        """
        super().__init__(1)
        self.R = R
        self.Z = Z
        self.B_phi = B_phi
        self.F_psi = F_psi

        self.F_psi_interpolator = RegularGridInterpolator((R, Z), F_psi, method='quintic')
        self.B_R_derived = lambda xx: -1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[0,1])
        self.B_Z_derived = lambda xx: 1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[1,0])
        self.B_phi_interpolator = RegularGridInterpolator((R, Z), B_phi, method='quintic')
        self.B_R_interpolator = RegularGridInterpolator((R, Z), B_R, method='quintic')
        self.B_Z_interpolator = RegularGridInterpolator((R, Z), B_Z, method='quintic')
        self.pertfield=pertfield
        self.pertamp = 1
        if pertfield is None:
            self.pertftun = lambda xx: np.zeros(3)
        else:
            self.pertfun = pertfield.B

    @classmethod
    def from_matlab_file(cls, filename, with_perturbation=False):
        """
        filename: string specifying the name of the .mat file containing the grid of points. 

        for example, MEQ equilibrium solver calculates the magnetic data in Matlab, the user needs to save the data in a .mat file and provide the filename to this function.
        """
        import scipy.io
        data = scipy.io.loadmat(filename)
        R = data['rr'][0,:]
        Z = data['zz'][:,0]
        B_R = data['Br'].T
        B_Z = data['Bz'].T
        B_phi = data['Bphi'].T
        F_psi = data['Fx'].T
        if with_perturbation:
            F_psi_cosphi = data['Fx_cosphi'].T
            F_psi_sinphi = data['Fx_sinphi'].T
            pertfield = AxisymmetricGridPerturbation(R, Z, F_psi_cosphi, F_psi_sinphi)
            return cls(R, Z, B_R, B_Z, B_phi, F_psi, pertfield=pertfield)
        else:
            return cls(R, Z, B_R, B_Z, B_phi, F_psi)
    
    def set_perturbation_amplitude(self, amplitude):
        self.pertamp = amplitude
        self.pertfun = lambda xx: amplitude * self.pertfield.B(xx)

    def B_axi(self, xx):
        """
        xx: numpy array of shape (2,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2] # xx is a rphiz vector, only pick R and Z.
        return np.hstack([self.B_R_derived(xx2d), self.B_phi_interpolator(xx2d)[0], self.B_Z_derived(xx2d)])
    
    def B(self, xx):
        """
        xx: numpy array of shape (2,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        return self.B_axi(xx) + self.pertfun(xx)

    def B_interpolated(self, xx):
        """
        Evaluate magnetic field directly, for comparison with the derivation of the flux funcitons. 
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2]
        return np.array([self.B_R_interpolator(xx2d)[0], self.B_phi_interpolator(xx2d)[0], self.B_Z_interpolator(xx2d)[0]])
    
    def A(self, xx): 
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the vector potential is to be evaluated
        """
        return np.array([0, 0, self.B_phi_interpolator(xx[0:2])[0] * xx[0]])
#    
#    def B_many(self, xx):
#        """
#        xx: numpy array of shape (N, 3) specifying the coordinates at which the magnetic field is to be evaluated
#        """
#        xx2d = xx[:, ::2]
#        return np.array([self.B_R_interpolator(xx2d), self.B_phi_interpolator(xx2d), self.B_Z_interpolator(xx2d)]).T
    
    def dBdX(self, xx):
        """
        xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field gradient is to be evaluated
        """
        xx2d = xx[0:2].reshape((1, 2))
        
        r_derivs = np.array(
            [1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[2,0]), 
            self.B_phi_interpolator(xx2d, nu=[1,0]), 
            1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[1,1])]
            )

        z_derivs = np.array(
            [1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[2,0]), 
            self.B_phi_interpolator(xx2d, nu=[1,0]), 
            1/(2*np.pi*xx[0]) * self.F_psi_interpolator(xx, nu=[1,1])]
            )
        return self.B(xx2d), np.array([self.B_R_interpolator(xx2d, nu=1)[0], self.B_phi_interpolator(xx2d, nu=1)[2], self.B_Z_interpolator(xx2d, nu=1)[1]])
    

class AxisymmetricGridPerturbation(CylindricalBfield):
    """
    Axisymmetric magnetic field perturbation provided by interpolating a grid of points given in the R-Z plane.
    """
    def __init__(self, R, Z, F_psi_cosphi, F_psi_sinphi):
        """
        Create the perturbation field from the perturbation
        flux function grids.
        """
        super().__init__(1)
        self.R = R
        self.Z = Z
        self.F_psi_cosphi = F_psi_cosphi
        self.F_psi_sinphi = F_psi_sinphi

        self.F_psi_cosphi_interpolator = RegularGridInterpolator((R, Z), F_psi_cosphi, method='quintic')
        self.F_psi_sinphi_interpolator = RegularGridInterpolator((R, Z), F_psi_sinphi, method='quintic')

    def B_R(self, xx):
        """
        xx: numpy array of shape (2,) specifying the coordinates at which the magnetic field is to be evaluated
        """
        xx2d = xx[::2]
        B_R_cosphi = -1/(2*np.pi*xx[0]) * self.F_psi_cosphi_interpolator(xx2d, nu=[0,1]) * np.cos(xx[1])
        B_R_sinphi = -1/(2*np.pi*xx[0]) * self.F_psi_cosphi_interpolator(xx2d, nu=[0,1]) * np.sin(xx[1])
        return B_R_cosphi + B_R_sinphi
    
    def B_Z(self, xx):
        xx2d = xx[::2]
        B_Z_cosphi = 1/(2*np.pi*xx[0]) * self.F_psi_sinphi_interpolator(xx2d, nu=[1,0]) * np.cos(xx[1])
        B_Z_sinphi = 1/(2*np.pi*xx[0]) * self.F_psi_sinphi_interpolator(xx2d, nu=[1,0]) * np.sin(xx[1])
        return B_Z_cosphi + B_Z_sinphi
    
    def B(self, xx): 
        return np.hstack([self.B_R(xx), 0, self.B_Z(xx)])
    
    def A(self):
        raise NotImplementedError("Vector potential is not implemented.")
    
    def dBdX(self, coords, *args):
        """WRONG placeholder for during development"""
        return self.B(coords)


#class NonAxisymmetricCylindricalGrid(CylindricalBfield):
#    """ 
#    Non-axisymmetric magnetic field provided by interpolating a grid of points given in the R-Z plane. 
#
#    Tokamak equilibrium solvers often provide the relevant data in a grid of points in the R-Z plane. 
#    """
#
#    def __init__(self, R, Z, phi, B_R, B_Z, B_phi):
#        """
#        R: 1D numpy array specifying the R coordinates of the grid points
#        Z: 1D numpy array specifying the Z coordinates of the grid points
#        phi: 1D numpy array specifying the phi coordinates of the grid points
#        B_R: numpy array specifying the R component of the magnetic field at each grid point
#        B_Z: numpy array specifying the Z component of the magnetic field at each grid point
#        B_phi: numpy array specifying the phi component of the magnetic field at each grid point
#        """
#        self.R = R
#        self.Z = Z
#        self.phi = phi
#        self.B_R = B_R
#        self.B_Z = B_Z
#        self.B_phi = B_phi
#
#        self.B_R_interpolator = RegularGridInterpolator((R, Z, phi), B_R, method='cubic')
#        self.B_Z_interpolator = RegularGridInterpolator((R, Z, phi), B_Z, method='cubic')
#        self.B_phi_interpolator = RegularGridInterpolator((R, Z, phi), B_phi, method='cubic')
#
#        def B(self, xx):
#            """
#            xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field is to be evaluated
#            """
#            return np.array([self.B_R_interpolator(xx)[0], self.B_Z_interpolator(xx)[0], self.B_phi_interpolator(xx)[0]])
#                        
#
#        def B_many(self, xx):
#            """
#            xx: numpy array of shape (N, 3) specifying the coordinates at which the magnetic field is to be evaluated
#            """
#            return np.array([self.B_R_interpolator(xx), self.B_Z_interpolator(xx), self.B_phi_interpolator(xx)]).T  
#
#        def dBdX(self, xx):
#            """
#            xx: numpy array of shape (3,) specifying the coordinates at which the magnetic field gradient is to be evaluated
#            """
#            return np.array([self.B_R_interpolator(xx, nu=1)[0], self.B_Z_interpolator(xx, nu=1)[1], self.B_phi_interpolator(xx, nu=1)[2]])
#                            
#