## @file cartesian_bfield.py
#  @brief containing a class for pyoculus ODE solver that deals with magnetic field given in Cartesian
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .cylindrical_bfield import CylindricalBfield
import numpy as np

class CartesianBfield(CylindricalBfield):
    @staticmethod
    def _inv_Jacobian(R, phi, _):
        return np.array([
            [np.cos(phi), np.sin(phi), 0], 
            [-np.sin(phi)/R, np.cos(phi)/R, 0], 
            [0,0,1]
            ])
    
    @staticmethod
    def _Jacobian(R, phi, Z):
        return np.linalg.inv(CartesianBfield._inv_Jacobian(R, phi, Z))

    @staticmethod
    def _xyz(R, phi, Z):
        return np.array([
            R * np.cos(phi),
            R * np.sin(phi),
            Z
        ])
    
    @staticmethod
    def _vec2cyl(vec, R, phi, Z):
        return np.matmul(CartesianBfield.invJacobian(R, phi, Z), np.atleast_2d(vec).T).T[0]
        
    @staticmethod
    def _mat2cyl(mat, R, phi, Z):
        invjac = CartesianBfield.invJacobian(R, phi, Z)
        jac = np.linalg.inv(invjac)
        return np.matmul(np.matmul(invjac, mat), jac)