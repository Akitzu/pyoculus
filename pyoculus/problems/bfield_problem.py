## @file bfield_problem.py
#  @brief containing a problem class with magnetic fields
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#


class BfieldProblem:
    def __init__(self):
        """
        Initializes the BfieldProblem class.
        """
        ## if the output magnetic field contains the jacobian factor or not
        self.has_jacobian = False

    def B(self, coords, *args):
        """
        Returns the contravariant magnetic fields at the given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the magnetic fields.
            *args: Additional parameters.
        """
        raise NotImplementedError("A problem class should implement member function B.")

    def dBdX(self, coords, *args):
        """
        Returns the contravariant magnetic fields and their derivatives at the given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the magnetic fields and their derivatives.
            *args: Additional parameters.

        Returns:
            array: The contravariant magnetic fields
            array: The derivatives of the contravariant magnetic fields
        """
        raise NotImplementedError(
            "A problem class should implement member function dBdX."
        )

    def A(self, coords, *args):
        """
        Returns the contravariant vector potential at the given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the vector potential.
            *args: Additional parameters.
        """
        raise NotImplementedError("Vector potential is not implemented.")

    def B_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant magnetic fields at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the magnetic fields.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        raise NotImplementedError("B_many is not implemented.")

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant magnetic fields and their derivatives at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the magnetic fields and their derivatives.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        raise NotImplementedError("dBdX_many is not implemented.")

    def A_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant vector potential at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the vector potential.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        raise NotImplementedError("A_many is not implemented.")
