## @file base_solver.py
#  @brief Contains base class for solvers
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#
from abc import ABC
from pyoculus.problems import BaseMap

class BaseSolver(ABC):
    """
    Abstract base class for solvers.
    """

    class OutputData:
        """
        Class to hold the output data of the solver.
        """
        def __init__(self):
            pass

    def __init__(self, map: BaseMap):
        """
        Initializes the BaseSolver object.
        
        Args:
            map (BaseMap): The map to use for the solver.
        """
        ## Flag to note if the computation was performed successfuly
        self._successful = False

        # Check if the map is derived from BaseMap
        if not isinstance(map, BaseMap):
            raise ValueError("The problem is not a derived type of BaseMap class.")

        self._map = map


    def is_successful(self):
        """
        Returns a boolean indicating if the solver was successful.
        """
        return self._successful
