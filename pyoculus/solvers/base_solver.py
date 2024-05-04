## @file base_solver.py
#  @brief Contains base class for solvers
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from pyoculus.problems import BaseMap

class BaseSolver:
    """
    Abstract class that used to setup all other solvers.
    """

    class OutputData:
        """
        Class that stores the output data of the solver.
        """
        def __init__(self):
            pass

    def __init__(self, map: BaseMap):
        """
        Sets up the solver.
        
        Args:
            map (BaseMap): The map to use for the solver.
        """
        ## flagging if the computation is done and successful
        self.successful = False

        # check the map
        if not isinstance(map, BaseMap):
            raise ValueError("The problem is not a derived type of BaseMap class.")

        self._map = map


    def is_successful(self):
        """
        Returns a boolean indicating if the solver was successful.
        """
        return self.successful
