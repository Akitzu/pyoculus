## @file base_solver.py
#  @brief Contains base class for solvers
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from pyoculus.problems import BaseMap

## Abstract class that used to setup all other solvers.
class BaseSolver:

    ## Used to return the output data
    class OutputData:
        def __init__(self):
            pass

    def __init__(self, map: BaseMap):
        """! Sets up the solver
        @param problem must inherit pyoculus.problems.BaseProblem, the problem to solve
        @param params dict, the parameters for the solver
        @param integrator the integrator to use, must inherit pyoculus.integrators.BaseIntegrator, if set to None by default using RKIntegrator
        @param integrator_params dict, the parmaters passed to the integrator
        """
        ## flagging if the computation is done and successful
        self.successful = False

        # check the map
        if not isinstance(map, BaseMap):
            raise ValueError("The problem is not a derived type of BaseMap class.")

        self._map = map


    def is_successful(self):
        """! Returns True if the computation is successfully completed
        @returns successful -- True if the computation is successfully completed, False otherwise
        """
        return self.successful
