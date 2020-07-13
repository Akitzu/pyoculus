########################################
# PoincarePlot.py: class for generating the Poincare Plot
# written by @zhisong (zhisong.qu@anu.edu.au)
#

import numpy as np
import concurrent.futures
from .BaseSolver import BaseSolver

class PoincarePlot(BaseSolver):
    """
    Class that used to setup the Poincare plot.

    Call signature:
        my_plot = PoincarePlot(problem, params, integrator, integrator_params) 

    Contains:
        compute -- construct the Poincare plot
        compute_iota -- compute iota profile according to Poincare plot
        plot -- plot the Poincare plot
        plot_iota -- plot the iota profile
    """

    # flagging if the computation is done and successful
    successful = False
    iota_successful = False

    def __init__(self, problem,  params=dict(), integrator=None, integrator_params=dict()):
        '''Set up the Poincare plot 
        parameters:

            problem -- BaseEquilibrium.BaseProblem class, the problem to solve
            integrator -- the integrator to use, by default using RKIntegrator
            params -- dict, the parameters used in the ODE solver
            integrator_params -- dict, the parmaters passed to the integrator

            params['theta']=0 -- the Poincare plots starts from theta
            params['zeta']=0 -- the Poincare plots for which toroidal section: zeta
            params['nPpts']=500 -- the number of iterations
            params['nPtrj']=10 -- the number of equidistant points in s coordinates
            params['sbegin']=-1.0 -- the lower bound of s
            params['send']=-1.0 -- the upper bound of s
            params['Nfp']=1 -- the toroidal periodicity, used to save computation time if there is a symmetry
            params['nthreads']=1 -- the number of threads
        '''

        if 'theta' not in params.keys():
            params['theta'] = 0.0
            
        if 'zeta' not in params.keys():
            params['zeta'] = 0.0

        if 'nPpts' not in params.keys():
            params['nPpts'] = 500

        if 'nPtrj' not in params.keys():
            params['nPtrj'] = 10

        if 'sbegin' not in params.keys():
            params['sbegin'] = -1.0

        if 'send' not in params.keys():
            params['send'] = 1.0

        if 'Nfp' not in params.keys():
            params['Nfp'] = 1
            print('Nfp not specified in params, using 1 as default, please confirm')

        if 'nthreads' not in params.keys():
            params['nthreads'] = 1

        integrator_params['ode'] = problem.f
        super().__init__(problem=problem,  params=params, integrator=integrator, integrator_params=integrator_params)

        # set up the result array
        self.x = np.zeros([self._params['nPtrj']+1,self._params['nPpts']+1],dtype=np.float64)
        self.y = np.zeros_like(self.x)
        self.z = np.zeros_like(self.x)
        self.s = np.zeros_like(self.x)
        self.theta = np.zeros_like(self.x)
        self.zeta = np.zeros_like(self.x)

    def compute(self):
        '''Compute the Poincare plot 
        '''

        self.successful = False

        # the zeta interval for each integration
        self.dt = 2*np.pi / float(self._params['Nfp'])

        for ii in range(self._params['nPtrj']+1):
            # find which s to start with
            swindow = self._params['send']-self._params['sbegin']
            ds = swindow / float(self._params['nPtrj'])
            self.s[ii,0] = self._params['sbegin'] + ds * float(ii)

            # put in the initial conditions
            self.theta[ii,0] = self._params['theta']
            self.zeta[ii,0] = self._params['zeta']

        if self._params['nthreads'] == 1: # single thread, do it straight away

            for ii in range(self._params['nPtrj']+1):
                # set up the initial value
                ic = [self.s[ii,0], self.theta[ii,0]]
                t0 = self.zeta[ii,0]

                # initialize the integrator
                self._integrator.set_initial_value(t0, ic)

                t = t0
                dt = self.dt

                for jj in range(1,self._params['nPpts']+1):

                    # run the integrator
                    try:
                        st = self._integrator.integrate(t + dt)
                    except Exception:
                        # integration failed, abort for this orbit
                        print('Integration failed for s=', ic[0])
                        break

                    # extract the result to s theta zeta
                    self.s[ii,jj] = st[0]
                    self.theta[ii,jj] = st[1]
                    self.zeta[ii,jj] = t + dt

                    # advance in time
                    t = t + dt

        else: # multi-threading
            
            print('Running the Poincare plot using ', self._params['nthreads'], ' threads.')
            raise Exception('multi-threading not working at the moment')
            # prepare a list of input each starting point
            inputs = []

            for ii in range(self._params['nPtrj']+1):
                pparams = dict()

                pparams['integrator'] = self._integrator
                pparams['t0'] = self.zeta[ii,0]
                pparams['ic'] = [self.s[ii,0], self.theta[ii,0]]
                pparams['dt'] = self.dt
                pparams['nPpts'] = self._params['nPpts']
                pparams['id'] = ii

                inputs.append(pparams)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._params['nthreads']) as executor:
                futures = {executor.submit(self._run_poincare, inputi): inputi for inputi in inputs}
                for future in concurrent.futures.as_completed(futures):
                    data = future.result()
                    ii = data['id']
                    self.s[ii,:] = data['s']
                    self.theta[ii,:] = data['theta']
                    self.zeta[ii,:] = data['zeta']

        # convert everything into xyz
        for ii in range(self._params['nPtrj']+1):
            for jj in range(self._params['nPpts']+1):
                stz = np.array([self.s[ii,jj],self.theta[ii,jj],self.zeta[ii,jj]],dtype=np.float64)
                xyz = self._problem.convert_coords(stz)
                self.x[ii,jj] = xyz[0]
                self.y[ii,jj] = xyz[1]
                self.z[ii,jj] = xyz[2]

        self.successful = True

        pdata = PoincarePlot.OutputData()
        pdata.x=self.x
        pdata.y=self.y
        pdata.z=self.z

        return pdata


    def compute_iota(self):
        '''Compute the iota profile
        '''

        if not self.successful:
            raise Exception('A successful call of compute() is needed')

        self.iota_successful = False

        # fit iota
        self.siota = self.s[:,0]
        self.iota = np.zeros_like(self.siota)
        for ii in range(self._params['nPtrj']+1):
            nlist = np.arange(self._params['nPpts']+1,dtype=np.float64)
            dzeta = 2.0*np.pi / self._params['Nfp']
            leastfit = np.zeros(6,dtype=np.float64)
            leastfit[1] = np.sum((nlist * dzeta)**2)
            leastfit[2] = np.sum((nlist * dzeta))
            leastfit[3] = np.sum((nlist * dzeta) * self.theta[ii,:])
            leastfit[4] = np.sum(self.theta[ii,:])
            leastfit[5] = 1.0
            
            self.iota[ii]= ( leastfit[5]*leastfit[3]-leastfit[2]*leastfit[4] ) / ( leastfit[5]*leastfit[1]-leastfit[2]*leastfit[2] )

        self.iota_successful = True

        return self.iota

    def plot(self, plottype=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
        import matplotlib.pyplot as plt
        '''generate the poincare plot
        parameters:
            plottype -- which variables to plot: 'RZ' or 'yx', by default using "poincare_plot_type" in problem
            xlabel, ylabel -- what to put for the xlabel and ylabel, by default using "poincare_plot_xlabel" in problem
            xlim, ylim -- the range of plotting, by default plotting the range of all data
            **kwargs -- passed to the plotting routine "scatter"
        '''

        if not self.successful:
            raise Exception('A successful call of compute() is needed')

        # default setting
        if plottype is None:
            plottype = self._problem.poincare_plot_type
        if xlabel is None:
            xlabel = self._problem.poincare_plot_xlabel
        if ylabel is None:
            ylabel = self._problem.poincare_plot_ylabel

        if plottype=='RZ':
            xdata = self.x
            ydata = self.z
        elif plottype=='yx':
            xdata = self.y
            ydata = self.x
        else:
            raise Exception('Choose the correct type for plottype')

        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else:
            fig, ax = plt.subplots()

        # set default plotting parameters
        # use dots
        if kwargs.get('marker') is None:
            kwargs.update({'marker': '.'})
        # use gray color
        # if kwargs.get('c') is None:
        #     kwargs.update({'c': 'gray'})
        # make plot depending on the 'range'
        

        for ii in range(self._params['nPtrj']+1):
            dots = ax.scatter(xdata[ii,:], ydata[ii,:], **kwargs)

        # adjust figure properties
        if plottype=='RZ':
            plt.axis('equal')
        if plottype=='yx':
            pass

        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        #plt.tight_layout()

    def plotiota(self, **kwargs):

        if not self.iota_successful:
            raise Exception('A successful call of compute_iota() is needed')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.siota, self.iota, **kwargs)

        plt.xlabel('s', fontsize=20)
        plt.ylabel('iotabar', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        #plt.tight_layout()

    @staticmethod
    def _run_poincare(params):

        """A function called in parallel to generate the Poincare plot for one starting point
            Called in PoincarePlot.compute, do not call otherwise
        """

        # copy the input to local
        integrator = params['integrator'].copy()
        nPpts = params['nPpts']
        t0 = params['t0']
        ic = params['ic']
        dt = params['dt']

        s = np.zeros([nPpts+1], dtype=np.float64)
        theta = np.zeros_like(s)
        zeta = np.zeros_like(s)


        integrator.set_initial_value(t0, ic)
        t = t0
        for jj in range(1,nPpts+1):

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
        output['s'] = s
        output['theta'] = theta
        output['zeta'] = zeta
        output['id'] = params['id']

        return output