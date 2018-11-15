# encoding: utf-8
# Utilities

from __future__ import division, print_function

import os
import numpy as np
from math import exp
import matplotlib.pylab as plt
plt.index_path_file = None
plt.type = 'mpl'
import os.path as op

###### Algos

from scipy.optimize import nnls     # scipy Non Negative Least Squares
from scipy import linalg
from scipy.linalg import norm
from  scipy.sparse.linalg import svds
#from sklearn import linear_model
from time import time
import savitzky_golay as sgm
from BrukerNMR import Import_1D 
from scipy.optimize import fmin, curve_fit # 
from scipy import interpolate
from sane import sane
from numpy.fft import rfft, ifft, irfft
# from bokeh.io import export_svgs
from matplotlib import pyplot as mplt
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot,layout,column,row
from bokeh.embed import components,file_html
from bokeh.models import Range1d,HoverTool,CrosshairTool,Toggle, CustomJS,LinearAxis,LogAxis
from bokeh.resources import CDN

class DATA_ILT(object):
    """
    read (from minispec) a T2 distribution dataset
    can also simulate using the read dataset as template
    data loaded in t,v  (time, value)
    """
    def __init__(self, addr, R2min, R2max, N=400, report=None, do_sane=False, debug=0):
        '''
        Parameters:
        addr : address of the input dataset
        R2min : minimum value for the damping factor range
        R2max : maximum value for the damping factor range
        N : number of values for the x axis of the Laplace Transform.
        '''
        self.do_sane = do_sane
        if self.do_sane:
            print("##### Using sane for cleaning the data")
        if debug>0: print("## in DATA_ILT, R2min {0} , R2max {1} ".format(R2min, R2max))
        self.R2min = R2min                  # Damping factor inferior limit
        self.R2max = R2max                  # Damping factor superior limit
        self.N = N                          # Number of points in the spectrum
        self.A = []                         # dictionary with the amplitudes of the gaussian shapes..
        self.R = []                         # dictionary with the damping factor values
        self.sigma = []                     # dictionary containing the widths of the gaussians..
        self.max_exp = 1.0                  # value max in the data space.
        file_ilt, ext = op.splitext(addr)
        if ext == '.dps':
            self.read_minispec(addr)  # Read the minispec dataset, time in milliseconds ## self.t, self.v = 
        elif op.basename(addr) == 'fid':
            self.read_Bruker(addr)    # Read Bruker dataset, time in milliseconds ## self.t, self.v = 
        # remove continuous level
        self.make_R2_axis()   # Make the R2 axis
        self.make_K()         # Make the Laplace matrix
        if report:
            report.write('        * min-max :: \n')
            report.write('            * min-max: {0:.3f}-{1:.3f} \n'.format(self.v.min(), self.v.max()) )

    def make_K(self):
        '''
        self.K is the matrix corresponding to the Laplace transform from the signal space to the data space.
        self.t in milliseconds
        '''
        M = len(self.t)
        self.K = np.exp(-np.kron(self.t.reshape((M,1)), self.axis.reshape((1,self.N))))

    def read_Bruker(self, fname):
        '''
        Routine used for reading data obtained with a classical Bruker spectrometer.
        Return : 
            self.v : signal
            self.t : x abscissa in milliseconds
        '''
        d = Import_1D(fname)
        if self.do_sane:
            self.v = np.real(sane(d.get_buffer(), 10))      # using sane
        else:
            self.v = np.real(d.get_buffer())      # using sane
        self.v[0] = self.v[1]
        milis = 2E3*float( d.params['acqu']['$D'][20] )    # assuming cpmg_T2 pulprog
        self.t = np.arange(self.v.size)*milis + milis/2    # times in milliseconds

    def read_minispec(self, fname):
        """
        Read an exponential decay from a Bruker minispec .dps text data file.
        Return time and values.. 
        """
        with open(fname,'r') as F:
            lines = F.readlines()
        time = []
        values = []
        for l in lines:
            if l.startswith("#"):   # skip comments
                continue
            v = l.strip().split()
            try:
                i = int(v[0])
                t = float(v[1])
                val = float(v[2])
            except:
                raise Exception("lines should contain 3 numbers")
            time.append(t)
            values.append(val)
        if i != len(time)-1:
            print("*** Warning possible size mismatch in %s"%fname)
        self.v = np.array(values)
        if self.do_sane:
            self.v = irfft(sane(ifft(rfft(self.v)), 10))
        t = np.array(time)
        self.t = t*1e2        # if time in 1/10 seconds instead of milliseconds
        print("#################### self.t is {0} ".format(self.t))

    def make_R2_axis(self, debug=0):
        "builds a log-spaced R2 axis"
        self.axis =  np.logspace(np.log10(self.R2min), np.log10(self.R2max), self.N)
        if debug>0:
            print("in make_R2_axis, R2min {0}, R2max {1} ".format(self.R2min, self.R2max))
            print("in make_R2_axis, self.axis is {0} ".format(self.axis))

    def simul_signal(self, noise=0, nb_random_shapes=False):
        '''
        Produce a signal in the signal space. The signal can be set to random.s
        Parameters:
            noise: gaussian noise added in the data space. - given in % -
            nbshape : number of random gaussian shapes which in signal space.
        '''
        if nb_random_shapes:
            self.A = np.abs(np.random.randn(nb_random_shapes)*1)
            self.R = np.abs(np.random.randn(nb_random_shapes)*0.01)
            self.sigma = np.abs(np.random.randn(nb_random_shapes)*0.001)

        def G(x, A, pos, sigma):
            '''
            Make a gaussian curve.
            Parameters:
                x: whole vector
                A: amplitude
                pos: position
                sigma: set the gaussian curve width..
            '''
            return A*np.exp(-((x-pos)/sigma)**2) # Return a gaussian curve
        self.distrib = np.zeros_like(self.axis)
        for i in range(len(self.A)):
            self.distrib += G(self.axis, self.A[i], self.R[i], self.sigma[i])  # make the whole signal in signal space

        self.v = np.dot(self.K, self.distrib)           # data from distribution in the signal space.

        #### Normalization for comparison with experimental curve

        fact_corr_A = self.max_exp/self.v.max()
        self.A /= fact_corr_A                                   # Correction on amplitudes
        self.distrib *= fact_corr_A                             # Correction on data space distribution
        self.v *= fact_corr_A                                   # Correction in the signal space
        self.v += np.random.randn(self.v.size)*(noise/100.0)    # adding white noise in the data space

    def show_plots(self, data_space=True, signal_space=True):
        '''
        Show the simulated data in data space (semilogx) and in signal space
        '''
        if data_space:
            plt.semilogx(self.axis, self.distrib)
        if signal_space:
            plt.figure()
            plt.xlabel('time')
            plt.plot(self.b)
        #plt.show()

def eval_noise(x, window_size=9, order=3):
    '''
    Noise evaluation (using Stavisky-Golay) for chi2 test. 
    '''
    m = sgm.sgolay_coef(window_size, order, deriv=0)
    noise = (sgm.sgolay_comp(x, m, window_size) - x).std()
    return noise

class METHODS_ILT(object):
    """
    Class containing methods for solving T2_ILT Inverse problem.
    Availbale methods are:
    * Ridge Regression (norm L2)
    * Lasso (norm L1)
    * ElasticNet ( norms L1 + L2)
    * NNLS (Non Negative Least Squares with Tikhonov regularization)
    """
    def __init__(self, similt, report=None):
        self.K = similt.K                # Matrix of the Laplace Transform to pass from the signal space to the data space.
        self.data = similt.v
        self.axis = similt.axis
        self.sol = {}                    # Dictionary containing the solutions of the ILT problem
        self.dic_meths = {'R': self.RidgeK, 'E': self.ElasticNetK, 'ECV': self.ElasticNetCVK,
                     'L': self.LassoK, 'S': self.SongK, 'N': self.NNLS_tikhonovK}
        self.noise = eval_noise(self.data)
        self.controls = {'chi2':[], 'energy':[]}
        if report:
            self.report = report

    def RidgeK(self, alpha = 0.000005):
        '''
        Ridge Regression, L2 regularization
        The solution of this problem usually contains ondulations.
        Not used.
        '''
        clf = linear_model.Ridge(alpha=alpha,  fit_intercept=False)
        clf.fit(self.K, self.data)
        self.sol['RidgeK'] = clf.coef_
        return clf.coef_

    def ElasticNetK(self, alpha=0.0001, l1_ratio= 0.9):
        '''
        ElasticNet, L1 + L2 regularization
        Not used.
        '''
        clf = linear_model.ElasticNet(alpha = alpha, l1_ratio= l1_ratio, fit_intercept=False)
        clf.fit(self.K, self.data)
        self.sol['ElasticNetK'] = clf.coef_
        return clf.coef_

    def ElasticNetCVK(self, alpha=0.001):
        '''
        ElasticNet with Cross validation
        Not used.
        '''
        clf = linear_model.ElasticNetCV(alpha=alpha, fit_intercept=False)
        clf.fit(self.K, self.data)
        self.sol['ElasticNetCVK'] = clf.coef_
        return clf.coef_

    def LassoK(self, alpha = 0.000005):
        '''
        Lasso, L1 norm minimization
        Not used.
        '''
        clf = linear_model.Lasso(alpha = alpha , fit_intercept=False)
        clf.fit(self.K, self.data)
        self.sol['LassoK'] = clf.coef_
        return clf.coef_

    def SongK(self):
        '''
        Song method, truncated SVD
        Not used.
        '''
        u, s, vt = svds(self.K)    #
        ###
        S = np.zeros((u[1].size, u[1].size))
        S[:u[1].size, :u[1].size] = np.diag(1/s)
        vs = np.dot(vt.T, S)
        invK = np.dot(vs, u.T)
        coef_ = np.dot(invK, self.data)
        self.sol['SongK'] = coef_
        return coef_

    def NNLS_tikhonovK(self, alpha=0):
        '''
        NNLS, non negative least-square with additive Tikhonov regularization
        Using nnls() from scipy.optimize
        Algorithm used for the processings.
        '''
        K1 = np.concatenate((self.K, alpha*np.identity(self.K.shape[1])))
        sig1 = np.concatenate((self.data, np.zeros(shape=(self.K.shape[1],))))
        coef_, self.rnorm_nnls_tikho = nnls(K1, sig1)
        self.sol['NNLS_tikhonovK'] = coef_
        return coef_

    def approx_random(self, k=10):
        '''
        Approximation with random projections.
        Not used.
        '''
        Omega = np.random.normal(size = (self.K.shape[1], k))                            # Omega random real gaussian matrix Nxk
        Y = np.dot(self.K,Omega)
        Q, r = linalg.qr(Y)
        Kapprox = np.dot(np.dot(Q, Q.T), self.K)
        sol_app = Kapprox
        print(self.K.shape)
        print(Kapprox.shape)
        ###
        return sol_app

    def make_all_methods(self):
        '''
        Perform all the methods
        Not used.
        '''
        self.LassoK()
        self.RidgeK(alpha=0.001)
        self.ElasticNetK()
#        self.ElasticNetCVK()
        self.SongK()
        self.NNLS_tikhonovK()

    def optim(self, meth,  interv, data= 's'):
        '''
        Routine for searching the right regularization parameter.
        Not used.
        '''

        lval = {}
        self.interv = np.linspace(interv[0],interv[1],15)
        for alpha in self.interv:
            asked_meth = self.dic_meths[meth]
            curr_meth = getattr(self, asked_meth)
            curr_meth(alpha = alpha)
            sol_meth  = self.sol[asked_meth]
            lval[alpha] = ((self.data-np.dot(self.K, sol_meth))**2).sum()
        plt.title('alpha optimization for {0}'.format(asked_meth))
        plt.plot(list(lval.keys()), list(lval.values()), 'rx')
        plt.xlabel('alpha')
        if plt.type == 'bokeh':
            plt.ylabel('chi2')
        else:
            plt.ylabel('$\chi^2$')
        plt.show()

    def analyze(self):
        '''
        Compute different values for controlling the quality of the processing
        '''
        self.backcalc = np.dot(self.K, self.sol_meth)     # Signal in observation space from solution
        self.residual = self.data-self.backcalc           # Residual
        #print(diff)
        self.error = (self.residual**2).sum()             # Error 
        self.chi2 = np.sqrt(self.error)/self.noise        # chi2
        self.nchi2 = self.chi2/np.sqrt(len(self.data))    # Normalized chi2
        self.energy = (self.sol_meth**2).sum()            # Energy of the solution
        self.controls['chi2'].append(self.chi2)
        self.controls['energy'].append(self.energy)

    def solve(self, meth='N', alpha=1e-3, l1_ratio=None, label=None):
        '''
        Apply methods with regularization parameter
        Parameters:
            * data : defines if we use real or simulated data
            * alpha : regularization parameter..
        '''
        def apply_meth(meth):
            '''
            Generic application of the methods
            '''
            asked_meth = self.dic_meths[meth]
            try:
                sol_meth = asked_meth(alpha = alpha, l1_ratio=l1_ratio)      # Apply the method with parameter alpha
            except:
                sol_meth = asked_meth(alpha = alpha)                         # Apply the method with parameter alpha
            return sol_meth
        if type(meth) == list:
            for m in meth:
                self.sol_meth = apply_meth(m)
        else:
            self.sol_meth = apply_meth(meth)

class VIEW_ILT(object):
    """
    display data and results from METHODS and DATA
    """
    def __init__ (self, data, methods, report=None):
        """
        data is a DATA_ILT
        methods is a METHODS_DATA
        """
        self.data = data
        self.meth = methods
        # filter for peaks area calculation and visualisation)
        self.lim_pp_down = 1.1   # limit down for filtering the peaks 
        self.lim_pp_up = 0.9     # limit up for filtering the peaks
        self.report = report

    def mean_L1(self, l):
        '''
        Find the mean value for the L1 regression.
        Parameters:
            * l : list of the values on which we make the L1 regression
        '''
        x = np.arange(len(l))
        delta1 = lambda param : np.abs(param[0]*x + param[1]-np.array(l)).sum()
        param = [1, 1]
        param_opt = fmin(delta1, param, xtol=1e-8) # L1 minimization
        a1, b1 =  param_opt[0], param_opt[1]
        mean = (a1*x+b1).mean()                    # middle position of the interval
        return mean

    def find_area(self, x, y, list_peaks, debug=0):
        '''
        Integral for (x, y), not using parabolae
        '''
        f_interp = interpolate.interp1d(x,y)
        xx = np.linspace(x.min(), x.max(), int(1e5))
        yy = f_interp(xx)
        interv = np.where(yy!=0)[0]
        diffint = np.diff(interv)  
        ind_sep = np.where(diffint>1)[0]       # index separating the regions
        ###########
        beg = 0
        if debug>0: print("ind_sep is ", ind_sep)
        self.list_peaks_areas = []
        plt.plot(xx, yy)
        ind_sep = np.append(ind_sep,interv[-1])
        for ind, i in enumerate(ind_sep):
            end = i
            int_sel = interv[beg:end+1]                  # interval selected
            plt.plot(xx[int_sel], yy[int_sel])
            integ = yy[int_sel].sum()                    # Integral calculation
            for p in list_peaks: 
                if xx[int_sel].min() < np.log(p) < xx[int_sel].max():
                    if debug>0: print("rounded values are p:{0} v:{1} ".format(round(p,2),round(integ,0)))
                    self.list_peaks_areas.append([round(float(p),2), round(integ,0)])
            beg = end +1
        if debug>0: print("self.list_peaks_areas ", self.list_peaks_areas)
        self.list_peaks_areas.sort()
        #plt.show()

    def parab(self, x, xo, intens, width):
        """
        the centroid definition
        """
        return intens*(1 - ((x-xo)/width)**2)
        # FWMH:
        #   ((x-xo)/width)**2 == 1/2
        #   |x-xo| = sqrt(2)/2 width
        #   x = xo +/- sqrt(2)/2 width

    def make_parab(self, x, y, ind, width, intens, pos, debug=0):
        '''
        Make the parabolae curves.
        '''
        widthN =  int(2*abs(width)/(x.max()-x.min()) * self.nbpts_fit)
        if debug>0:
            print("### widthN ", widthN)
            print("### ind ", ind)
        s = slice(ind-int(widthN),ind+int(widthN))
        if debug>0: print("##### s is ", s)
        x_part = x[s]
        y_part = y[s]
        if debug>0: print("### x_part.min(), x_part.max() ".format(x_part.min(), x_part.max()))
        #######
        x_interm = x_part
        y_interm = self.parab(x_part, pos, intens, width)
        ypos = np.where(y_interm>0)
        self.x_parab = x_interm[ypos]
        self.y_parab = y_interm[ypos]
        self.list_parab.append([np.exp(self.x_parab), self.y_parab])

    def fit_parab(self, x, y, p, N=500, width_guess=1, debug=0):
        '''
        Fit parabola
        x : abscissa of the peak
        y : height of the peak
        N : number of point on the right and on the left taken for the fit
        '''
        if debug>0: 
            print("peak p is  ", p)
        ind = list(x).index(p)
        if debug>0: print("ind is ", ind)
        s = slice(ind-N,ind+N)
        if debug>0: print("s is ", s)
        x_part = x[s]
        y_part = y[s]
        if debug>0:
            print("x_part.mean() is ", x_part.mean())
            print("y_part.mean() is ", y_part.mean())
            print("width_guess is ", width_guess)
        #plt.plot(x_part, y_part, 'b')
        guess = [x_part.mean(), y_part.mean(), width_guess]
        if debug>2:                      
            plt.figure()
            plt.title('fit_parab')
            plt.plot(x,y)
            plt.plot(x_part, y_part)
            plt.show()
        if debug>0:
            print("### using curve_fit !!")
            print("### guess  is ", guess)
        popt, pcov = curve_fit(self.parab, x_part, y_part, p0=guess)
        if debug>0: print("###### popt ", popt)
        pos, intens, width = popt[0], popt[1], popt[2]
        self.list_fitted_parab.append(popt)
        if debug>0: print("pos {0}, intens {1}, width {2}".format(pos, intens, width))
        area = abs(intens*4/3*width)                            # Area for a parabola for given width and intensity. 
        self.list_areas_parab.append(area)
        if debug>0: print("area with parabola is : ", area)
        self.make_parab(x, y, ind, width, intens, pos)                    # Makes the parabolae
        if debug>1:
            plt.plot(x,y)
            plt.plot(self.x_parab, self.y_parab, 'r--')                   # Plot the parabola
            plt.show()

    def find_area_parabola(self, x, y, list_peaks, debug=0):
        '''
        Find the area with parabolae, peakpicking on y, return position for x
        Calculations in log space. 
        '''
        #self.list_p = []
        self.list_fitted_parab = []
        self.list_areas_parab = []
        self.list_peaks_areas_parab = []
        self.list_parab = []
        if debug>0: print("### in find_area_parabola")
        f_interp = interpolate.interp1d(x,y)
        self.nbpts_fit = 100000
        xx = np.linspace(x.min(), x.max(), self.nbpts_fit)
        yy = f_interp(xx)
        if debug>0:
            print("made xx and yy ")
            print("x.min() {0}, x.max() {1} ".format(x.min(), x.max()))
            print("yy.max() ", yy.max())
        ind_pp = np.where(self.cndry(1,yy) & self.cndry(-1,yy))                # recalculate peaks
        if debug>0: print("made ind_pp")
        list_peaks_precise =  xx[ind_pp]  # 
        lpp = list_peaks_precise
        if debug>0: print('### limits R remove peaks are {0} and {1} '.format(self.data.R2min*self.lim_pp_down,self.data.R2max*self.lim_pp_up))
        if debug>0: print('### limits T remove peaks are {0} and {1} '.format(1/self.data.R2max*self.lim_pp_down, 1/self.data.R2min*self.lim_pp_up))
        list_peaks_precise = lpp[(lpp>np.log(1/self.data.R2max*self.lim_pp_down)) & (lpp<np.log(1/self.data.R2min*self.lim_pp_up))]     # Avoiding problems with the edges

        if debug>0: print("list_peaks_precise is ", list_peaks_precise)
        for p in list_peaks_precise:
            try:
                if debug>0: print('Dealing with peak ', p)
                self.fit_parab(xx, yy, p)                                                                  # Fit the parabolae
            except:
                print('Did not find peak !!!!')
        #  Dealing with normalization
        lpa = np.array(self.list_peaks_areas)                 # peaks
        lap = np.array(self.list_areas_parab)  # areas  np.nan_to_num()
        if debug>0: print("before normalization, self.list_areas_parab ", self.list_areas_parab)
        norm = lpa.max()/lap.max()
        if debug>0: print("lap.max() ", lap.max())
        self.list_areas_parab = list(norm*lap)    # np.round(,0)                                            # Normalisation
        if debug>0: print("after normalization, self.list_areas_parab ", self.list_areas_parab)
        self.list_areas_parab = np.round(self.list_areas_parab,0)
        #self.list_areas_parab = [ "{0:.3g}".format(float(surf)) for surf in self.list_areas_parab]                  # scientific format for area
        if debug>0: print("format numbers {0} ".format(self.list_areas_parab))
        self.list_peaks_areas_parab = list(zip(list_peaks[::-1], self.list_areas_parab))                    # mix peaks and areas
        if debug>0: print("after zip we have, self.list_peaks_areas_parab ", self.list_peaks_areas_parab)
        self.list_peaks_areas_parab.sort()                                                                  # sort on peaks

        if debug>0: print("after sort we have, self.list_peaks_areas_parab ", self.list_peaks_areas_parab)
        
    def cndry(self, roll, y, debug=0):
        '''
        General condition with roll
        '''
        if debug>0: print("### in cndry")
        return y > np.roll(y, roll)

    def cndr(self, roll):
        '''
        One condition with roll
        '''
        return self.meth.sol_meth > np.roll(self.meth.sol_meth, roll)

    def peakpicking(self, debug=0):
        '''
        Make the peak picking and 
        '''
        if debug>0 :  print("####### type(self.meth.sol_meth) is  ", type(self.meth.sol_meth))
        list_peaks_precise = 1/self.data.axis[np.where(self.cndr(1) & self.cndr(-1))]
        list_peaks =  np.round(list_peaks_precise, 2)                                        # rounded list peaks
        lp = list_peaks
        if debug>0 : print("#####  list of peaks before ", list_peaks)
        list_peaks_filtered = lp[(lp>1/self.data.R2max*self.lim_pp_down) & (lp<1/self.data.R2min*self.lim_pp_up)]     # avoiding the issues on the edges.. *1e3
        self.list_peaks_filtered = list_peaks_filtered
        if debug>0 :
            print("#####  list of list_peaks_log_filtered is ", list_peaks_filtered)
            print('### Calculating areas !!!!!!!! ')
        try:
            self.find_area(np.log(1/self.data.axis), self.meth.sol_meth, list_peaks_filtered)              # Calculating areas under the peaks.. 
        except:
            print('Could not calculate area with self.find_area !!! ')
        try:
            self.find_area_parabola(np.log(1/self.data.axis), self.meth.sol_meth, list_peaks_filtered)     # Calculating areas under the peaks with parabolae.. 
        except:
            print('Could not calculate area with self.find_area_parabola !!! ')

    def display(self, mode='results', mode_complement=None,  newfig=True, debug=0):
        """
        show results from processing
        Possible modes are :  raw, results, data and analysis
        parab: plot parabolae for area after peakpicking
        mode: raw, results data or analysis, by default results is used.
        mode_complement:
        """
        if debug>0: print('########## in ILT_tools.display !!! in mode {0}'.format(mode))
        TOOLS="pan, box_zoom, hover, undo, redo, reset, save"
        alphab=['a','b','c','d','e','f','g','h','i','j','k']
        diccol = {'r':'red', 'b':'blue', 'g':'green', 'o':'orange', 'k':'black', 'm':'magenta', 'f':'grey'}
        dbk = {'tools': TOOLS, 'sizing_mode':'scale_width'}
        dfig = {}
        self.meth.analyze()
        if mode in ['spec', 'pp']:                            #########  Show the result of the reconstruction, mode used in T2_ILT program
            dbk['title'] = 'T2 Spectrum'
            axx = 1/self.data.axis
            dbk['x_axis_type'] = "log"
            dbk['x_range'] = Range1d(1E0, 1E4)
            dbk['x_axis_label'] = 'T2(ms)'
            dbk['y_axis_label'] = 'a.u.'
            f1 = figure(**dbk)
            dfig["x"] = axx
            self.pos1D = axx
            dfig["y"] = self.meth.sol_meth
            dfig["legend"] = "Spectrum"
            if mode == 'pp':
                dfig['line_dash']=[4, 4]
                for p,c in zip(self.list_parab, diccol.values()): # Parabolae for surfaces
                    d_subfig = {}
                    d_subfig["x"] = p[0]
                    d_subfig["y"] = p[1]
                    d_subfig["color"]  = c
                    pp_x = []
                    pp_y=[]
                    for popt in self.list_fitted_parab:
                        pp_x.append(exp(popt[0]))
                        pp_y.append(popt[1])
                    f1.circle_x(pp_x,pp_y,size=20,color="#DD1C77", fill_alpha=0.2)
                    f1.line(**d_subfig)
            f1.line(**dfig)
            self.graph_html = file_html(f1,CDN)

        elif mode in ('datalin', 'datalog', 'analyzelin', 'analyzelog'):                           # Show the data used for reconstruction
            if mode in ('datalog', 'analyzelog'):
                dbk['y_axis_type'] = "log"
            if mode.startswith('data'):
                dbk['title'] = 'Original Data'
            else:
                dbk['title'] = 'Reconstructed Data'
            dbk['x_axis_label'] = 'ms'
            dbk['y_axis_label'] = 'a.u.'
            f2 = figure(**dbk)
            dfig["y"] = self.data.v
            dfig["x"] = self.data.t
            dfig["legend"] = "Original Data"
            f2.line(**dfig)
            if mode.startswith('analyze'):
                dfig['color'] = 'green'
                dfig["y"] = self.meth.backcalc
                dfig["legend"] = "Reconstructed Data"
                f2.line(**dfig)
            data_html = file_html(f2,CDN)
            self.graph_html=data_html

        elif mode == 'residual':
            dfig["y"] = self.meth.residual
            dbk['title'] = "Residual - normalized chi2 = %.2f"%self.meth.nchi2
            dbk['x_axis_label'] = 'ms'
            dbk['y_axis_label'] = 'a.u.'
            f3 = figure(**dbk)
            dfig["x"] = self.data.t
            dfig["size"] = 3
            f3.scatter(**dfig)
            data_html = file_html(f3,CDN)
            self.graph_html=data_html
        else:
            print("Internal ERROR in ILT_tools.display")
        return

        # if mode == 'analysis':                       # Show the reconstructed decay and the residual
        #     self.meth.analyze()
        #     if plt.type == 'bokeh':
        #         kindline = '-'      # Bokeh
        #     else:
        #         self.plt.subplot(211)
        #         kindline = '.'     # Mpl

        #     ####################           Decay

        #     if not mode_complement or mode_complement == 'decay':
        #         if debug>0 : print('#### dealing with decay')

        #         #self.report.write('        * test du chi2 : {0:.2f} \n'.format(self.meth.nchi2))

        #         self.plt.semilogy(self.data.t, self.data.v, kindline, label='original', **kwargs)   # Original decay
        #         self.plt.semilogy(self.data.t, self.meth.backcalc,  **kwargs)                       # Decay after reconstruction
        #         if debug>0: print('######## ylim are :  min {0}, max {1} '.format(self.data.v.min(), self.data.v.max()))
        #         self.plt.ylim(ymin=self.data.v.min(), ymax=self.data.v.max())
        #         print("#### Decay min max : self.data.v.min() {0} , self.data.v.max() {1} ".format(self.data.v.min(), self.data.v.max()))
        #         self.plt.title("Original and Reconstructed data")
        #         self.plt.legend()
        #         self.plt.xlabel('points')
        #         if debug>0: print('##### At the end of decay, self.plt.list_plot ', self.plt.list_plot)
        #         if debug>0: print('######## lim self.data.t are :  min {0}, max {1} '.format(self.data.t.min(), self.data.t.max()))
            
        #     ####################            Residual

        #     if not mode_complement or mode_complement =='residual':
        #         if debug>0 :
        #             print('#### dealing with residual')
        #             print('### For residual, self.plot_type is ', self.plt.plot_type)
        #         ####### Bokeh
        #         if plt.type == 'bokeh':
        #             self.plt.plot_type ='linear'
        #             #self.plt.figure()   # making new figure after decay plot
        #             self.plt.plot(self.data.t, self.meth.residual, '*', **kwargs)  # Bokeh
        #             if debug>3: print('##### In residual, self.plt.list_plot ', self.plt.list_plot)
        #             self.plt.title(r"Residual - normalized chi2 = %.2f"%self.meth.nchi2)
        #         ####### Mpl
        #         else: 
        #             self.plt.subplot(212)
        #             self.plt.plot(self.data.t, self.meth.residual, '*', **kwargs)  # Mpl
        #             self.plt.title(r"Residual - normalized $\chi^2 = %.2f$"%self.meth.nchi2)  
        #         ######## 
        #         mean_res = self.mean_L1(self.meth.residual)      # Using norm L1
        #         std_res = min(2*self.meth.residual.std(), 500) 
        #         liminf, lisup =  mean_res-std_res, mean_res+std_res         
        #         plt.ylim(liminf, lisup)
        #         self.plt.xlabel('points')
        #     if plt.type == 'bokeh':
        #         if debug>1: print(plt.list_plot)
        #         #if mode_complement == 'residual' or mode == 'raw':  # Calling show once all plots done
        #         self.plt.show()  # With Bokeh need to create the plot with show                 
        # else:
        #     raise Exception("non-valid option")
            
        # return graph_html

    def show(self):
        self.plt.show()
    def save(self, name, dpi=55, debug=0):
        print("#### Name for 1D is {0} ".format(name))
        try:
            self.plt.savefig(name, dpi=dpi)   # Save figure with dpi value (for Mpl)
            print("saved as Mpl plot")
        except:
            if debug>0: print('save without dpi')
            self.plt.savefig(name)                 # Bokeh
        if self.plt.type == 'bokeh':
            try:
                self.plt.figure()                  # Reinitialize the Bokeh parameters xlim, yilm etc..
            except:
                if debug>0: print('not Reinitializing with plt.figure()')

class ILT(object):
    """
    Class for performing T2_ILT analysis.
    Available methods are:
        * solve : solve the inverse problem with available methods (Lasso, NNLS, Ridge regression etc.. )
        * display : prepare the data representation ( mode: "data", "compare", "results" etc.. )
        * show : show the figure in Matplotlib

    """
    def __init__(self, addr, R2min=1E-4, R2max=1.0, N=400, report=None, debug=0):
        '''
        Read the data then instantiate self.meth and self.view
        '''
        self.data = DATA_ILT(addr, R2min=R2min, R2max=R2max, N=N, report=report)
        self.meth = METHODS_ILT(self.data, report=report)
        self.view = VIEW_ILT(self.data, self.meth, report=report)
        if debug>0: print("## in ILT, R2min is {0}, R2max is {1} ".format(R2min,R2max))

    def solve(self, meth='N', alpha=1e-3, l1_ratio=None, label=None, debug=0):
        """
        Solve the T2_ILT
        Available problems to be solved are:
        * 'R': Ridge Regression
        * 'E': ElasticNet
        * 'L': Lasso
        * 'N': NNLS (Non Negative Least Square) + Tikhonov
        """
        self.meth.data = self.data.v                 # take into account if self.data.v is modified
        self.meth.solve(meth=meth, alpha=alpha, l1_ratio=l1_ratio, label=label)
        if debug>0: print("alpha ", alpha)

    def display(self, mode='results', **kwargs):
        """
        Display the data or/and results of the Inverse problem with correct T2 axis.
        Available mode are:
        * results : plot all the results obtained with the different approaches used for solving the Inverse Problem.
        * raw : plot the solution of the Inverse problem.
        * data : plot the dataset used for the Inverse problem
        * analysis : plot the sum of exponential decays of the orginal dataset and reconstructed dataset.
        """
        self.view.display(mode=mode, **kwargs)

    def show(self):
        self.view.show()
    def save(self, name):
        self.view.save(name)

if __name__ == '__main__':
    if False :
        addr = "Manip_T2_CASC4DE/20.dps"    # Example on dps files
        ilt = ILT(addr, 1e-4, 0.1, N=400)
        ilt.data.A = [12,70,11,60]
        ilt.data.R = [0.001,0.002,0.007,0.06]
        ilt.data.sigma = [0.0001,0.0001, 0.0002, 0.01]
        ilt.data.simul_signal()
        ilt.display(mode='data', color='k', lineStyle='--')
        ilt.show()
    ###
    addr = "T2_billes_verre_Julia_1dec16/10/fid"
    ilt = ILT(addr, 1e-4, 0.1, N=400)
    ilt.solve(alpha=0.01)
    #ilt.display(mode='results', color='b', lineStyle='-') # show with different methods
    ilt.display(mode='raw', color='b', lineStyle='-')          # show with one method, if not precised it is NNLS.
    ilt.show()
