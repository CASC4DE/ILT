#!/usr/bin/env python
# encoding: utf-8
'''
2D inverse Laplace reconstruction from a Bruker dataset.
The algo uses random projections in combination with a fast NNLS algorithm.
nnlsm_activeset :
M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450
'''
from __future__ import division, print_function
import re
import os
op = os.path
opd, opb, opj = os.path.dirname, os.path.basename, os.path.join
import pickle
import csv
from time import time

import numpy as np
from scipy.optimize import nnls                         # scipy Non Negative Least Squares
from scipy import linalg, sparse
from matplotlib import pyplot as mplt
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot,layout,column,row
from bokeh.embed import components,file_html
from bokeh.models import Range1d,HoverTool,CrosshairTool,Toggle, CustomJS,LinearAxis,LogAxis
from bokeh.resources import CDN

from making_lists import csv2html
from BrukerNMR import Import_2D
from nnlsm import nnlsm_activeset



#T1T2_saturation_recovery = True

class ILT(object):
    def __init__(self, addr, kind="T2T2", start_decay=1, resol=None,
                    factthresh=[1,1], rank=16, threshold=None, T1T2_saturation_recovery=True, debug=True):
        '''
        addr: path of the dataset to be processed
        kind : kind of 2D (T2T2, T1T2, DT2)
        start_decay :
        resol : vertical and horizontal resolution
        sketch : kind of sketch procedure
        rank : rank for the random projection
        threshold: threshold for avoiding spurious spikes
        '''
        global plt, mplt

        self.T1T2_saturation_recovery = T1T2_saturation_recovery
        self.kind = kind
        if self.kind == 'T2T2' : self.F1, self.F2 =  [1e0,1e4], [1e0,1e4]
        elif self.kind == 'T1T2' : self.F1, self.F2 = [1e1,1e5], [1e0,1e4]
        elif self.kind == 'DT2' : self.F1, self.F2 = [1e1,1e4], [1e0,1e4]
        self.label_kind = {'T2T2' : ['T2(ms)', 'T2(ms)'], 'T1T2' : ['T1(ms)', 'T2(ms)'], 'DT2' : [r'D ($\mu m^2/s$)', 'T2(ms)']}
        self.k = rank
        self.start_decay = start_decay
        self.resol = resol
        N = [resol]*2
        self.addr = addr
        self.factthresh = factthresh
        self.threshold = threshold
        self.sol = {}
        self.K = {}
        self.simpleK = {}                    # simplififed kernels
        self.axis = {}
        self.N = {}
        self.N[1], self.N[2] = N[0], N[1]
        self.Rmin = {}                       # R axis min
        self.Rmax = {}                       # R axis max
        self.t = {}
        #plt = BOKEH_PLOT(plot_width = 400, plot_height = 400)
        mplt.figure()
        self.prepare_bounds()
        self.prepare_processing()

    def prepare_bounds(self, debug=0):
        '''
        Prepare the bounds for each case (T2T2, T1T2 and DT2)
        '''
        if self.kind == 'T2T2':
            # self.Tmax, self.Tmin = self.F1[1], self.F1[0]
            # self.R1, self.R2 = 1/self.Tmax, 1/self.Tmin
            # self.Rmin[1], self.Rmax[1] = min(self.R1, self.R2), max(self.R1, self.R2)
            # self.Rmin[2], self.Rmax[2] = self.Rmin[1], self.Rmax[1]
            self.T1max, self.T1min = self.F1[1], self.F1[0]
            self.T2max, self.T2min = self.F2[1], self.F2[0]
            self.R11, self.R12 = 1/self.T1max, 1/self.T1min
            self.R21, self.R22 = 1/self.T2max, 1/self.T2min
            self.Rmin[1], self.Rmax[1] = min(self.R11, self.R12), max(self.R11, self.R12)
            self.Rmin[2], self.Rmax[2] = min(self.R21, self.R22), max(self.R21, self.R22)
        elif self.kind == 'T1T2':
            self.T1max, self.T1min = self.F1[1], self.F1[0]
            self.T2max, self.T2min = self.F2[1], self.F2[0]
            self.R11, self.R12 = 1/self.T1max, 1/self.T1min
            self.R21, self.R22 = 1/self.T2max, 1/self.T2min
            self.Rmin[1], self.Rmax[1] = min(self.R11, self.R12), max(self.R11, self.R12)
            self.Rmin[2], self.Rmax[2] = min(self.R21, self.R22), max(self.R21, self.R22)
        elif self.kind == 'DT2':
            self.Dmax, self.Dmin = self.F1[1], self.F1[0]
            self.T2max, self.T2min = self.F2[1], self.F2[0]
            self.R11, self.R12 = self.Dmin, self.Dmax
            self.R21, self.R22 = 1/self.T2max, 1/self.T2min
            self.Rmin[1], self.Rmax[1] = min(self.R11, self.R12), max(self.R11, self.R12)
            self.Rmin[2], self.Rmax[2] = min(self.R21, self.R22), max(self.R21, self.R22)

    def debug_prepare_processing(self, time_elapsed, debug=0):
        '''
        '''
        print("time for compressing is {0} min".format(time_elapsed))
        if debug>0:
            print("self.simpleK[1].shape {0} , self.simpleK[2].shape {1}".format(self.simpleK[1].shape , self.simpleK[2].shape))
        if debug>1:
            for i in range(self.simpleK0.shape[1]):  # Show dictionary
                y = self.simpleK0[:,i]
                x = np.arange(y.size)
                mplt.plot(x,y)
            if show:
                mplt.show()
            mplt.close()

    def prepare_processing(self, show=False, debug=0):
        '''
        Prepare the processing from list delay, dataset etc..
        '''
        self.find_delays()
        self.sig_retrieve()
        self.make_R_axis(1)
        self.make_K(1)
        self.make_R_axis(2)
        self.make_K(2)
        t0compress = time()
        self.simplif_basis()                     # Using random projection to make the basis of projection for reduction.
        self.simplifM()                              # Simplifying the target matrix, result is : self.simpleM
        self.make_simpleK(1)
        self.make_simpleK(2)
        self.simpleK0 = np.kron(self.simpleK[1], self.simpleK[2])           # Cartesian product of the simplified kernels
        t1compress = time()
        time_elapsed = (t1compress-t0compress)/60
        self.debug_prepare_processing(time_elapsed, debug)

    def find_delays(self, debug=0):
        '''
        Builds self.list_delay from vclist, vdlist or difflist.
        '''
        # sp = self.addr.split('/')[:-1]
        dic_list = {'T2T2':'vclist', 'T1T2':'vdlist', 'DT2':'difflist'}
        sp = op.dirname(self.addr)
        vcl_addr = op.join(sp, dic_list[self.kind])
        # sp += [dic_list[self.kind]]
        # if debug>0: print(type(sp))
        # spl_addr = iter(sp)
        # if debug>0: print(spl_addr)
        # vcl_addr = ('/').join(spl_addr)
        if debug>0:
            print("self.kind is ", self.kind)
            print("##### vcl_addr is ", vcl_addr)
        self.list_delays = []
        with open(vcl_addr) as f:
            l = f.readlines()
            for ll in l:
                if self.kind == 'T2T2':
                    self.list_delays.append(int(ll))
                elif self.kind == 'T1T2':
                    self.list_delays.append(float(ll))
                elif self.kind == 'DT2':
                    self.list_delays.append(float(ll)*np.sqrt(1.35)) # gradient correction of np.sqrt(1.35)
        self.list_delays = self.list_delays[self.start_decay-1:]
        if debug>0:
            print("## self.list_delays ", self.list_delays)
            print("## len(self.list_delays) ", len(self.list_delays))
            print("## len(self.list_delays)", len(self.list_delays))

    def calibdosy(self, litdelta, bigdelta, recovery=0.0, seq_type='ste', nucleus='1H', maxgrad=50.0, maxtab=50.0, gradshape=1.0, unbalancing=0.2):
        """
        returns the DOSY calibrating factor from the parameters

          bigdelta float
           "Big Delta"  : diffusion delay in msec

          litdelta float
           "little Delta"  : gradient duration in msec

          seq_type enum "pgse","ste","bpp_ste","ste_2echoes","bpp_ste_2echoes","oneshot" / default ste
           the type of DOSY sequence used
            pgse : the standard hahn echoe sequence
            ste : the standard stimulated echoe sequence
            bpp_ste : ste with bipolar gradient pulses
            ste_2echoes : ste compensated for convection
            bpp_ste_2echoes : bpp_ste compensated for convection
            oneshot : the oneshot sequence from Pelta, Morris, Stchedroff, Hammond, 2002, Magn.Reson.Chem. 40, p147

          nucleus enum "1H","2H","13C","15N","17O","19F","31P" / default 1H
           the observed nucleus

         recovery float
           Gradient recovery delay

         maxgrad float
           Maximum Amplificator Gradient Intensity, in G/cm    / default 50.0

         maxtab float
           Maximum Tabulated Gradient Value in the tabulated file. / default 100.0
           Bruker users with gradient list in G/cm (difflist) use maxgrad here
           Bruker users with gradient list in % use 100 here
           Varian users use 32768 here

         gradshape float
           integral factor depending on the gradient shape used / default 1.0
           typical values are :
               1.0 for rectangular gradients
               0.6366 = 2/pi for sine bell gradients
               0.4839496 for 4% truncated gaussian (Bruker gauss.100 file)
           Bruker users using difflist use 1.0 here, as it is already included in difflist

        """
    # MAD : modified august-sept 2007 - corrected ste; added oneshot; added PGSE

        g = (maxgrad / maxtab)*1E-4 # now in Tesla/cm
        aire = g*gradshape*litdelta
        if nucleus == "1H":
            gama = 2.675E8                           # rad.s-1.T-1
        elif nucleus == '2H':
            gama = 0.411E8                           # rad.s-1.T-1
        elif nucleus =='13C':
            gama = 0.673E8                           # rad.s-1.T-1
        elif nucleus == '15N':
            gama = -0.271E8                          # rad.s-1.T-1
        elif nucleus == '17O':
            gama = -0.363E8                          # rad.s-1.T-1
        elif nucleus =='19F':
            gama = 2.517E8                           # rad.s-1.T-1
        elif nucleus =='31P':
            gama = 1.083E8                           # rad.s-1.T-1
        else:
            raise'Unknown nucleus'

        K = ((gama * aire)**2)            # Calcul de q^2

    # equation references are in    Jerschow,A.;Muller,N.;JMR;125;1997;Suppresion of convection artifacts
        if seq_type == 'ste' or seq_type == 'D_T2' :
            K = (K * (bigdelta + ((2 * litdelta)/3) + recovery))                 # cm 2 sec-1 pour Q
        elif seq_type == 'bpp_ste':
            K = (K * (bigdelta + ((2 * litdelta)/3) + ((3 * recovery)/4))) # cm 2 sec-1 pour Q
        elif seq_type == 'ste_2echoes':
            K = (K * (bigdelta + ((4 * litdelta)/3) + (2 * recovery)))     # cm 2 sec-1 pour Q
        elif seq_type == 'bpp_ste_2echoes':
            K = (K * (bigdelta + ((4 * litdelta)/3) + ((3 * recovery)/2))) # cm 2 sec-1 pour Q
        elif seq_type == 'oneshot':
            K = (K * (bigdelta + litdelta * (unbalancing * unbalancing - 2) / 6 + recovery * (unbalancing * unbalancing - 1) / 2))
        elif seq_type == 'pgse':
            K = (K * bigdelta + (2 * litdelta)/3)
        else:
            raise 'Unknown sequence'

        K = (K * 1e-8)      # from cm^2 to um^2
        return(1/K)

    def dcalibdosy(self, npk, nucleus='1H'):
        """use stored parameters to determine correct DOSY calbiration"""
        d20 = float(npk.params['acqu']['$D'][20])
        d40 = float(npk.params['acqu']['$D'][40])
        d16 = float(npk.params['acqu']['$D'][16])
        d17 = float(npk.params['acqu']['$D'][17])
        p1 =  float(npk.params['acqu']["$P"][1])*1e-6
        p19 = float(npk.params['acqu']["$P"][19])*1e-6
        p30 = float(npk.params['acqu']['$P'][30])*1e-6

        nuc1 = npk.params['acqu']["$NUC1"]
        if nucleus is None:
            if (nuc1 == '1H' or nuc1 == '15N' or nuc1 == '13C' or nuc1 == '31P' or nuc1 == '19F' or nuc1 == '17O'):
                nucleus = nuc1
            else:
                nucleus = '1H'
        print ("DOSY performed on %s"%(nucleus,))
        #print("npk.params['acqu'] ", npk.params['acqu'])
        pulprog = npk.params['acqu']['$PULPROG']
        seq_type = self.determine_seqtype(pulprog[1:-1])
        print("seq_type is ", seq_type)

        # STEBP_2echos Bruker avance sequences
        if seq_type == 'bpp_ste_2echoes':
            litdelta = (2*p30)
            bigdelta = (d20-(10*p1)-(8*p30)-(8*d16)-(8*d17)-(2*p19))
            recovery = d16
        # STE_2echos Bruker avance sequences
        elif seq_type == 'ste_2echoes':
            litdelta = p30
            bigdelta = (2*(d20-(2*p1)-(p30)-(2*d16)-(p19)))
            recovery = d16
        # BPP_LED NMRtec and Bruker Avance sequences
        elif seq_type == 'bpp_ste':
            litdelta = 2*p30
            bigdelta = d20-(4*p1)-(2*p30)-(3*d16)-(p19)
            recovery = 2*d16
        # LEDgp/STEgp Bruker Avance sequence
        elif seq_type == 'ste':
            litdelta = p30
            bigdelta = d20-(2*p1)-(p30)-(2*d16)-(p19)
            recovery = d16
        # D_T2_2d for D-T2 2D Laplace spectroscopy
        elif seq_type == 'D_T2':
            litdelta = p30
            bigdelta = d40-(2*p1)-(p30)-(2*d16)-(p19)
            recovery = d16
        #Doneshot from Morris and Nillson
        elif seq_type == 'oneshot':
            litdelta = 2*p30
            bigdelta = d20-(4*p1)-(2*p30)-(3*d16)-(p19)
            recovery = 2*d16
        else:
            litdelta = p30
            bigdelta = d20
            recovery = d16

        print (litdelta, bigdelta, recovery, seq_type, nucleus)
        #npk.axis1.dfactor = self.calibdosy(litdelta, bigdelta, recovery, seq_type=seq_type, nucleus=nucleus)
        dfactor = self.calibdosy(litdelta, bigdelta, recovery, seq_type=seq_type, nucleus=nucleus)
        return dfactor

    def determine_seqtype(self, pulprog):
        """
        given the PULPROG name, determines which seq_type is to be used
        PULPROG should be follow the standard Bruker naming scheme
        """
        # Bruker avance sequences
        if (re.search('dstebp',pulprog)):
            sequence = 'bpp_ste_2echoes'
        elif re.search('dstegp',pulprog):
            sequence = 'ste_2echoes'
        elif re.search('stegpbp|ledbp',pulprog):
            sequence = 'bpp_ste'
        elif re.search('stegp|led',pulprog):
            sequence = 'ste'
        elif re.search('oneshot',pulprog):
            sequence = 'oneshot'
        elif re.search('D_T2',pulprog):
            sequence = 'D_T2'
        else:
            print("<%s> : Unsupported pulse program."%pulprog)
            sequence = "None"
        print (sequence)
        return sequence

    def debug_sig_retrieve(self, sig, nblines, debug=0):
        '''
        '''
        if debug>0:
            print("sig.size ", sig.size)
            print("self.t[2].size*nblines {0} ".format(self.t[2].size*nblines))
            print("self.t[1].size is ", self.t[1].size)
            print("self.t[2].size is ", self.t[2].size)
            print("### sig.size  is ", sig.size )
        if debug>1:
            print("self.vv.size ", self.vv.size)
            print("self.v.shape ", self.v.shape)
            print('plotting the flattened signal ')
            if self.kind == 'DT2':
                print("Dfactor:",self.dcalibdosy(d))
            plt.title('Signal to be reconstructed')
            plt.plot(np.arange(self.vv.size), np.array(self.vv))
            plt.savefig('signal_retrieved.html')
            if show:
                plt.show()

    def test_pos(self, d, row, debug=True, debug_plot=False):
        '''
        Test if the signal is positive for T2T2..
        '''
        sig = d.real().get_buffer()[row,0:]
        if debug_plot:
            mplt.plot(sig, label="in test_pos")
            mplt.legend()
            mplt.show()
            print("### sig[1] {0}, sig[-1] {1} ".format(sig[1],sig[-1]))

        if ((sig[sig.size//4] > sig[sig.size//2]) and sig.sum()>0 and self.kind in ["T2T2",'T1T2']) or self.kind == 'DT2'  :
            if debug: print('########### Positive !!!')
            return 1
        else:
            if debug:
                print('########### Negative !!!')
                print("sig[1]{0} , sig[-1] {1} ".format(sig[1], sig[-1]))
                print("sig.sum() ", sig.sum())
            return -1

    def sig_retrieve(self, show=False, debug=1, debug_show_plot=False):
        '''
        Retrieving dataset
        '''
        print("######## Processing {0} !!!! ".format(self.kind))
        d = Import_2D(self.addr)                            # Reading the "ser" file

        millis = 2E3*float( d.params['acqu']['$D'][20] )    # assuming cpmg_T2 pulprog
        self.millis = millis
        buff = d.real().get_buffer()
        signsig = self.test_pos(d, row=7) # Test if signal is positive
        if debug > 0:
            print("signsig is ",signsig)
        if debug_show_plot:
            for i in range(buff.shape[0]):
                mplt.plot(buff[i,0:], label="before correction {0} ".format(i))
                mplt.legend()
                mplt.show()
        buff *=signsig
        # for i in range(buff.shape[0]):
        #     if i<5:
        #         buff[i,0:] = 0
        self.threshold = d.row(0).real()[1]                          # Threshold, positive value..
        if self.kind == 'T1T2' : # and self.T1T2_saturation_recovery
            self.threshold = np.abs(buff).max()                      # Threshold, positive value..
        if debug>0:
            print("#### Threshold is {0} ".format(self.threshold) )
            print("### buff.max() {0} !!! ".format(buff.max()))
        if debug_show_plot:
            for i in range(buff.shape[0]):
                mplt.plot(buff[i,0:], label="correction with sign of {0} ".format(i))
                mplt.legend()
                mplt.show()
        if debug > 0:
            print("### buff.shape ", buff.shape)
        self.vfirst = np.clip(buff[self.start_decay-1:, 1:], -self.threshold*self.factthresh[1],
                                   self.threshold*self.factthresh[0])                             #  Removing the spike
        # self.vfirst = np.clip(buff[self.start_decay:, 1:], -self.threshold*self.factthresh[1],
        #                    self.threshold*self.factthresh[0])
        sig = self.vfirst.copy().flatten()
        if debug > 0:
            print("### self.vfirst.shape ", self.vfirst.shape)                                    #  Matrix flattened
            print("### sig.shape ", sig.shape)
        ####
        self.t[2] = np.arange(self.vfirst.shape[1])*millis + millis/2                             #  horizontal
        if self.kind == 'T2T2':
            self.t[1] = np.array(self.list_delays)*millis + millis/2                              #  add initial dead time
        elif self.kind == 'T1T2':
            self.t[1] = np.array(self.list_delays)* 1E3                                           #     was in sec, change to msec
        elif self.kind == 'DT2':
            self.t[1] = ( np.array(self.list_delays)**2 )/self.dcalibdosy(d)                      # build laplace axis
        ##
        nblines = self.t[1].size
        ##
        if debug > 0:
            print("### self.t[1].size ", self.t[1].size)
            print("### self.t[2].size ", self.t[2].size)
            print("#### nblines {0} , self.t[2].size {1}".format(nblines, self.t[2].size))
            print("### sig.shape ",sig.shape)
        self.vv = sig[:self.t[2].size*nblines] #
        self.v = sig.reshape(nblines, self.t[2].size) #
        self.debug_sig_retrieve(sig, nblines, debug=debug)

    def prepare_matrix_and_signal(self, alpha, debug=False):
        '''
        Concatenate simpleK0 with simpleMflat for regularization purpose.
        '''
        K1 = np.concatenate((self.simpleK0, alpha*np.identity(self.simpleK0.shape[1])))           # Matrix
        sig1 = np.concatenate((self.simpleMflat, np.zeros(shape=(self.simpleK0.shape[1],))))      # targeted signal
        if debug:
            print('self.simpleK0.shape[1] ', self.simpleK0.shape[1])
            print('self.simpleK0.shape ', self.simpleK0.shape)
            print('self.simpleMflat.shape ', self.simpleMflat.shape)
            print("K1.shape {0} , sig1.shape {1} ".format(K1.shape, sig1.shape))
        return K1, sig1

    def solving_with_nnls(self, K1, sig1, kind_nnls, debug=False):
        '''
        Solving the NNLS problem
        '''
        t0 = time()
        ### Scipy nnls
        if kind_nnls == 'scipy':
            coef_, self.rnorm_nnls_tikho = nnls(K1, sig1)   # coef_ is the vector result..  # NNLS Scipy
        ### Using nnlsm active_set
        elif kind_nnls == 'active-set':
            sig1.resize(sig1.shape[0],1)
            coef_, infos = nnlsm_activeset(K1, sig1)           # NNLS nnlsm active-set
            if debug: print("##### coef_.shape ", coef_.shape)
        t1 = time()
        print("Time for processing is {0} min ".format((t1-t0)/60))
        return  coef_

    def debug_tikhonov(self, reconstruction):
        '''
        debug for NNLS Tikhonov
        '''
        print("### in NNLS_tikhonovK ")
        print("reconstruction.shape ", reconstruction.shape)
        print("self.simpleM.shape", self.simpleM.shape)
        print("self.simpleMflat.shape ", self.simpleMflat.shape)
        print("self.Q[1][:,:self.k].shape ", self.Q[1][:,:self.k].shape)
        print("self.Q[2][:,:self.k].T.shape ", self.Q[2][:,:self.k].T.shape)
        print("self.simpleMflat.shape[0] ", self.simpleMflat.shape[0])

    def NNLS_tikhonovK_simplified(self, alpha=1, kind_nnls='active-set', name='data', debug=True):
        '''
        NNLS, non negative least-square with additive Tikhonov regularization
        Using nnls() from scipy.optimize
        Algorithm used for the processings.
        alpha : regularization parameter
        kind_nnls : kind of algoirthm used
        '''
        print("in NNLS_tikhonovK")
        K1, sig1  = self.prepare_matrix_and_signal(alpha)
        self.sig1 = sig1
        self.alpha = alpha
        coef_ = self.solving_with_nnls(K1, sig1, kind_nnls)
        self.sol['NNLS_tikhonovK'] = coef_
        ####
        reconstruction = np.dot(K1, coef_)
        self.reconstruction = reconstruction
        title_fig = '{0}_{1}_control.html'.format(name, self.kind)
        if debug: self.debug_tikhonov(reconstruction)
        return coef_

    def debug_ticksF1F2(self, linscale1, linscale2, logscale1, logscale2):
        '''
        '''
        print("## np.log10(self.F1[0]) ", np.log10(self.F1[0]))
        print("linscale1 = ",linscale1)
        print("linscale2 = ",linscale2)
        print("logscale1 = ",logscale1)
        print("logscale2 = ",logscale2)

    def ticksF1F2(self, debug=0):
        '''
        Ticks for T2T2, T1T2 and DT2 experiments
        '''
        self.nbticks2 = int(np.log10(self.F2[1]/self.F2[0]))+1                                           # number of ticks horizontally (one/decade)
        self.nbticks1 = int(np.log10(self.F1[1]/self.F1[0]))+1                                           # number of ticks vertically (one/decade)
        listlogscale2 = list(np.logspace(np.log10(self.F2[0]), np.log10(self.F2[1]), self.nbticks2))     # values for horizontal ticks
        listlogscale1 = list(np.logspace(np.log10(self.F1[0]), np.log10(self.F1[1]), self.nbticks1))     # values for vertical ticks
        ###
        logscale2 =  list(map("{0:.0e}".format, listlogscale2))                         # text of the horizontal ticks
        logscale1 =  list(map("{:.2e}".format, listlogscale1))                          # text of the vertical ticks
        linscale2 = list(np.linspace(self.F2[0], self.F2[1], self.nbticks2))            # positions for the horizontal ticks
        linscale1 = list(np.linspace(self.F1[0], self.F1[1], self.nbticks1))            # positions for the vertical ticks
        if debug>0: self.debug_ticksF1F2(linscale1, linscale2, logscale1, logscale2)
        #### Ticks with log scale
        mplt.xticks( linscale2, logscale2 )
        mplt.yticks( linscale1, logscale1 )

    def make_axes(self):
        '''
        Makes the axes self.axisF2 and self.axisF1
        '''
        self.axisF2 = np.linspace(self.F2[0], self.F2[1], self.N[1])
        self.axisF1 = np.linspace(self.F1[0], self.F1[1], self.N[1])
        mplt.xlabel(self.label_kind[self.kind][1])   # horizontal axis label
        mplt.ylabel(self.label_kind[self.kind][0])   # vertical axis label

    def list_maxima(self, sol, debug=True):
        '''
        List of the peaks for the peakpicking
        '''
        posmaxsol = np.where(\
             (sol>np.roll(sol, 1, axis=0)) &
             (sol>np.roll(sol, 1, axis=1)) &
             (sol>np.roll(sol, -1, axis=0)) &
             (sol>np.roll(sol, -1, axis=1))
             )
        if debug: print("posmaxsol ", posmaxsol)
        lmaxsol = zip(list(posmaxsol[0]), list(posmaxsol[1]))
        if debug: print("lmaxsol ",lmaxsol)
        return lmaxsol

    def peakpicking(self, sol, file_pp=None, show_maxima=True, debug=False):
        '''
        Find peaks in the 2D spectrum
        return x, y, z lists
        '''
        m = sol.max()
        levels = (m*0.00625, m*0.0125, m*0.025, m*0.05, m*0.1, m*0.25, m*0.5)       # Levels for the contour plot
        lmaxsol = self.list_maxima(sol)
        def pp2(j):
            return self.axisF2[j]
#            return self.F2[0]*10**((j+1)/self.resol*(self.nbticks2-1))
        def pp1(i):
            return self.axisF1[i]
#            return self.F1[0]*10**((i+1)/self.resol*(self.nbticks1-1))
        x = []
        y = []
        z = []
        for i,j in lmaxsol:
            x.append(round( pp2(j), 2))
            y.append(round( pp1(i), 2))
            z.append(sol[i,j])
        if show_maxima:
            title_x='T2 (ms)'
            if self.kind == "T2T2":
                title_y='T2(ms)'
            elif self.kind == "T1T2":
                title_y='T1(ms)'
            elif self.kind == "DT2":
                title_y=u'Diff(µm²/s)'
            else:
                raise Exception('Internal error')
        return x,y,z

    def make_plot(self, ax, sol, scale=1,remove_artefacts=True, debug=0):
        '''
        Plot with axes according to the kind of experience
        '''
        if remove_artefacts:
            sol[:2,:] = 0
            sol[-2:,:] = 0
        m = sol.max()/scale
        #m = sol.max()
        #levels = (m*0.00625, m*0.0125, m*0.025, m*0.05, m*0.1, m*0.25, m*0.5)   # levels for contour plot
        levels = (m*0.00156, m*0.00312, m*0.00625, m*0.0125, m*0.025, m*0.05, m*0.1, m*0.25, m*0.5)   # levels for contour plot
        mplt.tick_params(labelsize = 8)
        mplt.subplots_adjust(left=0.2, right=0.75, top=0.9, bottom=0.15)     # margins for the picture
        self.nbticks2 = int(np.log10(self.F2[1]/self.F2[0]))+1                                           # number of ticks horizontally (one/decade)
        self.nbticks1 = int(np.log10(self.F1[1]/self.F1[0]))+1                                           # number of ticks vertically (one/decade)
        listlogscale2 = list(np.logspace(np.log10(self.F2[0]), np.log10(self.F2[1]), self.nbticks2))     # values for horizontal ticks
        listlogscale1 = list(np.logspace(np.log10(self.F1[0]), np.log10(self.F1[1]), self.nbticks1))     # values for vertical ticks
        ###
        logscale2 =  list(map("{0:.0e}".format, listlogscale2))                         # text of the horizontal ticks
        logscale1 =  list(map("{:.2e}".format, listlogscale1))                          # text of the vertical ticks
        linscale2 = list(np.linspace(self.F2[0], self.F2[1], self.nbticks2))            # positions for the horizontal ticks
        linscale1 = list(np.linspace(self.F1[0], self.F1[1], self.nbticks1))            # positions for the vertical ticks
        if debug>0: self.debug_ticksF1F2(linscale1, linscale2, logscale1, logscale2)
        #### Ticks with log scale
        ax.set_xticks( linscale2, logscale2 )
        ax.set_yticks( linscale1, logscale1 ) 
        (nf2,nf1) = sol.shape
        self.axisF2 = np.logspace(np.log10(self.F2[0]), np.log10(self.F2[1]), nf2)
        self.axisF1 = np.logspace(np.log10(self.F1[0]), np.log10(self.F1[1]), nf1)
        ax.contour(self.axisF2,self.axisF1, sol, levels)   #

        ax.set_xlabel('T2')
        if self.kind == "T2T2":
            ax.set_ylabel('T2')
        elif self.kind == "T1T2":
            ax.set_ylabel('T1')
        elif self.kind == "DT2":
            ax.set_ylabel('Diff')
        return ax

    def get_contour_data(self,ax):
        """
        Get informations about contours created by matplotlib.
        ax is the input matplotlob contour ax (cf. fig,ax produced by matplotlib)
        xs and ys are the different contour lines got out of the matplotlib. col is the color corresponding to the lines.
        """
        xs = []
        ys = []
        col = []
        isolevelid = 0
        for isolevel in ax.collections:
            isocol = isolevel.get_color()[0]
            thecol = 3 * [None]
            theiso = str(ax.collections[isolevelid].get_array())
            isolevelid += 1
            for i in range(3):
                thecol[i] = int(255 * isocol[i])
            thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])
            for path in isolevel.get_paths():
                v = path.vertices
                x = v[:, 0]
                y = v[:, 1]
                xs.append(x.tolist())
                ys.append(y.tolist())
                col.append(thecol)
        return xs, ys, col

    def debug_visu_proc(self):
        '''
        '''
        print("sol.shape ", sol.shape)
        print("sol.max() after ", sol.max())               # maximum height in solution
        print(dir(ax))

    def signals_for_comparison(self):
        '''
        '''
        q1k = self.Q[1][:,:self.k]
        q2kt = self.Q[2][:,:self.k].T
        resmat = self.reconstruction[:self.simpleMflat.shape[0]].reshape(self.simpleM.shape)
        result_calculation = np.dot(np.dot(q1k, resmat ), q2kt).flatten()
        sig_after_randomproj = np.dot(np.dot(q1k, self.sig1[:self.simpleMflat.shape[0]].reshape(self.simpleM.shape) ), q2kt).flatten()
        return result_calculation, sig_after_randomproj

    def create_plot(self, name,mode="spec", scale = 1.0):
        """
        creates displayed plots
        dumps also peaks in html and csv
        """
        TOOLS="pan, box_zoom, hover, undo, redo, reset, save"
        diccol = {'r':'red', 'b':'blue', 'g':'green', 'o':'orange', 'k':'black', 'm':'magenta', 'f':'grey'}
        dbk = {'tools': TOOLS, 'sizing_mode':'scale_width'}
        dfig = {}
        fact = 1
        if self.kind in ['T2T2', 'T1T2']:
            self.savesol = self.sol['NNLS_tikhonovK'].reshape(self.N[1], self.N[2])[::-1,::-1]
        elif self.kind == 'DT2':
            self.savesol = self.sol['NNLS_tikhonovK'].reshape(self.N[1], self.N[2])[:,::-1]
        #if debug>0: print("sol.max() before ", sol.max())                                # maximum height in solution
        self.savesol *= (1/self.savesol.max())                                                             # diminish height
        self.listpp_x, self.listpp_y, self.listpp_z = self.peakpicking(self.savesol) #creates the peak list html file in pandas dataframe format
        if mode in ("spec","pp"):
            self.fig, ax = mplt.subplots() 
            xs, ys, col = self.get_contour_data(self.make_plot(ax, self.savesol, scale=scale,remove_artefacts=True, debug=0))
            self.xlab='T2 (ms)'
            dbk['x_axis_label'] = self.xlab
            if self.kind == "T2T2":
                self.ylab='T2 (ms)'
            elif self.kind == "T1T2":
                self.ylab='T1 (ms)'
            elif self.kind == "DT2":
                self.ylab=u'Diff (µm²/s)'
            dbk['y_axis_label']=self.ylab
            dbk['title'] = self.kind+ ' spectrum'
            dbk['x_axis_type'] = 'log'
            dbk['y_axis_type'] = 'log'
            min_xs = []
            max_xs = []
            min_ys = []
            max_ys = []
            for i in range(len(xs)):
                min_xs.append(min(xs[i])) 
                max_xs.append(max(xs[i]))
            for i in range(len(ys)):
                min_ys.append(min(ys[i]))
                max_ys.append(max(ys[i]))
            dbk['x_range'] = Range1d(1E0, 2*max(max_xs))
            dbk['y_range'] = Range1d(1E0, 2*max(max_ys))
            p = figure(**dbk)
            dfig['xs']=xs
            dfig['ys']=ys
            dfig['color']=col
            p.multi_line(**dfig)
            p.line([1, 2*max(max_xs),2*max(max_xs),1,1], [1,1,2*max(max_ys),2*max(max_ys),1], line_width=1.5, color='black')
            if mode == "pp":
                p.circle_x(self.listpp_x, self.listpp_y, size=20,color="#DD1C77", fill_alpha=0.2)
            self.html_plot=file_html(p,CDN)
        elif mode in ("fidlin","fidlog"):
            result_calculation, sig_after_randomproj = self.signals_for_comparison()
            print("Plotting comparison")
            dbk['x_axis_label']='#FID'
            dbk['y_axis_label']='Intensity'
            dbk['x_axis_type'] = "linear"
            if mode == "fidlin":
                dbk['title'] = 'Linear FID'
                dbk['y_axis_type'] = "linear"
                p = figure(**dbk)
                p.line(np.arange(result_calculation.size//fact), result_calculation[::fact], legend='Calculated',line_width=1,line_color='blue')
                p.line(np.arange(self.vv.size//fact), self.vv[::fact], legend='Original',line_color='green')
                p.line(np.arange(sig_after_randomproj.size//fact), sig_after_randomproj[::fact], 
                    line_dash=[4, 4],legend='Signal with random projection',line_color='red')
            elif mode == "fidlog":
                dbk['title'] = 'Logarithmic FID'
                dbk['y_axis_type'] = "log"
                p = figure(**dbk)
                p.line(np.arange(self.vv.size//fact), self.vv[::fact], legend='Original',line_color='green')
                p.line(np.arange(result_calculation.size//fact), 
                    np.maximum(1.0,result_calculation[::fact]), legend='Calculated',line_width=1,line_color='blue')
            self.html_plot=file_html(p,CDN)
        elif mode == "residual":
            result_calculation, sig_after_randomproj = self.signals_for_comparison()
            dfig["y"] = (self.vv[::fact])-(result_calculation[::fact])
            dbk['title'] = "Residual"
            dbk['x_axis_label'] = 'a.u.'
            dbk['y_axis_label'] = 'a.u.'
            p = figure(**dbk)
            dfig["x"] = np.arange(self.vv.size//fact)
            dfig["size"] = 3
            p.scatter(**dfig)
            self.html_plot=file_html(p,CDN)
        else:
            print("Internal ERROR in generic_2D_ILT.create_plot")
        return 

    def save2D(self,folder_proc, name, debug=0):
        """
        Saves the processing results for 2D data in html format
        """
        folder_proc_plot = os.path.join(folder_proc, 'bokeh')
        folder_proc_csv = os.path.join(folder_proc, 'csv')
        if not os.path.exists(folder_proc_plot):
            os.mkdir(folder_proc_plot)
        if not os.path.exists(folder_proc_csv):
            os.mkdir(folder_proc_csv)
        ext = 'html'
        def computeName(option):
            "used to create file name on the fly"
            name_proc = ('{0}_{1}_ilt_2D_{2}.{3}'.format(name, self.kind,option,ext))
            return os.path.join(folder_proc_plot, name_proc)
        for opt in ('spec', 'pp', 'fidlin','fidlog','residual'):
            ext = 'html'
            self.create_plot(name,mode=opt)
            with open(computeName(opt),'w') as f:
                f.write(self.html_plot)
            if opt == 'spec':
                ext = 'csv'
                name_proc = ('{0}_{1}_ilt_2D_{2}.{3}').format(name, self.kind,opt, ext)
                sol =self.savesol
                with open(os.path.join(folder_proc_csv, name_proc),'a') as f:
                    f.write('x,y,z\n')
                    for i1 in range(sol.shape[0]):
                        if1 = self.axisF1[i1]
                        for i2 in range(sol.shape[1]):
                            if2 = self.axisF2[i2]
                            f.write("%.2f,%.2f,%.2f\n"%(if1,if2,100*sol[i1,i2]))
            if opt == 'pp':
                ext = 'csv'
                name_proc = ('{0}_{1}_ilt_2D_{2}.{3}').format(name, self.kind,opt, ext)
                with open(os.path.join(folder_proc_csv, name_proc), 'w') as f:
                    f.write('%s,%s,Intensity\n'%(self.xlab, self.ylab))
                    for x,y,z in zip(self.listpp_x,self.listpp_y,self.listpp_z):
                        f.write( "%.2f,%.2f,%.2f\n"%(x,y,100*z) )   # z is normalized to 1.00
                #then in html
                ext='html'
                with open(os.path.join(folder_proc_plot,'{0}_{1}_ilt_2D_peaklist.{2}'.format(name, self.kind,ext)), 'w') as f:
                    f.write( csv2html(os.path.join(folder_proc_csv, name_proc)) )
        return

    def visu_proc(self, file_pp=None, show=False, debug=0):
        '''
        Visualisation of the result
        using mplt.contour
        '''
        if debug>0: print("##### in visu_proc !!")
        if self.kind in ['T2T2', 'T1T2']:
            sol = self.sol['NNLS_tikhonovK'].reshape(self.N[1], self.N[2])[::-1,::-1]
        elif self.kind == 'DT2':
            sol = self.sol['NNLS_tikhonovK'].reshape(self.N[1], self.N[2])[:,::-1]
        if debug>0: print("sol.max() before ", sol.max())                                # maximum height in solution
        sol *= (1/sol.max())                                                             # diminish height
        #ax = mplt.gca()
        self.fig, ax = mplt.subplots()
        if debug>0:
            self.debug_visu_proc()
        self.make_axes()
        self.make_plot(ax, sol)                 # Make the 2D relaxation plot
        self.ticksF1F2()
        self.peakpicking(sol)
        if show:
            mplt.show()

    def bebug_make_K(self, ind, debug=0):
        '''
        Debug the dictionary
        '''
        col0 = self.K[ind][:,0]
        if debug>0:
            print("###### col0.min() {0}, col0.max() {1} ".format(col0.min(),col0.max()))
            print('### self.kind is ', self.kind)
        with open('a'.format(ind), 'wb') as f:
            pickle.dump(tt, f)
        with open('b'.format(ind), 'wb') as f:
            pickle.dump(self.axis[ind], f)
        mplt.figure()
        for i in range(self.K[ind].shape[1]):
            mplt.plot(self.K[ind][:,i])
        mplt.show()

    def make_K(self, ind, show=False, debug=0):
        '''
        self.K[ind] : matrix corresponding to the Laplace transform from the signal space to the data space.
        self.t[ind] : time in ms
        Using sparse.csr_matrix
        '''
        if debug>0: print("self.t[ind] ", self.t[ind])
        M = len(self.t[ind])
        if debug>0:
            print("## axis {2} min {0}, max {1} ".format(self.axis[ind].min(), self.axis[ind].max(), ind))
            print("## self.t {1} size is {0} ".format(self.t[ind].size, ind))
        tt = self.t[ind].reshape((M,1))
        a = sparse.csr_matrix(tt)
        b = sparse.csr_matrix(self.axis[ind].reshape((1,self.N[ind])))
        if self.kind in ['T2T2','DT2']:
            if debug>0:  print("############ Making K matrices !!!!")
            self.K[ind] = np.exp(-sparse.kron(a,b).toarray()) # modulation by T2
        elif self.kind == 'T1T2':
            if ind == 1:
                if not self.T1T2_saturation_recovery:
                    self.K[ind] = -(1-2*np.exp(-sparse.kron(a,b).toarray())) # modulation by T1 inversion recovery
                else:
                    self.K[ind] = 1-np.exp(-sparse.kron(a,b).toarray()) # modulation by T1 saturation recovery
            elif ind == 2:
                self.K[ind] = np.exp(-sparse.kron(a,b).toarray())
        if debug>0:
            self.bebug_make_K(ind)
            with open('K{}'.format(ind), 'wb') as f:
                pickle.dump(self.K[ind], f)     # Dump  matrix

    def debug_make_R_axis(self, ind, debug):
        '''
        '''
        if debug>1:
            print("self.axis[{1}].size {0}".format(self.axis[ind].size, ind))
            print("self.N[ind] is ", self.N[ind])
        if debug>0 and ind == 1:
            print("self.axis[ind].min() {0}, self.axis[ind].max() {1} ".format(self.axis[1].min(), self.axis[1].max()))
            print("self.axis[ind].size ", self.axis[1].size)

    def make_R_axis(self, ind, debug=0):
        '''
        builds R axes with log scales.
        '''
        logrmin, logrmax = np.log10(self.Rmin[ind]), np.log10(self.Rmax[ind])
        self.axis[ind] = np.logspace(logrmin, logrmax, self.N[ind])
        self.debug_make_R_axis(ind, debug)

    def debug_simplif_basis(self, i):
        '''
        '''
        #print("self.K[i] ",self.K[i])
        print("### in simplif ")
        print("self.k is ", self.k)
        print("self.Q[{0}].shape is {1} ".format(i, self.Q[i].shape))
        print("self.K[{0}].shape is {1} ".format(i, self.K[i].shape))
        print("self.Y.shape is {0} ".format(Y.shape))

    def simplif_basis(self, debug=0):
        '''
        Building the random basis Q for simplifying the calculations
        '''
        self.Q = {}
        print("in simplif_basis")
        if debug>0: print("self.K ", self.K)
        for i in [1,2]:
            Omega = np.random.normal(size = (self.K[i].shape[1], self.k))    # Omega random real gaussian matrix Nxk
            Y = np.dot(self.K[i], Omega)
            self.Q[i], r = linalg.qr(Y)
            if debug>0: self.debug_simplif_basis(i)

    def debug_simplifM(self):
        '''
        '''
        print("### in simplifM")
        print("self.v.shape ", self.v.shape)
        print("self.Q[1][:,:self.k].T.shape ", self.Q[1][:, :self.k].T.shape)
        print("self.Q[2][:,:self.k].shape ", self.Q[2][:, :self.k].shape)

    def simplifM(self, debug=0):
        '''
        Simplify the target matrix (solution matrix)
        '''
        if debug>0: self.debug_simplifM()
        self.simpleM = np.dot(np.dot(self.Q[1][:,:self.k].T, self.v), self.Q[2][:, :self.k])    # Simplify the target matrix
        self.simpleMflat = self.simpleM.flatten()                                               # Flattening the simplified matrix.
        if debug>0: print("## Result: self.simpleM.shape ", self.simpleM.shape)

    def random_p(self, length):
        '''
        Makes the p vector
        '''
        onesezer = np.array(np.concatenate((np.ones(self.k), np.zeros(length-self.k))).tolist())
        onesezer = np.random.permutation(onesezer)
        return onesezer

    def debug_make_simpleK(self, ind):
        '''
        Debug for K matrix
        '''
        print("### in make_simpleK ")
        print("self.Q[{0}].shape {1} ".format(ind, self.Q[ind].shape))
        print("self.K[{0}].shape {1}".format(ind, self.K[ind].shape))
        print("## Result: self.simpleK[{0}].shape {1} ".format(ind, self.simpleK[ind].shape))

    def make_simpleK(self, ind, debug=0):
        '''
        Simplify the kernels
        '''

        self.simpleK[ind] = np.dot(self.Q[ind][:,:self.k].T, self.K[ind])      # Simplifying the kernels with matrix Q
        if debug>0:
            self.debug_make_simpleK(ind)
            with open('simpleK{}'.format(ind), 'wb') as f:
                pickle.dump(self.simpleK[ind], f)
