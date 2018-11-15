'''
This program realises the 1D inverse Laplace reconstruction of a T2 relaxation experiment

It reads Bruker datasets, and produces binary 1D as well as html plots

The algo uses random projections in combination with a fast NNLS algorithm.
nnlsm_activeset :
M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450

Authors: L.Chiron and M-A Delsuc
date 2017-2018
license: CC BY-NC-SA
    This work is licensed under a license 
    Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
    http://creativecommons.org/licenses/by-nc-sa/4.0/

'''
from __future__ import division, print_function
import os, sys
import os.path as op
from ILT_tools import ILT as ILT1D
from ILT1D_plot import save1d
 


def run_ILT1D(folder_proc, data, name_data, hight, lowt, alpha1d):
    """ 
    create ILT1D object, produce the plot and save it
    """
    ilt = ILT1D(data, 1/hight, 1/lowt,  N=400, report=None)               # Instantiate the class for ILT
    ilt.solve(alpha=alpha1d)                                              # Solve the ILT problem
    save1d(ilt, folder_proc, name_data)
    #save_comparison(ilt, folder_proc, name_data)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # default values
    folder_proc = "./processed"
    data =  "./data_exemple/1/fid"

    name_data = "data_exemple_1"

    alpha1d = 0.1

    hight = 1e4

    lowt = 1


    parser.add_argument('--doc', action='store_true',
        help="print a description of the program")
    parser.add_argument('-n', '--dry', action='store_true',
        help="does not perform the computation - used for checking")
    parser.add_argument('-f', '--folder_proc', default=folder_proc, 
        help="the folder in which results are stored, default is %s"%folder_proc)
    parser.add_argument('-d', '--data', default=data,
        help="the data to be analysed, default is %s"%data)
    parser.add_argument('-a', '--alpha',  type=float, default=alpha1d,
        help="the value of alpha used for regularisation, default=%.2f"%alpha1d)
    parser.add_argument('-g', '--hight',  type=float, default=hight,
        help="the largest T2 in msec, default=%.2f"%hight)
    parser.add_argument('-l', '--lowt',  type=float, default=lowt,
        help="the smallest T2 in msec, default=%.2f"%lowt)

    args = parser.parse_args()

    if args.doc:
        print(__doc__)
        sys.exit(0)

    folder_proc = args.folder_proc
    data = args.data
    lowt = args.lowt
    hight = args.hight
    alpha1d = args.alpha
    dirname1 = op.dirname(data)
    expname, expno = op.split(dirname1)
    expname = op.basename(expname)
    name_data = "%s_%s"%(expname,expno)
    print(expname)
    print("""
        Processing dataset "{0}"
        T2 analysis, from {2} to {3} msec
        using alpha = {4}
    
        results stored in {5}
        """.format(data, " ", lowt, hight, alpha1d, op.join(folder_proc, name_data)))
    if args.dry:
        print('dry run!')
        sys.exit(0)
    if not os.path.exists(folder_proc):
        os.mkdir(folder_proc)

    run_ILT1D(folder_proc, data, name_data, hight, lowt, alpha1d)