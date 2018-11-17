'''
This program realises the 2D inverse Laplace reconstruction of a 2D relaxation experiment

It reads Bruker datasets, and produces binary 2D as well as html plots

The algo uses random projections in combination with a fast NNLS algorithm.
nnlsm_activeset :
M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450

Authors: L.Chiron and M-A Delsuc
date 2017-2018
license: GPL v3

for command usage, try
>> python ILT_2D.py --help
'''

from __future__ import division, print_function
import os, sys
import os.path as op
from generic_2D_ILT import ILT as ILT2D
import version

def run_ILT2D(folder_proc,resolution,kind,T1T2satrecov,alpha2d,data,name_data):
    """ 
    create ILT2D object, apply NNLSM, produce list of 2D points and save it
    """
    tt = ILT2D(data, kind=kind, start_decay=1, resol=resolution, rank=25, T1T2_saturation_recovery = T1T2satrecov)
    tt.NNLS_tikhonovK_simplified(alpha=alpha2d, kind_nnls='active-set', name=name_data)    # NNLS Tikhonov  
    tt.visu_proc()
    folder_save = os.path.join(folder_proc, 'bokeh')
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    folder_csv = os.path.join(folder_proc,'csv')
    if not os.path.exists(folder_csv):
        os.mkdir(folder_csv)
    tt.save2D(folder_proc,name_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # default values
    folder_proc = "./processed"
    data =  "./data_exemple/1/ser"

    name_data = "data_exemple_1"
    resolution = 40      # 2D Resolution

    kind = "T1T2"    # kind of dataset (T2T2,T1T2,DT2)
    T1T2satrecov=True
    alpha2d = 0.1

    parser.add_argument('--doc', action='store_true',
        help="print a description of the program")
    parser.add_argument('-n', '--dry', action='store_true',
        help="does not perform the computation - used for checking")
    parser.add_argument('-f', '--folder_proc', default=folder_proc, 
        help="the folder in which results are stored, default is %s"%folder_proc)
    parser.add_argument('-d', '--data', default=data,
        help="the data to be analysed, default is %s"%data)
    parser.add_argument('-r', '--resolution', type=int, default=resolution,
        help="the resolution of the final spectrum along both axes default=%d"%resolution)
    parser.add_argument('-k',  '--kind2D', default=kind,
        help="the kind of experiment to choose among T2T2, T1T2, DT2. default=%s"%kind)
    parser.add_argument('-v', '--inv_recov', action='store_true',
        help="for T1T2, uses inversion-recovery equation instead of saturation-recovery.")
    parser.add_argument('-a', '--alpha',  type=float, default=alpha2d,
        help="the value of alpha used for regularisation, default=%.2f"%alpha2d)

    args = parser.parse_args()

    if args.doc:
        print(__doc__)
        sys.exit(0)

    folder_proc = args.folder_proc
    data = args.data
    resolution = args.resolution
    kind = args.kind2D
    if kind not in ('T2T2', 'T1T2', 'DT2'):
        raise Exception('kind2D to choose among T2T2, T1T2, DT2')
    with_tt = " "
    if kind == 'T1T2':
        if args.inv_recov:
            T1T2satrecov = False
            with_tt = 'using Inversion-Recovery'
        else:
            with_tt = 'using Saturation-Recovery'
    alpha2d = args.alpha
    dirname1 = op.dirname(data)
    expname, expno = op.split(dirname1)
    expname = op.basename(expname)
    name_data = "%s_%s"%(expname,expno)
    print(expname)
    print("""
        Processing dataset "{0}"
        using {1} analysis, {2}
        using alpha = {3}    and resolution = {4}
    
        results stored in {5}
        """.format(data, kind, with_tt, alpha2d, resolution, op.join(folder_proc, name_data)))
    if args.dry:
        print('dry run!')
        sys.exit(0)
    if not os.path.exists(folder_proc):
        os.mkdir(folder_proc)
    try:
        run_ILT2D(folder_proc, resolution, kind, T1T2satrecov, alpha2d, data, name_data)
        print("""
        ======================================================================
                                 Processing complete
        Check results in {}
        ======================================================================
    """.format(op.join(folder_proc, name_data)))
    except:
        import traceback
        print("="*60)
        print('Error:',traceback.format_exc().splitlines()[-1],'\n')
        traceback.print_exc(limit=1)
        print("="*60)
        print("for command usage, try\n>> python ILT_2D.py --help")
        sys.exit(1)
    sys.exit(0)