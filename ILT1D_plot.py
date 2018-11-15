"""
Code for the 1D plots.. 
"""
from __future__ import division, print_function
import sys, os, shutil, time, json, re
opd, opb, opj = os.path.dirname, os.path.basename, os.path.join
from math import exp

from making_lists import csv2html

def make_plot_results_addresses(plt, folder_proc, name, mode, parab, mode_complement, debug=0):
    """
    folder_proc: 
    name:
    mode:
    parab: show paraboles for the area estimation
    mode_complement: decay, residual
    
    """
    if plt.type == 'bokeh':
        ext = '.html'
        folder_proc_plot = os.path.join(folder_proc, 'bokeh')
    else:
        ext = '.png'
        folder_proc_plot = os.path.join(folder_proc, 'images')
    if not os.path.exists(folder_proc_plot):
        os.mkdir(folder_proc_plot)
    if parab:
        name += '_parab'
    if mode == 'analysis':
        if mode_complement:
            name += '_' + mode_complement
        else:
            name += '_residual_and_decay'
    name_proc = ('proc_{0}_{1}').format(name, ext)
    if debug > 0:
        print ('name_proc is {0} ').format(name_proc)
    proc_file = os.path.join(folder_proc_plot, name_proc)
    if debug > 0:
        print ('saving at adress {0} ').format(proc_file)
        print ('plt.index_path_file {0} ').format(plt.index_path_file)
    return proc_file



def make_peakpicking(ilt, folder_proc, name):
    """
    Making the peakpicking and saving in json file.
    """
    # folder_pp = os.path.join(folder_proc, 'pp')
    # if not os.path.exists(folder_pp):
    #     os.mkdir(folder_pp)
    try:
        ilt.view.peakpicking()
    except:
        print('Error with peakpicking !!!!! ')

    if '.dps' in name:
        newname = re.findall('\\d+', name.split('.dps')[0])[0]
        print ('newname is {0} !!!!!', newname)
    else:
        newname = name
    dic_pp = {newname: ilt.view.list_peaks_areas_parab}
    # name_list_pp = os.path.join(folder_pp, 'pp_' + newname + '.json')
    # with open(name_list_pp, 'w') as (f):
    #     json.dump(dic_pp, f)


def make_plot(ilt, folder_proc, name, mode, parab=False, mode_complement=None, debug=0):
    """
    Make the plot (result, residual, decay)
    """
    proc_file = make_plot_results_addresses(folder_proc, name, mode, parab, mode_complement)
    ilt.display(parab=parab, mode=mode, mode_complement=mode_complement)
    # if debug > 2:
    #     print ('###### in make_plot after display, plt.list_plot  ', plt.list_plot)
    ilt.save(proc_file)


def save1d(ilt, folder_proc, name, debug=0):
    """
    Plot the results spectra with Bokeh as an interactive html file
              or with Matplotlib as a static png image file. Called by processing.
    Parameters:
        ilt: object ilt for making the processings
        plt: graphical object for producing Bokeh or Matplotlib
        folder_proc: folder containing the processings (Bokeh, csv, and image files).
        name: name of the processed file
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
        name_proc = ('proc_{0}_{1}.{2}').format(name, option, ext)
        return os.path.join(folder_proc_plot, name_proc)

    if debug > 0:
        print('################### in save1d')
        print(computeName('debug'))

    # makes all plots
    make_peakpicking(ilt, folder_proc, name)
    for opt in ('spec', 'pp', 'datalin', 'datalog', 'analyzelin', 'analyzelog','residual'):
        ext = 'html'
        ilt.display(mode=opt)
        with open(computeName(opt),'w') as f:
            f.write(ilt.view.graph_html)
        if opt == 'spec':
            ext = 'csv'
            name_proc = ('proc_{0}_{1}.{2}').format(name, opt, ext)
            with open(os.path.join(folder_proc_csv, name_proc),'w') as f:
                f.write('x,y\n')
                for x,y in zip(ilt.view.pos1D,ilt.view.meth.sol_meth):
                    f.write( "%.2f,%.2f\n"%(x,y) ) 
        if opt == 'pp':
            ext = 'csv'
            name_proc = ('proc_{0}_{1}.{2}').format(name, opt, ext)
            with open(os.path.join(folder_proc_csv, name_proc),'w') as f:
                f.write('T2 (ms),Intensity (a.u.),Width\n')
                for popt in ilt.view.list_fitted_parab: # Parabolae for surfaces
                    # x = xo +/- sqrt(2)/2 width
                    off = 0.707*popt[2]
                    width = abs(exp(popt[0]+off) - exp(popt[0]-off))
                    f.write( "%.2f,%.2f,%.2f\n"%(exp(popt[0]), popt[1], width) )
            ext='html'
            with open(os.path.join(folder_proc_plot,'{0}_peaklist.{1}'.format(name,ext)), 'w') as f:
                f.write( csv2html(os.path.join(folder_proc_csv, name_proc)) )

    return


    # for mode in ['raw','analysis']:
    #     if debug:
    #         print ('########## plotting mode is {0} !!!!! '.format(mode))
    #     folder_ctrl = opd(folder_proc)

    #     if mode == 'analysis':
    #         for mc in ['decay', 'residual']:
    #             if debug > 0:
    #                 print ('Making the plot for the {0} '.format(mc))

    #             make_plot(ilt, plt, folder_proc, name, mode, mode_complement=mc, debug=0)