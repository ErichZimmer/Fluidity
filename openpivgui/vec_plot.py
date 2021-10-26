#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Plotting vector data.

This module can be used in two different ways:

1. As a library. Just import the module and call the functions.
   This is the way, how this module is used in openpivgui, for
   example.

2. As a terminal-application. Execute 
   python3 -m openpivgui.vec_plot --help
   for more information.
   This is the way, how this module ist used in JPIV, for example.
   For now, not all functions are callable in this way.
'''

__licence__ = '''
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

__email__= 'vennemann@fh-muenster.de'


import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import points_in_poly
from openpiv.preprocess import prepare_mask_on_grid as grid_mask
from openpivgui.open_piv_gui_tools import (coords_to_xymask, 
    add_disp_roi, add_disp_mask)

# creating a custom rainbow colormap
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# creating a custom rainbow colormap
#short_rainbow = {'red':(
#                 (0.0, 0.0, 0.0),
#                 (0.2, 0.2, 0.2),
#                 (0.5, 0.0, 0.0),
#                 (0.8, 1.0, 1.0),
#                 (1.0, 1.0, 1.0)),
#        'green':((0.0, 0.0, 0.0),
#                 (0.2, 1.0, 1.0),
#                 (0.5, 1.0, 1.0),
#                 (0.8, 1.0, 1.0),
#                 (1.0, 0.0, 0.0)),
#        'blue': ((0.0, 1.0, 1.0),
#                 (0.2, 1.0, 1.0),
#                 (0.5, 0.0, 0.0),
#                 (0.8, 0.0, 0.0),
#                 (1.0, 0.0, 0.0))}

short_rainbow = {
        'red':  ((0.0,  0.0, 0.0),
                 (0.27, 0.0, 0.0),
                 (0.54, 0.0, 0.0),
                 (0.80, 1.0, 1.0),
                 (1.0,  1.0, 1.0)),
        'green':((0.0,  0.0, 0.0),
                 (0.27, 1.0, 1.0),
                 (0.55, 1.0, 1.0),
                 (0.80, 1.0, 1.0),
                 (1.0,  0.0, 0.0)),
        'blue': ((0.0,  1.0, 1.0),
                 (0.27, 1.0, 1.0),
                 (0.54, 0.0, 0.0),
                 (0.80, 0.0, 0.0),
                 (1.0,  0.0, 0.0))}

long_rainbow = {'red': 
                ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.3, 0.2, 0.2),
                 (0.5, 0.0, 0.0),
                 (0.7, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.3, 1.0, 1.0),
                 (0.5, 1.0, 1.0),
                 (0.7, 1.0, 1.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.3, 0.3)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.3, 1.0, 1.0),
                 (0.5, 0.0, 0.0),
                 (0.7, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 1.0, 1.0))}

short_rainbow = LinearSegmentedColormap('my_colormap',short_rainbow,256)
long_rainbow = LinearSegmentedColormap('my_colormap',long_rainbow,256)

def histogram(data, figure, quantity, bins, log_y):
    '''Plot an histogram.

    Plots an histogram of the specified quantity.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    figure : matplotlib.figure.Figure
        An (empty) Figure object.
    quantity : str
        Either v (abs v), v_x (x-component) or v_y (y-component).
    bins : int
         Number of bins (bars) in the histogram.
    log_scale : boolean
        Use logaritmic vertical axis.
    '''

    if quantity == 'v':
        xlabel = 'absolute displacement'
        h_data = np.array([(l[2]**2+l[3]**2)**0.5 for l in data])
    elif quantity == 'v_x':
        xlabel = 'x displacement'
        h_data = np.array([l[2] for l in data])
    elif quantity == 'v_y':
        xlabel = 'y displacement'
        h_data = np.array([l[3] for l in data])
    ax = figure.add_subplot(111)
    if log_y:
        ax.set_yscale("log")
    ax.hist(h_data, bins, label=quantity)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('number of vectors')
    ax.set_title(parameter['plot_title'])

    
def profiles(data, parameter, figure, orientation):
    '''Plot velocity profiles.

    Line plots of the velocity component specified.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    fname : str
        A filename containing vector data. 
        (will be deprecated in later updates)
    figure : matplotlib.figure.Figure 
        An (empty) Figure object.
    orientation : str 
        horizontal: Plot v_y over x.
        vertical: Plot v_x over y.
    '''
    for i in list(data.columns.values):
        data[i] = data[i].astype(float)
    data = data.to_numpy().astype(np.float)
    
    dim_x, dim_y = get_dim(data)
    
    p_data = []
    
    ax = figure.add_subplot(111)
    
    if orientation == 'horizontal':
        xlabel = 'x position'
        ylabel = 'y displacement'
        
        for i in range(0, dim_y, parameter['profiles_jump']):
            p_data.append(data[dim_x*i:dim_x*(i+1),3])
        #print(p_data[-1])
        for p in p_data:
            #print(len(p))
            ax.plot(range(dim_x), p, '.-')
            
    elif orientation == 'vertical':
        xlabel = 'y position'
        ylabel = 'x displacement'
        
        for i in range(0, dim_x, parameter['profiles_jump']):
            p_data.append(data[i::dim_x,2])
            
        for p in p_data:
            ax.plot(range(dim_y), p, '.-') 
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(parameter['plot_title'])


def scatter(
    data, 
    figure,
    ax = None,
    mask_coords = [],
    title = None,
    units = ['px', 'dt']
):
    '''Scatter plot.

    Plots v_y over v_x.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    figure : matplotlib.figure.Figure 
        An (empty) Figure object.
    '''     
    if len(mask_coords) > 0:
        mask = coords_to_xymask(data[0], data[1], mask_coords)
    else:
        mask = np.ma.nomask
        
    u = np.ma.masked_array(data[2], mask = mask)
    v = np.ma.masked_array(data[3], mask = mask)   
    
    if ax == None:
        ax = figure.add_subplot(111)
    ax.scatter(u, v, label='scatter',s = 1.5)
    ax.set_xlabel(f'x displacement [{units[0]}/{units[1]}]')
    ax.set_ylabel(f'y displacement [{units[0]}/{units[1]}]')
    if title != None:
        ax.set_title(title)

    
def vector(data,
           figure, 
           axes,
           parameter,
           mask_coords,
           **kw):
    '''Display a vector plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    figure : matplotlib.figure.Figure 
        An (empty) Figure object.
    '''
          
    try:
        invalid = data[4].astype('bool')
    except:
        invalid = np.asarray([True for i in range(len(data[0]))])

    # tilde means invert:
    valid = ~invalid
    ax = axes
    
    x = data[0]
    y = data[1]
    u = data[2]
    v = data[3]
    
    x = x[::parameter['nthArrY'], ::parameter['nthArrX']] 
    y = y[::parameter['nthArrY'], ::parameter['nthArrX']] 
    u = u[::parameter['nthArrY'], ::parameter['nthArrX']]
    v = v[::parameter['nthArrY'], ::parameter['nthArrX']]
    
    if len(mask_coords) > 0:
        mask = coords_to_xymask(x, y, mask_coords)
        u = np.ma.masked_array(u, mask)
        v = np.ma.masked_array(v, mask) 
        mask = mask.astype(bool)
        
        if parameter['show_masked_vectors']:
            ax.plot(
                x.flat[mask],
                y.flat[mask],
                color = parameter['mask_vec'],
                marker = parameter['mask_vec_style'],
                linestyle = '',
                zorder=1,
            )

    invalid = invalid[::parameter['nthArrY'], ::parameter['nthArrX']]
    valid = valid[::parameter['nthArrY'], ::parameter['nthArrX']]
        
    if parameter['uniform_vector_color']:
        ax.quiver(x[invalid],
                  y[invalid],
                  u[invalid],
                  v[invalid],
                  color      = parameter['invalid_color'],
                  label      = 'invalid', 
                  headwidth  = parameter['vec_head_width'],
                  headlength = parameter['vec_head_len'],
                  pivot      = parameter['vec_pivot'],
                  **kw,
                  zorder = 2)

        ax.quiver(x[valid],
                  y[valid],
                  u[valid],
                  v[valid],
                  color      = parameter['valid_color'],
                  label      = 'valid', 
                  headwidth  = parameter['vec_head_width'],
                  headlength = parameter['vec_head_len'],
                  pivot      = parameter['vec_pivot'],
                  zorder     = 3,
                  **kw)
    else:
        
        cmap = get_cmap(parameter['color_map'])
            
        try:
            vmin = float(parameter['vmin'])
        except:
            vmin = None
        try:
            vmax = float(parameter['vmax'])
        except:
            vmax = None
            
        ax.quiver(x,
                  y,
                  u,
                  v,
                  (u**2 + v **2) ** 0.5,
                  cmap       = cmap,
                  clim       = (vmin, vmax), # has no effect? #is brocken?
                  headwidth  = parameter['vec_head_width'],
                  headlength = parameter['vec_head_len'],
                  pivot      = parameter['vec_pivot'],
                  zorder     = 2,
                  **kw)

    
def contour(
    data, 
    parameter,
    figure, 
    axes,
    color_values, 
    mask = np.ma.nomask,
    borders = None,
):
    '''Display a contour plot    

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    parameter : openpivgui.OpenPivParams
        Parameter-object.
    figure : matplotlib.figure.Figure
       An (empty) Figure object.
    '''        
    ax = axes
    x, y = data
    
    if parameter['contours_custom_density'] == False:
        # try to get limits, if not possible set to None
        try:
            vmin = float(parameter['vmin'])
        except:
            vmin = None
        try:
            vmax = float(parameter['vmax'])
        except:
            vmax = None
        # settings for color scheme of the contour plot  
        if vmax is not None and vmin is not None:
            levels = np.linspace(vmin, vmax, int(parameter['color_levels']))
        elif vmax is not None:
            levels = np.linspace(0, vmax, int(parameter['color_levels']))
        elif vmin is not None:
            vmax = color_values.max().max()
            levels = np.linspace(vmin, vmax, int(parameter['color_levels']))
        else:
            levels = int(parameter['color_levels'])
    else:
        levels = parameter['contours_density']
        try:
            vmin = float(parameter['vmin'])
        except:
            vmin = None
        try:
            vmax = float(parameter['vmax'])
        except:
            vmax = None
        
    # Choosing the correct colormap
    colormap = get_cmap(parameter['color_map'])
    color_values = np.ma.masked_array(color_values, mask = mask)
        
    if parameter['contours_type'] == 'filled':
        if parameter['contours_uniform']:
            c = ax.contourf(
                x,
                y,
                color_values,
                levels = levels, 
                cmap = colormap,
                vmin = vmin,
                vmax = vmax,
                extend = 'both',
                alpha = parameter['contours_alpha']
            ) 
        else:
            color_values = np.flipud(color_values)
            c = ax.imshow(
                color_values,
                extent = borders,
                cmap = colormap,
                vmin = vmin,
                vmax = vmax,
                alpha = parameter['contours_alpha'],
                interpolation = "bilinear",
            )
        if parameter['contours_uniform_color']:
            if parameter['contours_uniform'] != True:
                color_values = np.flipud(color_values)
            c2 = ax.contour(
                x,
                y,
                color_values,
                levels = levels, 
                colors = parameter['contour_color'],
                vmin = vmin,
                vmax = vmax,
                extend = 'both',
                linewidths = parameter['contours_thickness'],
                linestyles = parameter['contours_line_style'],
                alpha = parameter['contours_alpha']
            )  
    else:
        if parameter['contours_uniform_color']:
            c2 = ax.contour(
                x,
                y,
                color_values,
                levels = levels, 
                colors = parameter['contour_color'],
                vmin = vmin,
                vmax = vmax,
                extend = 'both',
                linewidths = parameter['contours_thickness'],
                linestyles = parameter['contours_line_style'],
                alpha = parameter['contours_alpha']
            )    
        else:
            c = ax.contour(
                x,
                y,
                color_values,
                levels = levels, 
                cmap = colormap,
                vmin = vmin,
                vmax = vmax,
                extend = 'both',
                linewidths = parameter['contours_thickness'],
                linestyles = parameter['contours_line_style'],
                alpha = parameter['contours_alpha']
            )
     
    
def streamlines(
    data, 
    figure,
    parameter,
    mask = np.ma.nomask,
    
):
    '''Display a streamline plot.    

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    parameter : openpivgui.OpenPivParams
        Parameter object.
    figure : matplotlib.figure.Figure
        An (empty) Figure object.
    '''
    ax = figure
    x = data[0]
    y = data[1]
    u = np.ma.masked_array(data[2], mask=mask)
    v = np.ma.masked_array(data[3], mask=mask) 
        
    # get density for streamline plot.
    try:    
        density = (float(list(parameter['streamlines_density'].split(','))[0]),
            float(list(parameter['streamlines_density'].split(','))[1]))
    except:
        density = float(parameter['streamlines_density'])         

    try:

        fig = ax.streamplot(
            x,
            y,
            u,
            v,
            density    = density,   
            color      = parameter['streamlines_color'],
            integration_direction = parameter['integrate_dir'],
            linewidth  = parameter['streamlines_thickness'],
            arrowstyle = parameter['streamlines_arrow_style'],
            arrowsize  = parameter['streamlines_arrow_width'],
            zorder = 4,
        )
    except:
        print("Failed to display streamlines.")
        
    
def pandas_plot(data, parameter, plot_type, figure):
    '''Display a plot with the pandas plot utility.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    parameter : openpivgui.OpenPivParams
        Parameter-object.
    figure : matplotlib.figure.Figure
        An (empty) figure.

    Returns
    -------
    None.

    '''
    # set boolean for chosen axis scaling
    if parameter['plot_scaling'] == 'None':
        logx, logy, loglog = False, False, False
    elif parameter['plot_scaling'] == 'logx':
        logx, logy, loglog = True, False, False
    elif parameter['plot_scaling'] == 'logy':
        logx, logy, loglog = False, True, False
    elif parameter['plot_scaling'] == 'loglog':
        logx, logy, loglog = False, False, True
    # add subplot    
    ax = figure.add_subplot(111)
    # set limits initially to None
    xlim = None
    ylim = None
    # try to set limits, if not possible (no entry) --> None
    try:
        xlim = (float(list(parameter['plot_xlim'].split(','))[0]),
            float(list(parameter['plot_xlim'].split(','))[1]))
    except:  
        pass
        #print('No Values or wrong syntax for x-axis limitation.')
    try:
        ylim = (float(list(parameter['plot_ylim'].split(','))[0]),
            float(list(parameter['plot_ylim'].split(','))[1]))
    except:
        pass
        #print('No Values or wrong syntax for y-axis limitation.')
    # iteration to set value types to float
    for i in list(data.columns.values):
        data[i] = data[i].astype(float)
        
    if plot_type == 'histogram':
        # get column names as a list for comparing with chosen histogram
        # quantity
        col_names = list(data.columns.values)
        # if loop for histogram quantity
        if parameter['histogram_quantity'] == 'v_x':
            data_hist = data[col_names[2]]
        elif parameter['histogram_quantity'] == 'v_y':
            data_hist = data[col_names[3]]
        elif parameter['histogram_quantity'] == 'v':
            data_hist = (data[col_names[2]]**2 + data[col_names[3]]**2)**0.5
        # histogram plot
        ax.hist(data_hist,
                bins = int(parameter['histogram_bins']),
                label = parameter['histogram_quantity'],
                log = logy,
                range = xlim,
                histtype = parameter['histogram_type'],
                )
        ax.grid(parameter['plot_grid'])
        ax.legend()
        ax.set_xlabel('velocity [m/s]')
        ax.set_ylabel('number of vectors')
        #ax.set_title(parameter['plot_title'])
    else:
        data.plot(x = parameter['u_data'], 
              y = parameter['v_data'], 
              kind = parameter['plot_type'], 
              #title = parameter['plot_title'], 
              grid = parameter['plot_grid'], 
              legend = parameter['plot_legend'],
              logx = logx, 
              logy = logy , 
              loglog = loglog, 
              xlim = xlim,
              ylim = ylim,
              ax = ax)

def hist2(
    data, 
    parameter, 
    figure,
    units,
    mask_coords = [],
    title = None
):
    '''Display a plot with the pandas plot utility.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    parameter : openpivgui.OpenPivParams
        Parameter-object.
    figure : matplotlib.figure.Figure
        An (empty) figure.

    Returns
    -------
    None.

    '''
    # set boolean for chosen axis scaling
    if parameter['plot_scaling'] == 'None':
        log = False
    else:
        log = True
    # add subplot    
    ax = figure.add_subplot(111)

    if len(mask_coords) > 0:
        mask = coords_to_xymask(data[0], data[1], mask_coords)
    else:
        mask = np.ma.nomask
        
    if parameter['histogram_quantity'] == 'u-component':
        data_hist = data[2]
    elif parameter['histogram_quantity'] == 'v-component':
        data_hist = data[3]
    elif parameter['histogram_quantity'] == 'magnitude':
        data_hist = np.hypot(data[2], data[3])
    
    data_hist = np.ma.masked_array(data = data_hist, mask = mask)
    
    data_hist = np.vstack(data_hist.ravel())
               
    # histogram plot
    ax.hist(data_hist,
            bins = int(parameter['histogram_bins']),
            label = parameter['histogram_quantity'],
            log = log,
            histtype = parameter['histogram_type'],
            )
    ax.grid(parameter['plot_grid'])
    ax.legend()
    ax.set_xlabel(f'velocity [{units[0]}/{units[1]}]')
    ax.set_ylabel('number of vectors')
    if title != None:
        ax.set_title(title)

    
def get_dim(array):
    '''Computes dimension of vector data.

    Assumes data to be organised as follows (example):
    x  y  v_x v_y ..
    16 16 4.5 3.2 ..
    32 16 4.3 3.1 ..
    16 32 4.2 3.5 ..
    32 32 4.5 3.2 ..
    .. .. ..  ..

    Parameters
    ----------
    array : np.array
        Flat numpy array.

    Returns
    -------
    tuple
        Dimension of the vector field (x, y).
    '''
    return(len(set(array[:, 0])),
           len(set(array[:, 1])))


def get_cmap(cmap):
    if cmap == 'short rainbow':
        colormap = short_rainbow
    elif cmap == 'long rainbow':
        colormap = long_rainbow
    else:
        colormap = cmap
    return colormap


def get_component(x, y, u, v, component = 'magnitude'):
    if component   == 'u-component':
        component = u
    elif component == 'v-component':
        component = v
    elif component == 'magnitude':
        component = (u**2 + v**2)**0.5
    elif component == 'vorticity': 
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        vort = vx - uy
        component = -vort.T
    elif component ==  'enstrophy': 
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        vort = vx - uy
        component = vort.T ** 2
    elif component == 'divergence': 
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        component = (vy + ux).T
    elif component == 'shear strain':
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        strain = vx + uy
        component = -strain.T
    elif component == 'normal strain':
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        strain = ux + vy
        component = strain.T
    elif component == 'acceleration':
        vx, vy = np.gradient(v.T, x[0, :], y[:, 0])
        ux, uy = np.gradient(u.T, x[0, :], y[:, 0])
        component = np.sqrt( # hopefully, this is correct...
            np.power(((u.T * ux) + (v.T * uy)), 2) +
            np.power(((u.T * vx) + (v.T * vy)), 2)
        ).T
    elif component == 'kinetic energy':
        component = u**2 + v**2
    elif component == 'gradient dv/dx':
        component, _ = np.gradient(v.T, x[0, :], y[:, 0])
        component = component.T
    elif component == 'gradient dv/dy':
        _, component = np.gradient(v.T, x[0, :], y[:, 0])
        component = component.T
    elif component == 'gradient du/dx':
        component, _ = np.gradient(u.T, x[0, :], y[:, 0])
        component = component.T
    elif component == 'gradient du/dy':
        _, component = np.gradient(u.T, x[0, :], y[:, 0])
        component = component.T
    else:
        print('Component not supported.')
        component = np.hypot(u, v)
    return component
        
    
def plot_colorbar(
    fig, 
    component,
    cbaxis = None,
    cmap = 'viridis',
    vmin = None,
    vmax = None,
):
    if cbaxis == None:
        n = [0.05, 0.05, 0.9, 0.05]
        #n = [0,0,1,0.05]
        cbaxis = fig.add_axes(n)
    if vmin == None:
        vmin = np.nanmin(component)
    if vmax == None:
        vmax = np.nanmax(component)
    c = plt.cm.ScalarMappable(
        cmap = cmap,
        norm = plt.Normalize(
            vmin = vmin,
            vmax = vmax
        )
    )
    cb = fig.colorbar(
        c,
        cax = cbaxis,
        orientation = 'horizontal',
        #extendrect = True
        #pad = .05
    )
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
'''
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Plot vector data.')
    parser.add_argument('--plot_type',
                        type=str,
                        required=False,
                        choices=['histogram',
                                 'profiles',
                                 'vector',
                                 'scatter',
                                 'contour'
                                 'contour_and_vector',
                                 'streamlines'],
                        default='vector',
                        help='type of plot')
    parser.add_argument('--fname',
                        required=True,
                        type=str,
                        help='name of vector data file')
    parser.add_argument('--quantity',
                        type=str,
                        required=False,
                        choices=['v', 'v_x', 'v_y'],
                        default='v',
                        help='quantity to plot')
    parser.add_argument('--bins',
                        type=int,
                        required=False,
                        default=20,
                        help='number of histogram bins')
    parser.add_argument('--log_y',
                        type=bool,
                        required=False,
                        default=False,
                        help='logarithmic y-axis')
    parser.add_argument('--orientation',
                        type=str,
                        required=False,
                        choices=['horizontal', 'vertical'],
                        default='vertical',
                        help='plot profiles, either horizontal ' +
                             '(v_y over x) or vertical (v_x over y)')
    parser.add_argument('--invert_yaxis',
                        type=str,
                        required=False,
                        default=True,
                        help='Invert y-axis of vector plot')
    args = parser.parse_args()
    data = np.loadtxt(args.fname)
    fig = Figure()
    if args.plot_type=='histogram':
        histogram(data,
                  fig,
                  quantity=args.quantity,
                  bins=args.bins,
                  log_y=args.log_y)
    elif args.plot_type=='profiles':
        profiles(data,
                 fig,
                 orientation=args.orientation)
    elif args.plot_type=='vector':
        vector(data,
               fig,
               invert_yaxis=args.invert_yaxis)
    elif args.plot_type=='scatter':
        scatter(data,
                fig)
    elif args.plot_type=='contour':
        print('Not yet implemented')
    elif args.plot_type=='contour_and_vector':
         print('Not yet implemented')
    elif args.plot_type=='streamlines':
         print('Not yet implemented')
    plt.show()
'''