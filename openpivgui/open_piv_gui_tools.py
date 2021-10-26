#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Methods for reuse within the OpenPivGui project.'''
from skimage.measure import points_in_poly, profile_line
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

import numpy as np
import math
import os
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

__email__ = 'vennemann@fh-muenster.de'


def str2list(s):
    '''Parses a string representation of a list.

    Parameters
    ----------
    s : str
        String containing comma separated values.

    Example
    -------
    str2list('img01.png', 'img02.png')

    Returns
    -------
    list
    '''
    return([t.strip("' ") for t in s.strip('(),').split(',')])


def str2dict(s):
    '''Parses a string representation of a dictionary.

    Parameters
    ----------
    s : str
        Comma separated list of colon separated key value pairs.

    Example
    -------
    str2dict('key1: value1', 'key2: value2')
    '''
    d = {}
    l = str2list(s)
    for elem in l:
        key, value = elem.split(':')
        key.strip(' ')
        value.strip(' ')
        d.update({key: value})
    return(d)


def create_save_vec_fname(path=os.getcwd(),
                          basename=None,
                          postfix='',
                          count=-1,
                          max_count=9):
    '''Assembles a valid absolute path for saving vector data.

    Parameters
    ----------
    path : str
        Directory path. Default: Working directory.
    basename : str
        Prefix. Default: None.
    postfix : str
        Postfix. Default: None.
    count : int
        Counter for numbering filenames. 
        Default: -1 (no number)
    max_count : int
        Highest number to expect. Used for generating 
        leading zeros. Default: 9 (no leading zeros).
    '''
    if count == -1:
        num = ''
    else:
        num = str(count).zfill(math.ceil(math.log10(max_count)))
    if basename is None:
        basename = os.path.basename(path)
    elif basename == '':
        basename = 'out'
    return(os.path.dirname(path) +
           os.sep +
           basename.split('.')[0] +
           postfix +
           num +
           '.vec')


def get_dim(array):
    '''Computes dimension of vector data.

    Assumes data to be organised as follows (example):
    x  y  v_x v_y
    16 16 4.5 3.2
    32 16 4.3 3.1
    16 32 4.2 3.5
    32 32 4.5 3.2

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



def save(components, filename, fmt='%8.4f', delimiter='\t', header = ''):
    if isinstance(components[2], np.ma.MaskedArray):
        components[2] = components[2].filled(0.)
        components[3] = components[3].filled(0.)
    out = np.vstack([m.ravel() for m in components])
    np.savetxt(filename, out.T, fmt=fmt, delimiter=delimiter, header = header)

    
def _round(number, decimals = 0):
    multiplier = 10 **decimals
    return(math.floor(number * multiplier + 0.5) / multiplier) 


def coords_to_xymask(x, y, mask_coords):
    mask = []
    for masks in mask_coords:
        mask.append(np.flip(np.flipud(masks)))
            
    masks = []
    for i in range(len(mask)):
        masks.append(
            points_in_poly(
                np.c_[
                    y.flatten(),
                    x.flatten()
                ],
                mask[i]
            ).astype(np.int)
        )
    xymask = masks[0]
    for i in range(1, len(masks)):
        xymask += masks[i]
        xymask[xymask > 1] = 1
    return(xymask.astype(np.int))


def add_disp_roi(axes,
                 xmin, ymin, xmax, ymax, 
                 linewidth,
                 edgecolor,
                 linestyle,
                 padleft = 1, padright = 1):
    axes.add_patch(patches.Rectangle((xmin - padleft,
                                     ymin - padleft), 
                                     xmax - xmin + padright,
                                     ymax - ymin + padright,
                                     linewidth = linewidth,
                                     edgecolor = edgecolor,
                                     facecolor = 'none',
                                     linestyle = linestyle))
        
def add_disp_mask(axes, coords, color, alpha, invert = False):
    if len(coords) > 0:
        shapes = []
        for i in range(len(coords)):
            shapes.append(patches.Polygon(coords[i]))
            
        disp_mask = PatchCollection(shapes,
                               color = color, 
                               alpha = alpha)
        if invert:
            ~disp_mask
        axes.add_collection(disp_mask) 
        
        
def normalize_array(array, axis = None):
    array = array.astype(np.float32)
    if axis == None:
        return((array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array)))
    else:
        return((array - np.nanmin(array, axis = axis)) / 
               (np.nanmax(array, axis = axis) - np.nanmin(array, axis = axis)))


def standardize_array(array, axis = None):
    array = array.astype(np.float32)
    if axis == None:
        return((array - np.nanmean(array) / np.nanstd(array)))  
    else:
        return((array - np.nanmean(array, axis = axis) / np.nanstd(array, axis = axis)))
    
    
def get_selection(selection, max_):
    select_frames = []
    if ',' in selection:
        for i in range(len(selection.split(','))):
            try:
                if selection.split(',')[i].count(':') == 2:
                    start = int(list(selection.split(','))[i].split(':')[0])
                    end = int(list(selection.split(','))[i].split(':')[1])
                    step = int(list(selection.split(','))[i].split(':')[2])
                    if end == -1:
                        end = max_
                    select_frames += list(range(start, end, step))
                elif ':' in selection.split(',')[i]:
                    start = int(list(selection.split(','))[i].split(':')[0])
                    end = int(list(selection.split(','))[i].split(':')[1])
                    if end == -1:
                        end = max_
                    select_frames += list(range(start, end))
                else:

                    select_frames.append(
                        int(list(selection.split(','))[i])
                    )
            except:
                print('Ignoring {}. \nReason: failure to decode or out of index.'.format(
                    str(list(select_frames.split(','))[i]))
                )
    else: # bug in here somewhere
        try:
            if selection.count(':') == 2:
                start = int((selection.split(':')[0]))
                end = int((selection.split(':')[1]))
                step = int((selection.split(':')[2]))
                if end == -1:
                    end = max_
                select_frames += list(range(start, end, step))
            elif ':' in selection:
                start = int((selection.split(':')[0]))
                end = int((selection.split(':')[1]))
                if end == -1:
                    end = max_
                select_frames += list(range(start, end))
            else:
                select_frames.append(
                    int(selection)
                )
        except:
            print('Ignoring {} (sf). \nReason: failure to decode or out of index.'.format(
                str(list(select_frames))
            ))
    return select_frames
    
def line2profile(
    array,
    x_coords,
    y_coords,
    order = 1
):
    profile = profile_line(
        array,
        (y_coords[0], x_coords[0]),
        (y_coords[1], x_coords[1]),
        order = order,
    )
    return profile