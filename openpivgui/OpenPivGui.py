#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''A simple GUI for OpenPIV.'''

import openpivgui.vec_plot as vec_plot
from openpivgui.open_piv_gui_tools import (str2list, str2dict, get_dim, _round,
    add_disp_roi, add_disp_mask, save, get_selection, coords_to_xymask)
from openpivgui.ErrorChecker import check_PIVprocessing, check_processing
from openpivgui.AddIns import image_transformations, image_phase_separation, image_temporal_filters, image_spatial_filters
import openpivgui.AddInHandler as AddInHandler
from openpivgui.MultiProcessing import MultiProcessing
from openpivgui.CreateToolTip import CreateToolTip
from openpivgui.OpenPivParams import OpenPivParams
from openpivgui.widget_lists import widget_list, button_list, disabled_widgets
from openpiv.preprocess import mask_coordinates
from openpiv.pyprocess import get_rect_coordinates
from matplotlib.figure import Figure as Fig
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk)
from matplotlib.widgets import RectangleSelector, PolygonSelector, LassoSelector, Cursor
from multiprocessing import Manager
from pkg_resources import get_distribution
from functools import reduce
from operator import or_
from scipy.signal import get_window
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.path as Path
import openpiv.tools as piv_tls
import pandas as pd
import h5py
import pickle
import time
import numpy as np
from PIL import ImageDraw, Image, ImageTk
from skimage import measure
from tkinter import colorchooser
from datetime import datetime
import threading
import shutil
import webbrowser
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import tkinter as tk
import inspect
import json
import math
import sys
import re
import os

__version__ = '0.0.2'

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


class OpenPivGui(tk.Tk):
    '''OpenPIV GUI

    Usage:

    1. Press »File« then »load images«. 
       Select some image pairs (Ctrl + Shift for multiple).

    2. Click on the links in the image list to view the imported 
       images and press »Apply frequencing« to load the images
       into the GUI.

    3. Walk through the drop-down-menues in »Pre-processing«
       and »Analysis« and edit the parameters.

    4. Calibrate your images or results with the »Calibration« 
       drop-down-menu.
       
    5. Press the »Analyze all frame(s)« butten to 
       start the processing chain. Analyzing the current frame 
       saves the correlation matix for further analysis.
    
    6. Validate/modify your results with the »Post processing« 
       drop-down-menu.
    
    7. Inspect the results by clicking on the links in the frame-list
       on the right.
       Use the »Data exploration« drop-down menu for changing
       the plot parameters.

    8. Re-evaluate images if needed (results are automatically
       deleted) with new information/parameters.

    See also:

    https://github.com/OpenPIV/openpiv_tk_gui
   '''

    preprocessing_methods = {}
    postprocessing_methods = {}
    plotting_methods = {}
    toggled_widgets = {}
    toggled_buttons = []
    
    def __init__(self):
        '''Standard initialization method.'''
        print('Initializing GUI')
        self.VERSION = __version__
        self.openpivVersion = get_distribution('openpiv')
        self.TITLE = f'Using {self.openpivVersion}, GUI version'
        tk.Tk.__init__(self)
        self.path = os.path.dirname(
            os.path.abspath(__file__))  # path of gui folder
        self.icon_path = os.path.join(
            self.path, 'res/icon.png')  # path for image or icon
        # convert .png into a usable icon photo
        self.iconphoto(False, tk.PhotoImage(file=self.icon_path))
        self.title(self.TITLE + ' ' + self.VERSION)
        # handle for user closing GUI through window manager
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        # the parameter object
        self.p = OpenPivParams()
        AddInHandler.init_add_ins(self)
        self.toggled_widgets.update(widget_list())
        self.toggled_buttons += button_list()
        self.p.load_settings(self.p.params_fname)
        # background variable for widget data:
        self.tkvars = {}
        # handle for settings frames on riders
        self.set_frame = []
        # handle for ttk widgets that have state changes
        self.ttk_widgets = {};
        self.rasterize = False # what's this?
        # handle for text-area objects
        self.ta = []
        # handle for list-boxes
        self.lb = None
        # handle for masks
        self.mask_counter = 0
        self.object_mask = []
        # handle for frame widths
        self.frame_width = 250
        # widgets and others
        print('Initializing widgets')
        # handle for background color
        self.b_color = self.cget('bg')
        self.toggle = 'a'
        self.index = 0
        # padding settings
        self.padx = 3
        self.pady = 2
        self.sub_padx = 3
        self.sub_pady = 2
        # button widths
        self.large_button_width = 25
        self.small_button_width = 11
        # progressbar width
        self.progressbar_width = 220
        style = ttk.Style()
        style.configure('h12.TButton', font = ('Helvetica', 12))
        self.bold_apply = 'h12.TButton'
        # vector scaling
        self.scale_dist = 1
        self.scale_vel  = 1
        self.units = ['px', 'dt']
        self.xlim = [None, None]
        self.ylim = [None, None]
        self.img_shape = [None, None]
        # init widgets
        self.__init_widgets()
        self.set_settings()
        self.load_session(default = True)
        self.update_widget_state()
        self.overlap_percent = [0.5, 0.5]
        self.set_windowing(0)
        # other
        self.stats_frame = None
        self.scatter_frame = None
        self.hist_frame = None
        self.corr_frame = None
        self.background_frame_a = []
        self.background_frame_b = []
        try:
            self.show(self.p['files_a'][0], preview = False, bypass = True)
        except: pass
        for widget in disabled_widgets():
            self.ttk_widgets[widget].config(state = 'disabled')
        self.log(timestamp=True, text='--------------------------------' +
                                      '\nTkinter OpenPIV session started.')
        self.log(text='OpenPivGui version: ' + self.VERSION)
        print('Initialized GUI, ready for processing')
        
    
    def get_parameters(self):
        return self.p
    
    
    def start_processing(self, frame = None):
        '''Wrapper function to start processing in a separate thread.'''
        start_processing = True
        if frame == None:
            start_processing = messagebox.askyesno(
                title = 'Batch Processing Manager',
                message = ('Do you want to perform batch processing?\n'+
                    'You may not be able to stop once the analysis starts.')
            )
        if start_processing:
            try:
                self.get_settings()
                #check_processing(self)  # simple error checking. # this doesn't work for HDF5 multiprocessing
                message = 'Please stop all threads/processes to start processing.'
                checker = 0
                # check if any threads are alive
                try:
                    if self.processing_thread.is_alive():
                        if self.p['warnings']:
                            messagebox.showwarning(title='Error Message',
                                                   message=message)
                        checker += 1
                except: pass

                try:
                    if self.postprocessing_thread.is_alive():
                        if self.p['warnings']:
                            messagebox.showwarning(title='Error Message',
                                                   message=message)
                        checker += 1
                except: pass
                # if a thread is alive, an error shall be raised
                if checker != 0:
                    # raising errors did not work in try statement for some reason
                    raise Exception(message)
                check_PIVprocessing(
                    self.p, 
                    self.session['images']['settings']['frame_0'].attrs['roi_coords']
                )
                self.processing_thread = threading.Thread(target=lambda: self.processing(select_frame = frame))
                self.processing_thread.start()
            except Exception as e:
                print('PIV evaluation thread stopped. ' + str(e))
            

            
    def processing(self, select_frame = None):
        try: # if an error occurs, GUI will not lock up
            self.disable_widgets(exclude_tab = 5)
            self.ttk_widgets['clear_results'].config(state = 'normal',
                                                    text = 'Stop analysis',
                                                    command = self.stop_analysis)
            self.p['analysis'] = True
            self.log(timestamp=True,
                     text='-----------------------------' +
                     '\nStarting evaluation with current image settings.',
                     group=self.p.PREPROC)
            '''Start the processing chain.
            This is the place to implement additional function calls.
            '''
            # parallel PIV evaluation:
            print('Starting evaluation.')
            self.progressbar.start()
            self.get_settings()
            # keep number of cores in check
            if os.cpu_count() == 0:  # if there are no cored available, then raise exception
                raise Exception('Warning: no available threads to process in.')

            if self.p['manual_select_cores']:  # multiprocessing disabled until results are  properly eppended to dict.
                cpu_count = self.p['cores']
            else:
                cpu_count = os.cpu_count() - 1

            if "idlelib" in sys.modules:
                self.log('Running as a child of IDLE: ' +
                         'Deactivated multiprocessing.'
                )
                cpu_count = 1

            if cpu_count > os.cpu_count():
                raise Exception('Please lower the amount of cores ' +
                                'or deselect >manually select cores<.'
                )
            
            print('Cores utilized: {} of {}.'.format(
                (cpu_count), os.cpu_count())
            )
            
            if select_frame != None:
                i = select_frame
                self.process_type.config(text = f'Processing frame {i}')
                img_settings = {}
                mask = list(literal_eval(self.session['images']['settings'][f'frame_{i}'].attrs['mask_coords']))
                img_settings[f'{i}'] = [
                    self.session['images']['settings'][f'frame_{i}'].attrs['roi_coords'],
                    mask
                ]

                mp = MultiProcessing(
                    self,
                    settings = img_settings,
                    #session = self.session, # can't be pickled
                    bg_a = self.background_frame_a,
                    bg_b = self.background_frame_b,
                    files_a = [self.p['files_a'][self.index]],
                    files_b = [self.p['files_b'][self.index]],
                )
                results = mp.process(args = (self.p['files_a'][i],
                                   self.p['files_b'][i],
                                   i)
                )

                print(f'Storing frame {i}')
                frame = self.session['results'][f'frame_{i}']
                frame.attrs['processed']    = results['processed']
                frame.attrs['roi_present']  = results['roi_present']
                frame.attrs['roi_coords']   = results['roi_coords']
                frame.attrs['mask_coords']  = str(results['mask_coords'])
                frame.attrs['process_time'] = results['process_time']

                frame.attrs['offset_x']   = 0
                frame.attrs['offset_y']   = 0
                try:
                    test = frame.attrs['scale_dist']
                except:
                    frame.attrs['scale_dist'] = 1
                    frame.attrs['scale_vel']  = 1
                    frame.attrs['units']      = ['px', 'dt']

                if 'x' in frame: del frame['x']
                if 'y' in frame: del frame['y']
                if 'u' in frame: del frame['u']
                if 'v' in frame: del frame['v']
                if 'tp' in frame: del frame['tp']
                if 's2n' in frame: del frame['s2n']
                if 'corr' in frame: del frame['corr']

                frame.create_dataset('x', data = results['x'])
                frame.create_dataset('y', data = results['y'])
                frame.create_dataset('u', data = results['u'])
                frame.create_dataset('v', data = results['v'])
                frame.create_dataset('tp', data = results['tp'])
                frame.create_dataset('s2n', data = results['s2n'])
                frame.create_dataset('corr', data = results['corr'])

                if 'u_vld' in frame:
                    del frame['u_vld']
                    del frame['v_vld']
                    del frame['tp_vld']

                if 'u_mod' in frame:
                    del frame['u_mod']
                    del frame['v_mod']

                text = f'Processed frame {self.index}'

            else: # batch processing
                if self.p['ensemble_correlation'] != True:
                    self.process_type.config(text = ('Processing {} frames(s)'.format(
                        len(self.p['files_a']))))

                    img_settings = {}
                    for j in range(len(self.p['files_a'])):
                        mask = list(literal_eval(self.session['images']['settings'][f'frame_{j}'].attrs['mask_coords']))
                        img_settings[f'{j}'] = [
                            list(self.session['images']['settings'][f'frame_{j}'].attrs['roi_coords']),
                            mask
                        ]
                    print('Created settings dictionary for all frames')
                    self.ttk_widgets['clear_results'].config(state = 'disabled')
                    mp = MultiProcessing(
                        self, 
                        files_a = self.p['files_a'],
                        files_b = self.p['files_b'],
                        bg_a = self.background_frame_a,
                        bg_b = self.background_frame_b,
                        settings = img_settings,
                        #session = self.session,
                        parallel = True
                    )

                    start = time.time()
                    mp.run(func = mp.process, n_cpus = cpu_count)
                    print(f'Finished processing ({time.time() - start} s)')
                    self.ttk_widgets['clear_results'].config(state = 'normal')
                    for i in range(len(self.p['files_a'])):
                        if self.p['analysis'] == False:
                            break;
                        results = np.load(
                            os.path.join(
                                self.path,
                                f'tmp/frame_{i}.npy'
                            ),
                            allow_pickle = True
                        )
                        print(f'Storing frame {i}')
                        frame = self.session['results'][f'frame_{i}']
                        frame.attrs['processed']    = results.item().get('processed')
                        frame.attrs['roi_present']  = results.item().get('roi_present')
                        frame.attrs['roi_coords']   = results.item().get('roi_coords')
                        frame.attrs['mask_coords']  = str(results.item().get('mask_coords'))
                        frame.attrs['process_time'] = results.item().get('process_time')
                        frame.attrs['offset_x']   = 0
                        frame.attrs['offset_y']   = 0
                        
                        try:
                            test = frame.attrs['scale_dist']
                        except: # no calibration has been applied
                            frame.attrs['scale_dist'] = 1
                            frame.attrs['scale_vel']  = 1
                            frame.attrs['units']      = ['px', 'dt']


                        if 'x' in frame: del frame['x']
                        if 'y' in frame: del frame['y']
                        if 'u' in frame: del frame['u']
                        if 'v' in frame: del frame['v']
                        if 'tp' in frame: del frame['tp']
                        if 's2n' in frame: del frame['s2n']
                        if 'corr' in frame: del frame['corr']

                        frame.create_dataset('x', data = results.item().get('x'))
                        frame.create_dataset('y', data = results.item().get('y'))
                        frame.create_dataset('u', data = results.item().get('u'))
                        frame.create_dataset('v', data = results.item().get('v'))
                        frame.create_dataset('tp', data = results.item().get('tp'))
                        frame.create_dataset('s2n', data = results.item().get('s2n'))

                        if 'u_vld' in frame:
                            del frame['u_vld']
                            del frame['v_vld']
                            del frame['tp_vld']
                        if 'u_mod' in frame:
                            del frame['u_mod']
                            del frame['v_mod'] 
                        print(f'Finished storing frame {i}')
                        os.remove(
                            os.path.join(
                                self.path,
                                f'tmp/frame_{i}.npy'
                            )
                        )
                        print(f'Deleted temporary file for frame {i}')
                    text = 'Processed '+ str(len(self.p['files_a'])) + ' frames(s)'
                else:
                    self.process_type.config(text = ('Processing {} frames(s)'.format(
                        len(self.p['files_a'])))
                    )

                    img_settings = {}
                    for j in range(len(self.p['files_a'])):
                        mask = list(literal_eval(self.session['images']['settings'][f'frame_{j}'].attrs['mask_coords']))
                        img_settings[f'{j}'] = [
                            list(self.session['images']['settings'][f'frame_{j}'].attrs['roi_coords']),
                            mask
                        ]
                    print('Created settings dictionary for all frames')
                    mp = MultiProcessing(
                        self, 
                        files_a = self.p['files_a'],
                        files_b = self.p['files_b'],
                        bg_a = self.background_frame_a,
                        bg_b = self.background_frame_b,
                        settings = img_settings,
                        parallel = True,
                    )

                    results = mp.ensemble_solution()
                    print('Storing ensemble results')
                    
                    if 'ensemble' in self.session['results']:
                        del self.session['results']['ensemble']
                        
                    frame = self.session['results'].create_group('ensemble')
                    frame.attrs['processed']    = results['processed']
                    frame.attrs['roi_present']  = results['roi_present']
                    frame.attrs['roi_coords']   = results['roi_coords']
                    frame.attrs['mask_coords']  = str(results['mask_coords'])
                    frame.attrs['process_time'] = results['process_time']

                    frame.attrs['offset_x']   = 0
                    frame.attrs['offset_y']   = 0

                    try:
                        test = frame.attrs['scale_dist']
                    except:
                        frame.attrs['scale_dist'] = 1
                        frame.attrs['scale_vel']  = 1
                        frame.attrs['units']      = ['px', 'dt']

                    if 'x' in frame: del frame['x']
                    if 'y' in frame: del frame['y']
                    if 'u' in frame: del frame['u']
                    if 'v' in frame: del frame['v']
                    if 'tp' in frame: del frame['tp']
                    if 's2n' in frame: del frame['s2n']
                    if 'corr' in frame: del frame['corr']

                    frame.create_dataset('x', data = results['x'])
                    frame.create_dataset('y', data = results['y'])
                    frame.create_dataset('u', data = results['u'])
                    frame.create_dataset('v', data = results['v'])
                    frame.create_dataset('tp', data = results['tp'])
                    frame.create_dataset('s2n', data = results['s2n'])

                    if 'u_vld' in frame:
                        del frame['u_vld']
                        del frame['v_vld']
                        del frame['tp_vld']

                    if 'u_mod' in frame:
                        del frame['u_mod']
                        del frame['v_mod']
                        
                    if 'Ensemble' not in self.p['fnames']:
                        self.p['fnames'].append('Ensemble')
                        self.tkvars['fnames'].set(self.p['fnames'])
                        del self.session['images']['frames']
                        self.session['images'].create_dataset('frames', data = self.p['fnames'])

                    text = 'Processed '+ str(len(self.p['files_a'])) + ' frames(s)'
                    
            # log everything
            self.log(timestamp=True,
                     text='\nPIV evaluation finished.',
                     group=self.p.PIVPROC)

            self.progressbar.stop()
            self.enable_widgets()
            self.ttk_widgets['clear_results']['command'] = self.clear_results
            self.ttk_widgets['clear_results'].config(text = 'Clear all results')
            self.process_type.config(text = text)
           #self.show(self.p['files_' + self.toggle][self.index]) # this will break the gui for some reason...
            
        except Exception as e:
            print('PIV evaluation thread stopped.\nReason: ' + str(e))
            self.progressbar.stop()
            self.enable_widgets()
            self.ttk_widgets['clear_results']['command'] = self.clear_results
            self.ttk_widgets['clear_results'].config(text = 'Clear all results')
            self.process_type.config(text = 'Failed to process frame(s)')
                

    def start_validations(self, index = None, clear = False):
        '''Wrapper function to start post-processing in a separate thread.'''
        start_processing = True
        if index == None:
            start_processing = messagebox.askyesno(
                title = 'Batch Processing Manager',
                message = ('Do you want to perform batch processing?\n'+
                    'You will not be able to stop once the postprocessing starts.')
            )
        if start_processing:
            try:
                check_processing(self)
                if clear:
                    self.validation_thread = threading.Thread(
                        target = self.clear_validations
                    )
                else:
                    self.validation_thread = threading.Thread(
                        target = lambda: self.validate_results(index = index)
                    )
                self.validation_thread.start()
            except Exception as e:
                print('Stopping current processing thread \nReason: ' + str(e))
        
            
    def validate_results(self, index):
        self.get_settings()
        try:
            self.disable_widgets()
            
            if index != None: frame_len = 1
            else: 
                n = 0
                if 'average' in self.session['results']:
                    n += 1
                if 'ensemble' in self.session['results']:
                    n += 1
                frame_len = len(self.p['fnames']) - n
            
            for i in range(frame_len):
                if index != None: i = index
                print(f'Validating frame {i}')
                self.process_type.config(text = f'Validating frame {i}')
                frame = self.session['results'][f'frame_{i}']
                mask_coords  = list(literal_eval(frame.attrs['mask_coords']))
                
                x = np.array(frame['x']) # incase if there is a different roi somewhere, it won't cause bad errors
                y = np.array(frame['y'])
                u = np.array(frame['u'])
                v = np.array(frame['v'])
                
                if len(mask_coords) > 0 and self.p['validation_exlude_mask'] == True:
                    mask = coords_to_xymask(x, y, mask_coords)
                else: 
                    mask = np.ma.nomask               
 
                flag = np.array(frame['tp']) # maybe rename this to flag?                
                u, v, flag, isSame = self.postprocessing_methods['validate_results'](
                    u, v, mask, flag, 
                    None, # no s2n validation
                    global_thresh       = self.p['vld_global_thr'],
                    global_minU         = self.p['MinU'],
                    global_maxU         = self.p['MaxU'],
                    global_minV         = self.p['MinV'],
                    global_maxV         = self.p['MaxV'],
                    global_std          = self.p['vld_global_std'],
                    global_std_thresh   = self.p['global_std_threshold'],
                    #z_score             = self.p['zscore'],
                    #z_score_thresh      = self.p['zscore_threshold'],
                    local_median        = self.p['vld_local_med'],
                    local_median_thresh = self.p['local_median_threshold'],
                    local_median_kernel = self.p['local_median_size'],
                    replace             = self.p['repl'],
                    replace_method      = self.p['repl_method'],
                    replace_inter       = self.p['repl_iter'],
                    replace_kernel      = self.p['repl_kernel'],
                )

                if 'u_vld' in frame:
                    del frame['u_vld']
                    del frame['v_vld']
                    del frame['tp_vld']
                    
                if isSame != True:
                    frame.create_dataset('u_vld', data = u)
                    frame.create_dataset('v_vld', data = v)
                    frame.create_dataset('tp_vld', data = flag)
                print(f'Validated frame {i}')
                self.process_type.config(text = f'Validated frame {i}')
        except Exception as e:
            print('Could not finish validating results\nReason: ' + str(e))
        self.enable_widgets()
     

    def start_modifications(self, index = None):
        '''Wrapper function to start post-processing in a separate thread.'''
        start_processing = True
        if index == None:
            start_processing = messagebox.askyesno(
                title = 'Batch Processing Manager',
                message = ('Do you want to perform batch processing?\n'+
                    'You will not be able to stop once the postprocessing starts.')
            )
        if start_processing:
            try:
                check_processing(self)
                self.modify_thread = threading.Thread(
                    target = lambda: self.modify_results(index = index)
                )
                self.modify_thread.start()
            except Exception as e:
                print('Stopping current processing thread \nReason: ' + str(e))
        
            
    def modify_results(self, index):
        self.get_settings()
        try:
            self.disable_widgets()
            if index != None:
                frame_len = 1
            else:
                n = 0
                if 'average' in self.session['results']:
                    n += 1
                if 'ensemble' in self.session['results']:
                    n += 1
                frame_len = len(self.p['fnames']) - n
            
            for i in range(frame_len):
                remove = 0
                if index != None: i = index
                print(f'Modifying frame {i}')
                self.process_type.config(text = f'Modifying frame {i}')
                frame = self.session['results'][f'frame_{i}']
                mask_coords  = list(literal_eval(frame.attrs['mask_coords']))
                
                x = np.array(frame['x']) # incase if there is a different roi somewhere, it won't cause bad errors
                y = np.array(frame['y'])
                
                if len(mask_coords) > 0 and self.p['modification_exlude_mask'] == True:
                    mask = coords_to_xymask(x, y, mask_coords)
                else:
                    mask = np.ma.nomask  
                    
                if 'u_vld' in frame:
                    u = np.array(frame['u_vld'])
                    v = np.array(frame['v_vld'])
                else:
                    u = np.array(frame['u'])
                    v = np.array(frame['v'])
                
                _, _, u, v, isSame = self.postprocessing_methods['modify_results'](
                    x, y, u, v, mask,
                    modify_velocity = self.p['modify_velocity'],
                    u_component     = self.p['modify_u'],
                    v_component     = self.p['modify_v'],
                    smooth          = self.p['smoothn'],
                    strength        = self.p['smoothn_val'],
                    robust          = self.p['robust']
                )
                
                if self.p['offset_grid']:
                    frame.attrs['offset_x'] = self.p['offset_x']
                    frame.attrs['offset_y'] = self.p['offset_y']
                    isSame = False
                else:
                    frame.attrs['offset_x'] = 0
                    frame.attrs['offset_y'] = 0

                if 'u_mod' in frame:
                    del frame['u_mod']
                    del frame['v_mod']
                    
                if isSame != True:
                    frame.create_dataset('u_mod', data = u)
                    frame.create_dataset('v_mod', data = v)
                print(f'Modifyied frame {i}')
                self.process_type.config(text = f'Modified frame {i}')
        except Exception as e:
            print('Could not finish modifying results\nReason: ' + str(e))
        self.enable_widgets()
        
        
    def average_results(self):
        start_processing = True
        start_processing = messagebox.askyesno(
            title = 'Warning',
            message = ('Do you want to average all frames?\n'+
                'You will not be able to stop once the averaging starts.')
        )
        if start_processing != True:
            raise Exception('Terminated process')
            
        #self.disable_widgets()
        #try:
        if 1 == 1:
            start = time.time()
            frame = self.session['results']['frame_0']
            x = np.array(self.session['results']['frame_0']['x'])
            y = np.array(self.session['results']['frame_0']['y'])
            u = np.array(frame['u'])
            v = np.array(frame['v'])    
            if 'u_val' in frame:
                u = np.array(frame['u_val'])
                v = np.array(frame['v_val'])
            if 'u_mod' in frame:
                u = np.array(frame['u_mod'])
                v = np.array(frame['v_mod'])
            u = u.astype('float32').reshape(-1)
            v = v.astype('float32').reshape(-1)
            u[u == np.nan] = 0
            v[v == np.nan] = 0
            n = 0
            if 'average' in self.session['results']:
                n += 1
            if 'ensemble' in self.session['results']:
                n += 1
                
            for i in range(1, len(self.p['fnames'])-n):
                frame = self.session['results'][f'frame_{i}']
                un = np.array(frame['u'])
                vn = np.array(frame['v'])    
                if 'u_val' in frame:
                    un = np.array(frame['u_val'])
                    vn = np.array(frame['v_val'])
                if 'u_mod' in frame:
                    un = np.array(frame['u_mod'])
                    vn = np.array(frame['v_mod'])
                un = u.astype('float32').reshape(-1)
                vn = v.astype('float32').reshape(-1)
                un[un == np.nan] = 0
                vn[vn == np.nan] = 0
                u += un.astype('float32')
                v += vn.astype('float32')
                print(f'Accululated frame {i}')
                
            u = np.reshape((u / (len(self.p['fnames'])-n)), x.shape)
            v = np.reshape((v / (len(self.p['fnames'])-n)), x.shape)
            
            end = time.time()
            
            if 'average' in self.session['results']:
                del self.session['results']['average']
            else:
                self.p['fnames'].append('Average')
                self.tkvars['fnames'].set(self.p['fnames'])
                del self.session['images']['frames']
                self.session['images'].create_dataset('frames', data = self.p['fnames'])
            
            frame = self.session['results'].create_group('average')
            frame.attrs['processed']    = True
            frame.attrs['roi_present']  = self.session['results']['frame_0'].attrs['roi_present']
            frame.attrs['roi_coords']   = self.session['results']['frame_0'].attrs['roi_coords']
            frame.attrs['mask_coords']  = '[]'
            frame.attrs['process_time'] = _round((end - start), 3)
            frame.attrs['offset_x']   = 0
            frame.attrs['offset_y']   = 0
            frame.attrs['scale_dist'] = self.session['results']['frame_0'].attrs['scale_dist']
            frame.attrs['scale_vel']  = self.session['results']['frame_0'].attrs['scale_vel']
            frame.attrs['units']      = self.session['results']['frame_0'].attrs['units']
            
            flag = np.zeros_like(x)
            s2n = np.zeros_like(x)
            frame.create_dataset('x', data = x)
            frame.create_dataset('y', data = y)
            frame.create_dataset('u', data = u)
            frame.create_dataset('v', data = v)
            frame.create_dataset('tp', data = flag)
            frame.create_dataset('s2n', data = s2n)            
        #except Exception as e:
        #    print("Failed to average results.\nReason: " + str(e))
        #self.enable_widgets()
        
            
    def __init_widgets(self):
        '''Creates a widget for each variable in a parameter object.'''
        self.__init_buttons()
        f = ttk.Frame(self)
        f.pack(side='left',
               fill='both',
               expand='True')
        
        # holds riders for parameters
        self.__init_notebook(f)
        
        # plotting area
        self.__init_fig_canvas(f)
        
        # variable widgets:
        for key in sorted(self.p.index, key=self.p.index.get):
            if self.p.type[key] == 'dummy':
                pass
            elif self.p.type[key] == 'dummy2':
                pass
            elif self.p.type[key] == 'bool':
                self.__init_checkbutton(key)
            elif self.p.type[key] == 'str[]':
                self.__init_listbox(key)
            elif self.p.type[key] == 'text':
                self.__init_text_area(key)
            elif self.p.type[key] == 'labelframe':
                self.__init_labelframe(key)
            elif self.p.type[key] == 'label':
                self.__init_label(key)
            elif self.p.type[key] == 'button_static_c':
                self.__init_button_static_c(key)
            elif self.p.type[key] == 'button_static_c2':
                self.__init_button_static_c2(key)
            elif self.p.type[key] == 'h-spacer':
                self.__init_horizontal_spacer(key)
            elif self.p.type[key] == 'sub_bool':
                self.__init_checkbutton(key, sub = True)
            elif self.p.type[key] == 'sub_bool2':
                self.__init_sub_checkbutton2(key)
            elif self.p.type[key] == 'sub_button2':
                self.__init_sub_button2(key) 
            elif self.p.type[key] == 'sub_button_static_c':
                self.__init_button_static_c(key, sub = True) 
            elif self.p.type[key] == 'sub_button_static_c2':
                self.__init_button_static_c2(key, sub = True)
            elif self.p.type[key] == 'sub_labelframe':
                self.__init_sub_labelframe(key)
            elif self.p.type[key] == 'sub_label':
                self.__init_label(key, sub = True)
            elif self.p.type[key] == 'sub_h-spacer':
                self.__init_horizontal_spacer(key, sub = True)
            elif self.p.type[key] is None:
                self.__add_tab(key)
            else:
                self.__init_entry(key)

            # create widgets that are not in OpenPivParams
            if self.p.index[key] == 2710:
                self.__init_ROI()
            elif self.p.index[key] == 2723:
                self.__init_mask_status() # status label widget 
            elif self.p.index[key] == 2408:
                self.__init_background_status()
            elif self.p.index[key] == 3015:
                self.__init_windowing_hint()
            elif self.p.index[key] == 8045:
                self.__init_vec_colorpicker(key)
            elif self.p.index[key] == 8070:
                self.__init_mask_vec_colorpicker(key)   
            #elif self.p.index[key] == 8045:
            #    self.__init_exclusions_preferences()
        self.selection(0)
                

    def __init_fig_canvas(self, mother_frame):
        '''Creates a plotting area for matplotlib.

        Parameters
        ----------
        mother_frame : ttk.Frame
            A frame to place the canvas in.
        '''
        self.fig = Fig(
            facecolor = '#eeeeee'#self.cget('bg')
        )
        self.fig_frame = ttk.Frame(mother_frame)
        self.fig_frame.pack(side='left',
                            fill='both',
                            expand='True')
        self.fig_canvas = FigureCanvasTkAgg(
            self.fig, master=self.fig_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(
            side='left',
            fill='x',
            expand='True'
        )
        NavigationToolbar2Tk.toolitems = []
        self.fig_toolbar = NavigationToolbar2Tk(
            self.fig_canvas,
            self.fig_frame
        )

        self.homeButton = self.url2img(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                'res\home.png'
            )
        )
            
        self.zoomButton = self.url2img(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                'res\zoom_to_rect.png'
            )
        )
        
        self.panButton = self.url2img(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                'res\move.png'
            )
        )
            
        self.ttk_widgets['home_button'] = ttk.Button(
            self.fig_toolbar,
            image = self.homeButton,
            command = self.fig_toolbar.home
            
        )
        self.ttk_widgets['home_button'].pack(side = 'left')
            
        self.ttk_widgets['zoom_button'] = ttk.Button(
            self.fig_toolbar,
            image = self.zoomButton,
            command = self.plot_zoom
        )
        self.ttk_widgets['zoom_button'].pack(side = 'left')
            
        self.ttk_widgets['pan_button'] = ttk.Button(
            self.fig_toolbar,
            image = self.panButton,
            command = self.plot_pan
        )
        self.ttk_widgets['pan_button'].pack(side = 'left')
        
        self.progressbar = ttk.Progressbar(
            self.fig_toolbar,
            orient = 'horizontal',
            length = self.progressbar_width, 
            mode = 'indeterminate'
        )
        self.progressbar.pack(side = 'left')
        
        self.process_type = ttk.Label(
            self.fig_toolbar,
            text = ' ',
        )
        self.process_type.pack(side = 'left')
        
        self.fig_toolbar.update()  
        self.fig_canvas._tkcanvas.pack(
            side='top',
            fill='both',
            expand='True'
        )
        self.fig_canvas.mpl_connect(
            "key_press_event",
            lambda: key_press_handler(
                event,
                self.fig_canvas,
                self.fig_toolbar
            )
        )

        
    def url2img(self, url):
        img = Image.open(url)
        img = ImageTk.PhotoImage(img)
        return img
    
    
    def plot_zoom(self):
        self.fig_toolbar.zoom()
        self.release_event = 0
        self.plot_interact = self.fig_canvas.mpl_connect(
            'button_release_event', 
            self.plot_end_zoom_interact
        )
        self.ttk_widgets['home_button'].config(state = 'disabled')
        self.ttk_widgets['zoom_button'].config(state = 'disabled')
        self.ttk_widgets['pan_button'].config(state = 'disabled')
    
    
    def plot_pan(self):
        self.fig_toolbar.pan()
        self.release_event = 0
        self.plot_interact = self.fig_canvas.mpl_connect(
            'button_release_event', 
            self.plot_end_pan_interact
        )
        self.ttk_widgets['home_button'].config(state = 'disabled')
        self.ttk_widgets['zoom_button'].config(state = 'disabled')
        self.ttk_widgets['pan_button'].config(state = 'disabled')
        
        
    def plot_end_zoom_interact(self, event):
        reset = False
        if event.inaxes is not None:
            reset = True
        else: pass # need to add suport of out of axis release
        if reset:
            self.fig_toolbar.zoom()
            self.disconnect(self.plot_interact)
            self.ttk_widgets['home_button'].config(state = 'normal')
            self.ttk_widgets['zoom_button'].config(state = 'normal')
            self.ttk_widgets['pan_button'].config(state = 'normal')
        
        
    def plot_end_pan_interact(self, event):
        reset = False
        if event.inaxes is not None:
            reset = True
        else: pass
        if reset:
            self.fig_toolbar.pan()
            self.disconnect(self.plot_interact)
            self.ttk_widgets['home_button'].config(state = 'normal')
            self.ttk_widgets['zoom_button'].config(state = 'normal')
            self.ttk_widgets['pan_button'].config(state = 'normal')
        
        
    def __fig_toolbar_key_pressed(self, event):
        '''Handles matplotlib toolbar events.'''
        key_press_handler(event,
                          self.fig_canvas,
                          self.fig_toolbar)

        
    def __init_notebook(self, mother_frame):
        '''The notebook is the root widget for tabs or riders.'''
        
        style = ttk.Style()
        style.configure(
            'lefttab.TNotebook', 
            #tabposition='es'
        )        
        style.layout('TNotebook.Tab', layoutspec = [])
        self.nb = ttk.Notebook(mother_frame, 
                               style = 'lefttab.TNotebook',
                               width=self.frame_width)
        
        self.nb.pack(side='right',
                     fill='y',
                     expand='False')

    def __add_tab(self, key):
        '''Add an additional rider to the notebook.'''
        self.set_frame.append(ttk.Frame(self.nb))
        self.nb.add(self.set_frame[-1], 
                    text='')
        
        
    def selection(self, num):
        for tabs in range (0, self.nb.index('end')):
            self.nb.tab(tabs, state='hidden')
        self.nb.tab(num, state = 'normal')
        self.nb.select(num)

        
    def __init_buttons(self):
        '''Add buttons and bind them to methods.'''
        f = ttk.Frame(self)
        files = ttk.Menubutton(f, text='File')
        options = tk.Menu(files, tearoff=0)
        files.config(menu = options)
        submenu = tk.Menu(options, tearoff=0)
        submenu.add_command(label = 'images',
                            command = lambda: self.selection(0))
        submenu.add_command(label = 'results',
                            command = lambda: self.selection(1))
        submenu.add_command(label = 'settings',
                            command = self.load_settings)
        submenu.add_command(label = 'session',
                            command = self.load_session)
        
        options.add_cascade(label='Load', menu=submenu)
        options.add_separator()
        submenu = tk.Menu(options, tearoff=0)
        submenu.add_command(label='settings', command = lambda: self.p.dump_settings(
            filedialog.asksaveasfilename(
                title = 'Settings Manager',
                defaultextension = '.json',
                filetypes = [('json', '*.json'), ]
            )))
        submenu.add_command(label='current figure', command = lambda: self.selection(25))
        submenu.add_command(label='pre-processed images', command = lambda: self.selection(27))
        submenu.add_command(label='results as ASCI-II', command = lambda: self.selection(26))
        #submenu.add_command(label='TecPlot')
        #submenu.add_command(label='ParaView')
        options.add_cascade(label='Export', menu=submenu)
        options.add_separator()
        submenu = tk.Menu(options, tearoff=0)
        submenu.add_command(label='settings', command = self.reset_params)
        submenu.add_command(label='session', command = self.reset_session)
        options.add_cascade(label='Reset', menu=submenu)
        options.add_separator()

        options.add_command(label='Exit', command=self.destroy)
        files.pack(side='left', fill='x')
        
        preproc = ttk.Menubutton(f, text='Pre-processing')
        options = tk.Menu(preproc, tearoff=0)
        preproc.config(menu=options)
        options.add_command(label='Transformations',
                             command=lambda: self.selection(3))
        options.add_command(label='Phase separation',
                             command=lambda: self.selection(4))
        options.add_command(label='Temporal filters',
                             command=lambda: self.selection(5))
        options.add_command(label='Spatial filters',
                             command=lambda: self.selection(6))
        preproc.pack(side='left', fill='x')
        options.add_command(label='Exclusions',
                             command=lambda: self.selection(7))

        calibrate = ttk.Menubutton(f, text='Calibration')
        options = tk.Menu(calibrate, tearoff=0)
        calibrate.config(menu=options)
        options.add_command(label='Calibration',
                             command=lambda: self.selection(14))
        calibrate.pack(side='left', fill='x')
        
        piv = ttk.Menubutton(f, text='Analysis')
        options = tk.Menu(piv, tearoff=0)
        piv.config(menu=options)
        options.add_command(label='PIV settings\\analyze',
                             command=lambda: self.selection(8))
        options.add_command(label='Advanced settings',
                             command=lambda: self.selection(9))
        options.add_command(label='First pass validation',
                             command=lambda: self.selection(10))
        options.add_command(label='Other pass validations',
                             command=lambda: self.selection(11))
        options.add_command(label='Post-processing',
                             command=lambda: self.selection(12))
        options.add_command(label='Data probe',
                             command=lambda: self.selection(13))
        piv.pack(side='left', fill='x')
        
        postproc = ttk.Menubutton(f, text='Post-processing')
        options = tk.Menu(postproc, tearoff=0)
        postproc.config(menu=options)
        options.add_command(label='Validate components',
                             command=lambda: self.selection(15))
        options.add_command(label='Modify components',
                             command=lambda: self.selection(16))
        postproc.pack(side='left', fill='x')

        plot = ttk.Menubutton(f, text='Data exploration')
        options = tk.Menu(plot, tearoff=0)
        plot.config(menu=options)
        options.add_command(
            label='Vectors', command=lambda: self.selection(17))
        options.add_command(
            label='Contours', command=lambda: self.selection(18))
        options.add_command(
            label='Streamlines', command=lambda: self.selection(19))
        options.add_command(
            label='Statistics', command=lambda: self.selection(20))
        options.add_command(
            label='Extractions', command=lambda: self.selection(21))
        options.add_command(
            label='Preferences', command=lambda: self.selection(22))
        plot.pack(side='left', fill='x')
        
        u_func = ttk.Menubutton(f, text='User function')
        options = tk.Menu(u_func, tearoff=0)
        u_func.config(menu=options)
        options.add_command(label='Show user function',
                             command=lambda: self.selection(24))
        options.add_command(label='Execute user function',
                             command=self.text_function)
        u_func.pack(side='left', fill='x')

        lab_func = ttk.Menubutton(f, text='Lab book')
        options = tk.Menu(lab_func, tearoff=0)
        lab_func.config(menu=options)
        options.add_command(label='Show lab book',
                             command=lambda: self.selection(23))
        lab_func.pack(side='left', fill='x')

        usage_func = ttk.Menubutton(f, text='Usage')
        options = tk.Menu(usage_func, tearoff=0)
        usage_func.config(menu=options)
        options.add_command(label='Usage',
                             command=lambda: messagebox.showinfo(
                                 title='Help',
                                 message=inspect.cleandoc(
                                     OpenPivGui.__doc__)))
        usage_func.pack(side='left', fill='x')

        web_func = ttk.Menubutton(f, text='Web')
        options = tk.Menu(web_func, tearoff=0)
        web_func.config(menu=options)
        options.add_command(label='Web', command=self.readme)
        web_func.pack(side='left', fill='x')

        f.pack(side='top', fill='x')

            
    def calculate_statistics(self, results):
        try:
            self.get_settings()
            typevector = results[5]
            if len(results[7]) == 0:
                mask_percent = 0
            else:
                mask = coords_to_xymask(results[1], results[2], results[7]).astype('bool')
                mask_percent = _round((np.count_nonzero(mask) / np.size(results[1]) * 100), 4)
            
            invalid = typevector.astype('bool') # possibly subtract the mask??
            invalid = np.count_nonzero(invalid)
            invalid_percent = _round(((invalid / np.size(results[1])) * 100), 4)
            try:
                s2n_mean = _round(np.mean(results[6]), 3)
            except: s2n_mean = 0.0
            self.tkvars['statistics_vec_amount'].set(results[1].size)
            self.tkvars['statistics_vec_time'].set(_round(results[0], 5))
            self.tkvars['statistics_vec_time2'].set(_round(((results[0] * 1000) / np.size(results[1])), 3))
            self.tkvars['statistics_vec_invalid'].set(invalid_percent)
            self.tkvars['statistics_vec_valid'].set(100 - invalid_percent)
            self.tkvars['statistics_vec_masked'].set(mask_percent)
            self.tkvars['statistics_s2n_mean'].set(float(str(s2n_mean)[0:5]))
        except Exception as e:
            print('Could not calculate statistics.')
            print('Reason: '+str(e))
            
            
    def disable_widgets(
        self,
        exclude_tab = None#deprecated
    ):
        print('Disabling widgets...')
        for key in self.ttk_widgets:
            self.ttk_widgets[key].config(state = 'disabled')
        self.lb.config(state='disabled')
    
    
    def enable_widgets(self):
        print('Enabling widgets...') 
        for key in self.ttk_widgets:
            self.ttk_widgets[key].config(state = 'normal')
        self.lb.config(state='normal')
        self.update_widget_state()
        self.set_windowing(0)
        for widget in disabled_widgets():
            self.ttk_widgets[widget].config(state = 'disabled')            
            
            
    def text_function(self, func = None):
        '''Executes user function.'''
        self.get_settings()
        if func == None:
            func = self.p['user_func_def']
        exec(func)

        
    def reset_params(self, do_menu = True):
        '''Reset parameters to default values.'''
        if do_menu:
            answer = messagebox.askyesno(
                title='Settings Manager',
                message='Reset all parameters to default values?')
        else:
            answer = True
        if answer == True:
            self.get_settings()
            f_a = self.p['files_a']
            f_b = self.p['files_b']
            self.p = OpenPivParams()
            AddInHandler.init_add_ins(self)
            self.set_settings()
            
            self.p['files_a'] = f_a
            self.p['files_b'] = f_b
            
            self.set_windowing(0)
            self.index = 0
            self.toggle = 'a'
            self.set_settings()
            
            if len(self.p['files_a']) != 0:
                self.num_of_frames.config(text = '0/{}'.format(len(self.p['files_a'])))
    
    
    def reset_session(self):
        '''Reset session'''
        answer = messagebox.askyesno(
            title='Session Manager',
            message='Reset/delete current session?')
        
        if answer == True:
            print('Closing session')
            self.session.close()
            print('Deleting session')
            os.remove(self.p.session_file)
            print('Creating sesssion')
            self.load_session(default = True)
            self.update_buttons_state(state = 'disabled', apply = True)
            self.update_widget_state()
            self.fig.clear()
            self.fig.canvas.draw()
        self.get_settings()
            
            
    def readme(self):
        '''Opens https://github.com/OpenPIV/openpiv_tk_gui.'''
        webbrowser.open('https://github.com/OpenPIV/openpiv_tk_gui')

                
    def load_settings(self):
        '''Load settings from a JSON file.'''
        settings = filedialog.askopenfilename(
            title = 'Settings Manager',
            defaultextension = '.json',
            filetypes = [('json', '*.json'), ])
        if len(settings) > 0:
            self.p.load_settings(settings)
            self.set_settings()
            self.update_widget_state()
            self.set_windowing(0)
            for widget in disabled_widgets():
                self.ttk_widgets[widget].config(state = 'disabled')  
    
    
    '''~~~~~~~~~~~~~~~~~~~~~~~listbox~~~~~~~~~~~~~~~~~~~~~~~'''
    def __init_listbox(self, key): # A bug occurs when separating the two listboxes. Why?
        '''Creates an interactive list of filenames.

        Parameters
        ----------
        key : str
            Key of a settings object.
        '''
        # root widget
        f_m = ttk.Frame(self)
        if key != 'img_list':
            f = ttk.Frame(f_m)
            width = 25
            side = 'top'
            padx, pady = 0, 0
        else:
            f = ttk.Frame(self.lf)
            width = 50
            side = None
            padx, pady =  5, 20
            F = ttk.Frame(f)
            ttk.Label(F, text='Number of images:    ').pack(
                side='left', anchor = 'nw')
            
            F2 = ttk.Frame(F)
            self.num_of_files = ttk.Label(F2, 
                text=str(len(self.p['img_list']) - 1)) 
            self.num_of_files.pack(side='left')
            F2.pack(side='left')
            F.pack(fill = 'x')
        # scrolling
        if key != 'img_list':
            sby = ttk.Scrollbar(f, orient="vertical")
            sby.pack(side='right', fill='y')
            self.lb = tk.Listbox(f, yscrollcommand=sby.set)
            sby.config(command=self.lb.yview)
            self.lb['width'] = width
        else:
            sbx = ttk.Scrollbar(f, orient="horizontal")
            sbx.pack(side='top', fill='x')
            sby = ttk.Scrollbar(f, orient="vertical")
            sby.pack(side='right', fill='y')
            
            self.lb = tk.Listbox(f, yscrollcommand=sbx.set)
            self.lb = tk.Listbox(f, yscrollcommand=sby.set)
            sbx.config(command=self.lb.xview)
            sby.config(command=self.lb.yview)
            self.lb['width'] = width

        f.pack(side=side,
               fill='both',
               expand='True',
               padx = padx,
               pady=pady)
        
        # background variable
        self.tkvars.update({key: tk.StringVar()})
        #self.tkvars[key].set(self.p[key])
        self.lb['listvariable'] = self.tkvars[key]
        try:
            self.tkvars[key].set(self.p[key])
        except: # no images stored
            pass

        # interaction and others
        if key != 'img_list':
            self.lb.bind('<<ListboxSelect>>', self.__listbox_selection_changed)
            self.lb.pack(side='top', fill='y', expand='True')
            
            # navigation buttons
            F = ttk.Frame(f_m)
            self.ttk_widgets['toggle_frames_button'] = ttk.Button(
                F,
                text='Toggle A/B',
                command=self.toggle_frames,
                width = width +1 
            )
            
            self.ttk_widgets['toggle_frames_button'].pack(
                fill='x',
                pady = 2
            )
            F.pack()
            
            #tools/info
            f = ttk.Frame(f_m)
            f.pack(fill = 'x')
            lf = tk.LabelFrame(f, text='statistics')
            lf.pack(side = 'bottom', fill = 'x')
            lf.config(borderwidth=2, height = 300, relief='groove')
            
            # number of files
            f = ttk.Frame(lf)
            ttk.Label(f, text=' frame: ').pack(side = 'left')
            self.num_of_frames = ttk.Label(f,
                text= str(len(self.p['files_a']) - 1))
            self.num_of_frames.pack(side = 'right')
            f.pack()
            
            # current point
            f = ttk.Frame(lf)
            ttk.Label(f, text = 'current point:').pack(side = 'left')
            f.pack(fill = 'x')
            
            f = ttk.Frame(lf)
            
            f1 = ttk.Frame(f)
            ttk.Label(f1, text = 'x: ').pack(side = 'left')
            self.point_x = ttk.Label(f1, text = 'N/A')
            self.point_x.pack(side = 'left')
            f1.pack(fill = 'x')
            
            f1 = ttk.Frame(f)
            ttk.Label(f1, text = 'y: ').pack(side = 'left')
            self.point_y = ttk.Label(f1, text = 'N/A')
            self.point_y.pack(side = 'left')
            f1.pack(fill = 'x')
            
            f1 = ttk.Frame(f)
            ttk.Label(f1, text = 'u: ').pack(side = 'left')
            self.point_u = ttk.Label(f1, text = 'N/A')
            self.point_u.pack(side = 'left')
            f1.pack(fill = 'x')
            
            f1 = ttk.Frame(f)
            ttk.Label(f1, text = 'v: ').pack(side = 'left')
            self.point_v = ttk.Label(f1, text = 'N/A')
            self.point_v.pack(side = 'left')
            f1.pack(fill = 'x')
            
            f1 = ttk.Frame(f)
            ttk.Label(f1, text = 'flag: ').pack(side = 'left')
            self.point_flag = ttk.Label(f1, text = 'N/A')
            self.point_flag.pack(side = 'left')
            f1.pack(fill = 'x')
            
            f.pack(fill = 'x', side='bottom')
            f_m.pack(expand = True, fill = 'y')
        else:
            self.lb.bind('<<ListboxSelect>>', self.__listbox2_selection_changed)
            self.lb.pack(side='top', fill='y', expand='True')
    
    
    def change_xy_current(self, event):
        if event.inaxes is not None:
            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
            if None in self.xlim or self.ylim:
                xmin = 0; ymin = 0
                ymax, xmax = self.img_shape
            else:
                xmin, xmax = self.xlim
                ymin, ymax = self.ylim
                
            if x > xmin and x < xmax and y > ymin and y < ymax: 
                self.point_x.config(text = f"{_round((x * self.scale_dist), 3)} {self.units[0]}")
                self.point_y.config(text = f"{_round((y * self.scale_dist), 3)} {self.units[0]}")
                try:
                    if self.session['results'][f'frame_{self.index}'].attrs['processed']:
                        xint, yint = x, y
                        x, y, u, v, flag = self.results[1:6]
                        mask = self.results[7]
                        xint = np.abs(x[0,:] / xint - 1) # find index from closest distance to cursor point
                        yint = np.abs(y[:,0] / yint - 1)
                        xint = int(np.where(xint == xint.min())[0])
                        yint = int(np.where(yint == yint.min())[0])
                        xint, yint = yint, xint
                        if len(mask) != 0:
                            mask = coords_to_xymask(x, y, mask).reshape(x.shape)
                        else:
                            mask = np.zeros_like(x)
                        if mask[xint, yint] == 0:
                            u = _round((u[xint, yint]), 4)
                            v = _round((v[xint, yint]), 4)

                            if flag[xint, yint] == 0: 
                                flag = 'valid'
                            else: flag = 'invalid'
                            if self.p['debug']:
                                print(f'x index: {xint}\ny index: {yint}')
                                print(f'axis limits: ({xmin},{xmax}), ({ymin}, {ymax})')

                            self.point_u.config(text = f"{u} {self.units[0]}/{self.units[1]}")
                            self.point_v.config(text = f"{v} {self.units[0]}/{self.units[1]}")
                            self.point_flag.config(text = f'{flag}')
                        else:
                            self.point_u.config(text = "N/A")
                            self.point_v.config(text = "N/A")
                            self.point_flag.config(text = 'masked')
                    else:
                        self.point_u.config(text = "N/A")
                        self.point_v.config(text = "N/A")
                        self.point_flag.config(text = 'N/A')
                except Exception as e:
                    self.point_u.config(text = "N/A")
                    self.point_v.config(text = "N/A")
                    self.point_flag.config(text = 'N/A')
                    if self.p['debug']:
                        print('Could not extract components for statistics/nReason: ' + str(e))
            else:
                self.point_x.config(text = "N/A")
                self.point_y.config(text = "N/A")
                self.point_u.config(text = "N/A")
                self.point_v.config(text = "N/A")
                self.point_flag.config(text = 'N/A')
            
            
    def __listbox_selection_changed(self, event):
        '''Handles selection change events of the file listbox.'''
        try:
            self.index = event.widget.curselection()[0]
        except IndexError: pass  # nothing selected
        else:
            if self.p['fnames'][self.index] not in ['Average', 'Ensemble']:
                for widget in self.toggled_buttons[0:12]:
                    self.ttk_widgets[widget].config(state = 'normal')
                self.get_settings()
                self.num_of_frames.config(text = (str(self.index)+'/'+str(len(self.p['files_a']) - 1)))
                self.mask_load_applied()
                print('Reason: moved to a different frame')
                self.show(self.p['files_' + self.toggle][self.index], preview = False)
                
            elif self.p['fnames'][self.index] == 'Average':
                for widget in self.toggled_buttons[0:12]:
                    self.ttk_widgets[widget].config(state = 'disabled')
                self.get_settings()
                self.num_of_frames.config(text = 'Average results')
                self.show(
                    self.p['files_' + self.toggle][0], 
                    preview = False,
                    perform_check = False,
                    results = self.session['results']['average'],
                )
            
            elif self.p['fnames'][self.index] == 'Ensemble':
                for widget in self.toggled_buttons[0:12]:
                    self.ttk_widgets[widget].config(state = 'disabled')
                self.get_settings()
                self.num_of_frames.config(text = 'Ensemble results')
                self.show(
                    self.p['files_' + self.toggle][0], 
                    preview = False,
                    perform_check = False,
                    results = self.session['results']['ensemble'],
                )
                
                    
            #if self.p['data_information'] == True:
            #    self.show_informations(self.p['files_' + self.toggle][self.index])
                
                
                
    def __listbox2_selection_changed(self, event):
        '''Handles selection change events of the file listbox.'''
        try:
            self.index_or = event.widget.curselection()[0]
        except IndexError: pass  # nothing selected
        else:
            self.get_settings()
            self.show(self.p['img_list'][self.index_or], bypass = True, preproc = False, perform_check = False)
            
            
    def get_filelistbox(self):
        '''Return a handle to the file list widget.

        Returns
        -------
        tkinter.Listbox
            A handle to the listbox widget holding the filenames
        '''
        return(self.lb)
    
    
    def toggle_frames(self):
        if self.toggle == 'a':
            self.toggle = 'b'
        else:
            self.toggle = 'a'
        self.show(self.p['files_' + self.toggle][self.index])
        print('Toggled to ' + self.toggle + ' frames')

        
    
    '''~~~~~~~~~~~~~~~~~~~~~~~textbox~~~~~~~~~~~~~~~~~~~~~~~'''
    def __init_text_area(self, key):
        '''Init a text area, here used as a lab-book, for example.

        The content is saved automatically to the parameter object,
        when the mouse leaves the text area.'''
        self.ta.append(tk.Text(self.set_frame[-1], undo=True))
        ta = self.ta[-1]
        ta.pack()
        ta.bind('<Leave>',
                (lambda _: self.__get_text(key, ta)))
        ttk.Button(self.set_frame[-1],
                   text='clear',
                   command=lambda: ta.delete(
            '1.0', tk.END)
        ).pack(fill='x')
        ttk.Button(self.set_frame[-1],
                   text='undo',
                   command=lambda: ta.edit_undo()
                   ).pack(fill='x')
        ttk.Button(self.set_frame[-1],
                   text='redo',
                   command=lambda: ta.edit_redo()
                   ).pack(fill='x')

        
    def __get_text(self, key, text_area):
        '''Get text from text_area and copy it to parameter object.'''
        self.p[key] = text_area.get('1.0', tk.END)
    
    
    
    '''~~~~~~~~~~~~~~~~~~~~frames/widgets~~~~~~~~~~~~~~~~~~~'''
    def __init_labelframe(self, key):
        '''Add a label frame for widgets.'''
        f = ttk.Frame(self.set_frame[-1])
        self.pane = ttk.Panedwindow(f, orient='vertical', width=self.frame_width, height=720)
        self.lf = tk.LabelFrame(self.pane, text=self.p.label[key])
        self.lf.config(borderwidth=2, width=self.frame_width, height=720, relief='groove')
        self.pane.add(self.lf)
        self.pane.pack(side='left', fill='both')
        f.pack(fill='both')

        
    def __init_sub_labelframe(self, key):
        '''Add a label frame for widgets.'''
        self.sub_lf = tk.LabelFrame(self.lf, text=self.p.label[key])
        self.sub_lf.config(borderwidth=2, width=self.frame_width, relief='groove')
        self.sub_lf.pack(fill='both', pady=4, padx=4)
        
        
    def __init_horizontal_spacer(self, key, sub = False):
        '''Add a horizontal spacer line for widgets.'''
        if sub:
            f = ttk.Frame(self.sub_lf)
        else:
            f = ttk.Frame(self.lf)
        hs = ttk.Separator(f)
        hs.pack(fill='x')
        f.pack(fill='both', pady=4)

        
    def __init_label(self, key, sub = False):
        if sub:
            f = ttk.Frame(self.sub_lf)
        else:
            f = ttk.Frame(self.lf)
        self.ttk_widgets[key] = ttk.Label(f,
                           text=self.p.label[key])
        self.ttk_widgets[key].pack(side='left')
        f.pack(fill = 'x')
        
        
    def roi_widgets(self, frame, key, label, label2, padx, pady):
        F = ttk.Frame(frame)
        self.ttk_widgets[label] = ttk.Label(F, text = label2)
        self.tkvars.update({key: tk.StringVar()})
        self.ttk_widgets[key] = ttk.Entry(F, width=8, justify = 'center')
        self.ttk_widgets[key]['textvariable'] = self.tkvars[key]
        self.ttk_widgets[label].pack()
        self.ttk_widgets[key].pack()
        F.pack(side='left', padx=padx, pady=pady)
    
    
    def __init_entry(self, key):
        '''Creates a label and an entry in a frame.

        A corresponding tk background textvariable is also crated. An 
        option menu is created instead of en entry, if a hint is given
        in the parameter object. The help string in the parameter object
        is used for creating a tooltip.

        Parameter
        ---------
        key : str
            Key of a parameter obj.
        '''
        optionMenuWidth = 10
        # sub label frames
        if(self.p.type[key] == 'sub_int' or
            self.p.type[key] == 'sub_int2' or
            self.p.type[key] == 'sub_float' or
            self.p.type[key] == 'sub'):
            f = ttk.Frame(self.sub_lf)
            f.pack(fill='x')
            
            if self.p.type[key] != 'sub_int2':
                side = 'right'
            else:
                side = 'left'
            if self.p.type[key] == 'sub_int':
                self.tkvars.update({key: tk.IntVar()})
            elif self.p.type[key] == 'sub_int2':
                self.tkvars[key] = tk.StringVar()
                self.tkvars.update({key: self.tkvars[key]})
                key2 = 'overlap' + key[4:13]
                self.tkvars[key2] = tk.StringVar()
                self.tkvars.update({key2: self.tkvars[key2]})
            elif self.p.type[key] == 'sub_float':
                self.tkvars.update({key: tk.DoubleVar()})
            elif self.p.type[key] == 'sub':
                self.tkvars.update({key: tk.StringVar()})
                
            if self.p.hint[key] is not None:
                if self.p.type[key] != 'sub_int2':
                    self.ttk_widgets[key] = ttk.OptionMenu(f,
                                      self.tkvars[key],
                                      self.p[key],
                                      *self.p.hint[key])
                    #self.ttk_widgets[key].config(width = optionMenuWidth)
                    
                    
                else:
                    self.ttk_widgets[key] = ttk.Combobox(f,
                                      textvariable = self.tkvars[key],
                                      width = 10, justify = 'center')
                    self.ttk_widgets[key]['values'] = self.p.hint[key]
                    #self.ttk_widgets[key2] = ttk.Entry(f,
                    #                  width = 12,
                    #                  justify = 'center'
                    #)
                    #self.ttk_widgets[key2]['textvariable'] = self.tkvars[key2]
                    self.ttk_widgets[key2] = ttk.Combobox(f,
                                      textvariable = self.tkvars[key2],
                                      width = 10, justify = 'center')
                    self.ttk_widgets[key2]['values'] = self.p.hint[key2]
                    
            else:
                self.ttk_widgets[key] = ttk.Entry(f, width=10, justify = 'center')
                self.ttk_widgets[key]['textvariable'] = self.tkvars[key]
            CreateToolTip(self.ttk_widgets[key], self.p.help[key])
            self.ttk_widgets[key].pack(side=side, padx = self.sub_padx, pady = self.sub_pady)
            if self.p.type[key] == 'sub_int2':
                self.ttk_widgets[key2].pack(side='right', padx = self.sub_padx, pady = self.sub_pady)
                #self.tkvars[key2].set(int(_round(self.overlap_percent * self.p[key], 0)))
                #self.generateOnChange(self.ttk_widgets[key])
                self.ttk_widgets[key].bind('<<Change>>', self.find_overlap)
                self.ttk_widgets[key].bind('<FocusOut>', self.find_overlap)
            else:
                self.ttk_widgets[key + '_label'] = ttk.Label(f, text=self.p.label[key])
                CreateToolTip(self.ttk_widgets[key + '_label'], self.p.help[key])
                self.ttk_widgets[key + '_label'].pack(side='left', padx = self.padx, pady = self.pady)
        else:
            f = ttk.Frame(self.lf)
            f.pack(fill='x')
            self.ttk_widgets[key + '_label'] = ttk.Label(f, text=self.p.label[key])
            CreateToolTip(self.ttk_widgets[key + '_label'], self.p.help[key])
            self.ttk_widgets[key + '_label'].pack(side='left', padx = self.padx, pady = self.pady)
            
            if self.p.type[key] == 'int':
                self.tkvars.update({key: tk.IntVar()})
            elif self.p.type[key] == 'float':
                self.tkvars.update({key: tk.DoubleVar()})
            else:
                self.tkvars.update({key: tk.StringVar()})
                
            if self.p.hint[key] is not None:
                self.ttk_widgets[key] = ttk.OptionMenu(f,
                                      self.tkvars[key],
                                      self.p[key],
                                      *self.p.hint[key])
                #self.ttk_widgets[key].config(width = optionMenuWidth)
            else:
                self.ttk_widgets[key] = ttk.Entry(f, width=10, justify = 'center')
                self.ttk_widgets[key]['textvariable'] = self.tkvars[key]
            CreateToolTip(self.ttk_widgets[key], self.p.help[key])
            self.ttk_widgets[key].pack(side='right', padx = self.padx, pady = self.pady)
        
        
    def __init_checkbutton(self, key, sub = False):
        '''Create a checkbutton with label and tooltip.'''
        if sub:
            f = ttk.Frame(self.sub_lf)
        else:
            f = ttk.Frame(self.lf)
        f.pack(fill='x', padx = self.padx, pady = self.pady)
        self.tkvars.update({key: tk.BooleanVar()})
        self.tkvars[key].set(bool(self.p[key]))
        self.ttk_widgets[key] = ttk.Checkbutton(f)
        self.ttk_widgets[key]['variable'] = self.tkvars[key]
        self.ttk_widgets[key]['onvalue'] = True
        self.ttk_widgets[key]['offvalue'] = False
        self.ttk_widgets[key]['text'] = self.p.label[key]
        CreateToolTip(self.ttk_widgets[key], self.p.help[key])
        self.ttk_widgets[key].pack(side='left')
        if self.p.hint[key] == 'bind': # for use of updating button font/bold
            self.ttk_widgets[key]['command'] = self.update_widget_state
        elif self.p.hint[key] == 'bind2':
            self.ttk_widgets[key]['command'] = self.update_buttons_state
        elif self.p.hint[key] == 'bind3':
            self.ttk_widgets[key]['command'] = lambda: self.set_windowing(0)

            
    def __init_sub_checkbutton2(self, key):
        '''Create a checkbutton with label and tooltip.'''
        f = ttk.Frame(self.sub_lf)
        f.pack(fill='x',padx = self.sub_padx, pady = self.sub_pady)
        self.tkvars[key] = tk.BooleanVar()
        self.tkvars.update({key: self.tkvars[key]})
        self.tkvars[key].set(bool(self.p[key]))
        self.ttk_widgets[key] = ttk.Checkbutton(f)
        self.ttk_widgets[key]['variable'] = self.tkvars[key]
        self.ttk_widgets[key]['onvalue'] = True
        self.ttk_widgets[key]['offvalue'] = False
        self.ttk_widgets[key]['text'] = self.p.label[key]
        CreateToolTip(self.ttk_widgets[key], self.p.help[key])
        self.ttk_widgets[key].pack(side='left')
        self.ttk_widgets[key]['command'] = lambda: self.set_windowing(0)

        
    def __init_exclusions_preferences(self):
        whitespace = '                   '
        keys = ['roi_border', 'mask_fill'] 
        command = [self.roi_border_colorpicker,
                   self.mask_fill_colorpicker]
        i = 0
        for key in keys:
            f = ttk.Frame(self.sub_lf)
            l = ttk.Label(f, text=self.p.label[key])
            CreateToolTip(l, self.p.help[key])
            l.pack(side='left', padx=self.sub_padx, pady=self.sub_pady)
            self.ttk_widgets[key] = tk.Button(f,
                                       text = whitespace,
                                       bg = self.p[key],
                                       relief = 'groove',
                                       command = command[i])
            self.ttk_widgets[key].pack(side='right', padx = self.sub_padx, pady = self.sub_pady)
            CreateToolTip(self.ttk_widgets[key], self.p.help[key])
            f.pack(fill='x')
            i+=1
        
        
    def __init_vec_colorpicker(self, key):
        whitespace = '                   '
        f = ttk.Frame(self.sub_lf)
        l = ttk.Label(f, text='valid vector color')
        CreateToolTip(l, self.p.help[key])
        l.pack(side='left', padx = self.sub_padx, pady = self.sub_pady)
        self.valid_color = tk.Button(f,
                                     text=whitespace,
                                     bg=self.p['valid_color'],
                                     relief='groove',
                                     command=self.valid_colorpicker)
        self.valid_color.pack(side='right', padx = self.sub_padx, pady = self.sub_pady)
        f.pack(fill='x') 
        
        f = ttk.Frame(self.sub_lf)
        l = ttk.Label(f, text='invalid vector color')
        CreateToolTip(l, self.p.help[key])
        l.pack(side='left', padx = self.sub_padx, pady = self.sub_pady)
        self.invalid_color = tk.Button(f,
                                       text=whitespace,
                                       bg=self.p['invalid_color'],
                                       relief='groove',
                                       command=self.invalid_colorpicker)
        self.invalid_color.pack(side='right', padx = self.sub_padx, pady = self.sub_pady)
        f.pack(fill='x')
        
        
    def __init_mask_vec_colorpicker(self, key):
        whitespace = '                   '
        f = ttk.Frame(self.sub_lf)
        l = ttk.Label(f, text='mask vector color')
        CreateToolTip(l, self.p.help[key])
        l.pack(side='left', padx = self.sub_padx, pady = self.sub_pady)
        self.mask_vec_color = tk.Button(f,
                                     text=whitespace,
                                     bg=self.p['mask_vec'],
                                     relief='groove',
                                     command=self.mask_vec_colorpicker)
        self.mask_vec_color.pack(side='right', padx = self.sub_padx, pady = self.sub_pady)
        f.pack(fill='x') 
        
        
    def __init_button_static_c(self, key, sub = False):
        if sub:
            f = ttk.Frame(self.sub_lf)
        else:
            f = ttk.Frame(self.lf)
        self.ttk_widgets[key] = ttk.Button(
            f,
            text = self.p.label[key],
            style = 'h12.TButton',
            width = self.large_button_width,
            command = lambda: self.text_function(self.p.hint[key])
        )
        self.ttk_widgets[key].pack(padx = 2, pady = 2)
        f.pack(fill = 'x')
        
        
    def __init_button_static_c2(self, key, sub = False):
        if sub:
            f = ttk.Frame(self.sub_lf)
        else:
            f = ttk.Frame(self.lf)
        self.ttk_widgets[key.split(',')[0]] = ttk.Button(
            f,
            text = self.p.label[key][0],
            style = 'h12.TButton',
            width = self.small_button_width,
            command = lambda: self.text_function(self.p.hint[key][0])
        )
        self.ttk_widgets[key.split(',')[0]].pack(
            side = 'left',
            padx = 2, 
            pady = 2
        )
        self.ttk_widgets[key.split(',')[1]] = ttk.Button(
            f,
            text = self.p.label[key][1],
            style = 'h12.TButton',
            width = self.small_button_width,
            command = lambda: self.text_function(self.p.hint[key][1])
        )
        self.ttk_widgets[key.split(',')[1]].pack(
            side = 'right',
            padx = 2, 
            pady = 2
        )
        f.pack(fill = 'x')
        
        
    def roi_border_colorpicker(self):
        self.p['roi_border'] = colorchooser.askcolor()[1]
        self.ttk_widgets['roi_border'].config(bg=self.p['roi_border'])
        
        
    def mask_fill_colorpicker(self):
        self.p['mask_fill'] = colorchooser.askcolor()[1]
        self.ttk_widgets['mask_fill'].config(bg=self.p['mask_fill'])
        
        
    def mask_vec_colorpicker(self):
        self.p['mask_vec'] = colorchooser.askcolor()[1]
        self.mask_vec_color.config(bg=self.p['mask_vec'])
        
        
    def invalid_colorpicker(self):
        self.p['invalid_color'] = colorchooser.askcolor()[1]
        self.invalid_color.config(bg=self.p['invalid_color'])

        
    def valid_colorpicker(self):
        self.p['valid_color'] = colorchooser.askcolor()[1]
        self.valid_color.config(bg=self.p['valid_color'])
    
    
    
    '''~~~~~~~~~~~~~~~~~~~~~loading images~~~~~~~~~~~~~~~~~~'''
    def select_image_files(self):
        '''Show a file dialog to select one or more filenames.'''
        print('Use Ctrl + Shift to select multiple files.')
        files = filedialog.askopenfilenames(multiple=True, filetypes = ((".tif","*.tif"),
                                                                        (".jpeg","*.jpeg"),
                                                                        (".jpg","*.jpg"),
                                                                        (".pgm","*.pgm"),
                                                                        (".png","*.png"),
                                                                        (".bmp","*.bmp"),
                                                                        ("all files","*.*")))
        if len(files) > 0:
            if len(files)==1:
                self.get_settings()
                warning = 'Please import two or more images.'
                if self.p['warnings']:
                    messagebox.showwarning(title='Error Message',
                                   message=warning)
                print(warning)
            else:
                self.p['img_list'] = list(files)
                self.tkvars['img_list'].set(self.p['img_list'])
                if 'img_list' in self.session['images']:
                    del self.session['images']['img_list']
                self.session['images'].create_dataset('img_list', data = self.p['img_list']) 
                self.num_of_files.config(text = str(len(files)))
                self.num_of_frames.config(text = '0/0')
                self.p['fnames'] = []
                self.tkvars['fnames'].set(self.p['fnames'])
                self.xy_connect = self.fig_canvas.mpl_connect('motion_notify_event', 
                                                          self.change_xy_current)
                self.update_buttons_state(state = 'disabled', apply = False)
                self.ttk_widgets['apply_frequence_button'].config(state = 'normal')
                self.ttk_widgets['remove_current_image'].config(state = 'normal')
                self.show(self.p['img_list'][0], bypass = True, preproc = False, perform_check = False)
                self.background_frame_a = []
                self.background_frame_b = []
                
                
    def remove_image(self, index):
        self.p['img_list'].pop(index)
        self.tkvars['img_list'].set(self.p['img_list'])
        self.num_of_files.config(text = str(len(self.p['img_list'])))
        
        
    def apply_frequencing(self):
        # custom image sequence with (1),(1+[x]) or (1+[1+x]),(2+[2+x]) or ((1+[1+x]),(3+[3+x]))
        print('Applying frequencing...')
        self.get_settings()
        self.disable_widgets()
        self.ttk_widgets['apply_frequence_button'].config(state = 'disabled')
        #self.p['fnames'].config(state = 'disabled')
        try:
            try:
                if self.p['img_list'][0] == 'none':
                    message = ('None type images. External results were loaded.\n'+
                               'Please select image files to continue.')
                    if self.p['warnings']:
                        messagebox.showwarning(title='Error Message',
                                       message=message)
                    raise Exception(message)
            except: pass
                    
            img_grp = self.session['images']

            if self.p['sequence'] == '(1+2),(1+3)':
                self.p['files_a'] = self.p['img_list'][0]
                self.p['files_b'] = self.p['img_list'][self.p['skip']::1]
                # making sure files_a is the same length as files_b
                for i in range(len(self.p['files_b'])):
                    self.p['files_a'].append(self.p['img_list'][0])
            else:
                if self.p['sequence'] == '(1+2),(2+3)':
                    step = 1
                else:
                    step = 2
                self.p['files_a'] = self.p['img_list'][0::step]
                self.p['files_b'] = self.p['img_list'][self.p['skip']::step]
                # making sure files_a is the same length as files_b
                diff = len(self.p['files_a'])-len(self.p['files_b'])
                if diff != 0:
                    for i in range(diff):
                        self.p['files_a'].pop(len(self.p['files_b']))

            if 'img_list' in img_grp:
                del img_grp['img_list']
            img_grp.create_dataset('img_list', data = list(self.p['img_list']))

            if 'files_a' in img_grp:
                del img_grp['files_a']
            img_grp.create_dataset('files_a', data = list(self.p['files_a']))

            if 'files_b' in img_grp:
                del img_grp['files_b']
            img_grp.create_dataset('files_b', data = list(self.p['files_b']))

            if 'settings' in img_grp:
                del img_grp['settings']
            settings = img_grp.create_group('settings')

            if 'results' in self.session:
                del self.session['results']
            self.session.create_group('results')

            print('Number of a files: ' + str(len(self.p['files_a'])))
            print('Number of b files: ' + str(len(self.p['files_b'])))
            self.p['fnames'] = []
            # set listbox names and results structure
            for i in range(len(self.p['files_a'])):
                self.p['fnames'].append('Frame '+str(str(i).zfill(math.ceil(
                                                        math.log10(len(self.p['files_a']))))))
                frame_set = settings.create_group(f'frame_{i}')
                frame_set.attrs['roi_coords'] = ['', '', '', '']
                frame_set.attrs['mask_coords'] = '[]'
                
                frame = self.session['results'].create_group(f'frame_{i}')
                frame.attrs['processed'] = False
                frame.attrs['units'] = ['px', 'dt']
                frame.attrs['scale_dist'] = 1
                frame.attrs['scale_vel'] = 1

            if 'frames' in img_grp:
                del img_grp['frames']
            img_grp.create_dataset('frames', data = self.p['fnames'])

            self.update_buttons_state(state = 'normal')
            self.toggle = 'a'
            self.p['fnames'] = list(self.p['fnames'])
            self.tkvars['fnames'].set(self.p['fnames'])
            self.num_of_frames.config(text=str(len(self.p['fnames'])))
            self.num_of_frames.config(text = '0/{}'.format(str(len(self.p['fnames']) - 1)))
            self.index = 0
            self.show(self.p['files_' + self.toggle][0])
            self.background_frame_a = []
            self.background_frame_b = []
            print('Allocated space for {} frames(s)'.format(len(self.p['fnames'])))
            print(f"Memory space for each list: ~{_round(np.array(self.p['files_a']).nbytes/1e6, 2)} megabytes")
            print('Frequencing applied')
            
        except Exception as e:
            print('Could not apply frequencing to currently select images.\n' +
                  'Reason: ' + str(e))
        self.enable_widgets()
        
        
    
    '''~~~~~~~~~~~~~~~~~~~~~loading results~~~~~~~~~~~~~~~~~~'''
    def select_result_files(self):
        '''Show a file dialog to select one or more filenames.'''
        print('Use Ctrl + Shift to select multiple files.')
        files = filedialog.askopenfilenames(
            multiple=True,
            filetypes = (
                (".vec","*.vec"),
                (".txt","*.txt"),
                (".dat","*.dat"),
                (".jvc","*.jvc"),
                (".csv","*.csv"),
            )
        )

        found = False
        if len(files) > 0:
            self.p['fnames'] = []
            self.p['files_a'] = []
            self.p['files_b'] = []
            self.p['img_list'] = []
            img_grp = self.session['images']
            if 'settings' in img_grp:
                del img_grp['settings']
            settings = img_grp.create_group('settings')
            if 'results' in self.session:
                del self.session['results']
            self.session.create_group('results')
            frame_num = 0
            for fname in files:
                sep = self.p['sep']
                if sep == 'tab':
                    sep = '\t'
                if sep == 'space':
                    sep = ' '

                ext = fname.split('.')[-1]
                if ext in ['txt', 'dat', 'jvc', 'vec', 'csv']:
                    found = True
                    data = pd.read_csv(
                        fname,
                        names=self.p['header_names'].split(','),
                        decimal=self.p['decimal'],
                        skiprows=int(self.p['skiprows']),
                        sep=sep
                    )
                    x = data[['x']].to_numpy()
                    y = data[['y']].to_numpy()
                    
                    rows = np.unique(y).shape[0]
                    cols = np.unique(x).shape[0]
                    x = x.reshape(rows, cols)
                    y = y.reshape(x.shape)
                    u = data[['u']].to_numpy().reshape(x.shape)
                    v = data[['v']].to_numpy().reshape(x.shape)
                    self.img_shape = [y.max(), x.max()]

                    if self.p['flip_u']:
                        u = np.flipud(u)

                    if self.p['flip_v']:
                        v = np.flipud(v)

                    if self.p['invert_u']:
                        u *= -1

                    if self.p['invert_v']:
                        v *= -1
                    try:
                        flag = data[['flag']].to_numpy().reshape(x.shape)
                        if np.all(flag):
                            flag = np.zeros_like(x)
                    except:
                        flag = np.zeros_like(x)
                    try:
                        s2n = data[['s2n']].to_numpy().reshape(x.shape)
                        if np.all(s2n):
                            flag = np.zeros_like(x)
                    except:
                        s2n = np.zeros_like(x)
                        
                    frame = self.session['results'].create_group(f'frame_{frame_num}')
                    frame.attrs['processed']    = True

                    frame.attrs['roi_present']  = False
                    frame.attrs['roi_coords']   = ['', '', '', '']
                    frame.attrs['mask_coords']  = '[]'
                    frame.attrs['process_time'] = 1e-10 # protect from possible division by zero 
                    frame.attrs['offset_x']   = 0
                    frame.attrs['offset_y']   = 0
                    frame.attrs['scale_dist'] = 1
                    frame.attrs['scale_vel']  = 1
                    frame.attrs['units']      = [self.p['loaded_units_dist'], self.p['loaded_units_vel']]
                    
                    frame.create_dataset('x', data = x)
                    frame.create_dataset('y', data = y)
                    frame.create_dataset('u', data = u)
                    frame.create_dataset('v', data = v)
                    frame.create_dataset('tp', data = flag)
                    frame.create_dataset('s2n', data = s2n)
                    self.p['fnames'].append(
                        'Frame '+str(str(frame_num).zfill(
                            math.ceil(
                                math.log10(len(files))
                            )
                        ))
                    )
                    self.p['files_a'].append('none')
                    self.p['files_b'].append('none')
                    self.p['img_list'].append('none')
                    frame_set = settings.create_group(f'frame_{frame_num}')
                    frame_set.attrs['roi_coords'] = ['', '', '', '']
                    frame_set.attrs['mask_coords'] = '[]' 
                    print(f'Loaded frame {frame_num}')
                    frame_num += 1
                else:
                    print('File could not be read. Possibly it is an image file.')
                    found = False
                    break;
            if found:
                self.update_buttons_state(state = 'normal')
                self.tkvars['fnames'].set(self.p['fnames'])
                self.tkvars['img_list'].set(self.p['img_list'])
                
                if 'img_list' in img_grp: del img_grp['img_list']
                img_grp.create_dataset('img_list', data = list(self.p['img_list']))

                if 'files_a' in img_grp: del img_grp['files_a']
                img_grp.create_dataset('files_a', data = list(self.p['files_a']))

                if 'files_b' in img_grp:
                    del img_grp['files_b']
                img_grp.create_dataset('files_b', data = list(self.p['files_b']))
                
                if 'frames' in img_grp: del img_grp['frames']
                img_grp.create_dataset('frames', data = self.p['fnames']) 
            
            self.background_frame_a = []
            self.background_frame_b = []
        
        
        
    '''~~~~~~~~~~~~~~~~~~~~~~~ROI~~~~~~~~~~~~~~~~~~~~~~~'''
    def __init_ROI(self):
        f = ttk.Frame(self.lf)
        f.pack(fill='x')
        sub_lf = tk.LabelFrame(f, text='Region of interest')
        sub_lf.config(borderwidth=2, width=self.frame_width, relief='groove')
        sub_lf.pack(fill='both', pady=4, padx=4)
        f = ttk.Frame(sub_lf)
        self.roi_status_frame = tk.Frame(f)
        self.ttk_widgets['roi_status'] = tk.Label(self.roi_status_frame, 
                                                text = 'ROI inactive')
        self.ttk_widgets['roi_status'].pack(anchor='n', fill='x', padx = 10, pady = 3)
        self.roi_status_frame.pack(fill='x', padx=4, pady=4)
        self.ttk_widgets['select_roi'] = ttk.Button(f,
                                              text='Select ROI',
                                              command=self.roi_select,
                                              style = 'h12.TButton',
                                              width=self.small_button_width)
        self.ttk_widgets['select_roi'].pack(side='left', padx=2)
        self.ttk_widgets['clear_roi'] = ttk.Button(f,
                                              text='Reset ROI',
                                              command=self.roi_clear,
                                              style = 'h12.TButton',
                                              width=self.small_button_width)
        self.ttk_widgets['clear_roi'].pack(side='right', padx=2)
        f.pack(fill='x')
        padx = 2
        pady = 2
        f = ttk.Frame(sub_lf)
        f.pack(fill='x', pady=0)
        keys = [['roi-xmin', 'roi-xmin_label', 'x'],
                ['roi-ymin', 'roi-xmax_label', 'y'],
                ['roi-xmax', 'roi-ymin_label', 'width'],
                ['roi-ymax', 'roi-ymax_label', 'height']]
        for i in range(4):
            self.roi_widgets(f, keys[i][0], keys[i][1], keys[i][2], padx, pady)
        f = ttk.Frame(sub_lf)
        self.ttk_widgets['apply_roi'] = ttk.Button(f,
                                              text='Apply to all frames',
                                              command=self.roi_apply,
                                              style = self.bold_apply,
                                              width=self.large_button_width)
        self.ttk_widgets['apply_roi'].pack(side='bottom', padx=2, pady=2)
        f.pack(fill='x')
    
    
    def roi_select(self):
        self.disable_widgets(exclude_tab = 2)
        self.initialize_roi_interact()
    
    
    def roi_apply(self):
        self.get_settings()
        if self.p['roi-xmin'] and self.p['roi-xmax'] and self.p['roi-ymin'] and self.p['roi-ymax'] != ('', ' '):
            self.ttk_widgets['roi_status'].config(bg = 'lime', text = 'ROI active')
            self.roi_status_frame.config(bg = 'lime')
        print('Applying current roi settings..')
        for i in range(len(self.p['fnames'])):
            self.session['images']['settings'][f'frame_{i}'].attrs['roi_present'] = True
            self.session['images']['settings'][f'frame_{i}'].attrs['roi_coords'] = [
                self.p['roi-xmin'],
                self.p['roi-xmax'],
                self.p['roi-ymin'],
                self.p['roi-ymax'],
            ]
        print('Applyied current roi settings')
        self.show(self.p['files_' + self.toggle][self.index])
            
            
    def roi_clear(self):
        self.tkvars['roi-xmin'].set('')
        self.tkvars['roi-xmax'].set('')
        self.tkvars['roi-ymin'].set('')
        self.tkvars['roi-ymax'].set('')
    
    
    def initialize_roi_interact(self):
        print('Initializing interactive roi selector')
        self.toggle_selector = RectangleSelector(self.ax, 
                                                 self.onselect_roi,
                                                 drawtype='box',
                                                 button=[1],
                                                 useblit = self.p['use_blit'],
                                                 rectprops = dict(facecolor=self.p['roi_border'], 
                                                                  edgecolor=self.p['roi_border'], 
                                                                  alpha=0.4, 
                                                                  fill=False))
        self.roi_rect = self.fig_canvas.mpl_connect('key_press_event', self.toggle_selector)
        plt.show()
    
    
    def onselect_roi(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        x1 = int(_round(eclick.xdata, 0))
        y1 = int(_round(eclick.ydata, 0))
        x2 = int(_round(erelease.xdata, 0))
        y2 = int(_round(erelease.ydata, 0))
        print('start position: (%f, %f)' % (x1, y1))
        print('end position: (%f, %f)' % (x2, y2))
        self.tkvars['roi-xmin'].set(x1)
        self.tkvars['roi-xmax'].set(x2)
        self.tkvars['roi-ymin'].set(y1)
        self.tkvars['roi-ymax'].set(y2)
        self.terminate_roi_interact()
        self.toggle_selector.set_active(False)
            
            
    def terminate_roi_interact(self):
        self.disconnect(self.roi_rect)
        self.enable_widgets()
        print('Exited interactive roi selector')
        
        
        
    '''~~~~~~~~~~~~~~~~~~~~~~~masking~~~~~~~~~~~~~~~~~~~~~~~'''
    def __init_mask_status(self):
        f = ttk.Frame(self.sub_lf)
        self.mask_status_frame = tk.Frame(f)
        self.mask_status_frame.pack(fill='x', padx = 4, pady = 4)
        self.ttk_widgets['mask_status'] = tk.Label(
            self.mask_status_frame, 
            text = 'Mask(s) inactive'
        )
        self.ttk_widgets['mask_status'].pack(
            anchor = 'n', 
            fill = 'x',
            padx = 10, 
            pady = 3
        )
        f.pack(fill = 'x')
    
    
    def mask_select(self):
        self.initialize_mask_interact()
        self.disable_widgets(exclude_tab = 2)
    
    
    def initialize_mask_interact(self):
        self.get_settings()
        if self.p['mask_type'] == 'polygon':
            self.toggle_selector = PolygonSelector(
                self.ax, 
                self.onselect_mask_poly,
                useblit = self.p['use_blit'],
                lineprops = dict(
                    color='#0198e0',
                    linestyle='-',
                    linewidth=2, 
                    alpha=0.9
                ),
                markerprops = dict(
                    marker='o',
                    markersize=1, 
                    mec='red', 
                    mfc='red', 
                    alpha=0.9
                ),
                vertex_select_radius = 10
            )
            
        elif self.p['mask_type'] == 'rectangle':
            self.toggle_selector = RectangleSelector(
                self.ax, 
                self.onselect_mask_rect,
                useblit = self.p['use_blit'],
                drawtype='box',
                button=[1],
                rectprops = dict(
                    facecolor='#0198e0', 
                    edgecolor='#0198e0', 
                    alpha=0.9, 
                    fill=False
                )
            )
        
        elif self.p['mask_type'] == 'lasso':
            self.toggle_selector = LassoSelector(
                self.ax,
                self.onselect_mask_poly,
                button=[1]
            )
        self.toggle_select = self.fig_canvas.mpl_connect('key_press_event', self.toggle_selector)
    
    
    def onselect_mask_poly(self, verts):
        "eclick and erelease are matplotlib events at press and release."
        #verts.append(verts[0])
        mask = list(literal_eval(self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords']))
        mask.append(verts)
        self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords'] = str(mask)
        
        self.mask_counter += 1
        self.show(self.p['files_' + self.toggle][self.index])
        self.mask_load_applied()
        self.terminate_mask_interact()
        self.toggle_selector.set_active(False)
        print('Exiting masking \nReason: clicked starting vertex')
        self.fig.canvas.draw()
        
        
    def onselect_mask_rect(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        x1 = int(_round(eclick.xdata, 0))
        y1 = int(_round(eclick.ydata, 0))
        x2 = int(_round(erelease.xdata, 0))
        y2 = int(_round(erelease.ydata, 0))
        
        mask = list(literal_eval(self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords']))
        mask.append([(x1, y1), (x2, y1),
            (x2, y2), (x1, y2)])
        self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords'] = str(mask)
        self.show(self.p['files_' + self.toggle][self.index])
        self.mask_load_applied()
        self.terminate_mask_interact()
        self.toggle_selector.set_active(False)
        print('Exiting masking \nReason: released mouse button.')       
        
        
    def mask_clear(self, update_plot = True):
        self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords'] = '[]'
        self.show(self.p['files_' + self.toggle][self.index])
        self.mask_load_applied()
    
    
    def mask_clear_all(self):
        for i in range(len(self.p['fnames'])):
            self.session['images']['settings'][f'frame_{i}'].attrs['mask_coords'] = '[]'
        self.show(self.p['files_' + self.toggle][self.index])
        self.mask_load_applied()
        
        
    def mask_save(self):
        filename = filedialog.asksaveasfilename(
            title = 'mask manager', 
            defaultextension = '.npy',
            filetypes = (('Text file', '*.npy'),)
        )
                    
        if len(filename) > 0:
            np.save(filename, self.object_mask)
            print('Saved current mask.')
                    
    
    def mask_load(self):
        print('Select a single mask file to apply to current frame')
        files = filedialog.askopenfilenames(
            title = 'Mask Manager',
            multiple = False, 
            filetypes = (('Text file', '*.npy'),)
        )
        if len(files) > 0:
            mask_coords = []
            for mask_object in np.load(files[0], allow_pickle = True):
                mask = [] # gets rid of an annoying bug, but is inefficient
                for coords in mask_object:
                    mask.append((coords[0], coords[1]))
                mask_coords.append(mask)
            self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords'] = str(mask_coords)
            self.mask_load_applied()
            self.show(self.p['files_' + self.toggle][self.index]) 
            print('Applied mask to frame {}'.format(self.p['fnames'][self.index]))
    
    
    def mask_load_applied(self):
        mask = list(literal_eval(self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords']))
        if len(mask) > 0:
            self.object_mask = mask
            print('Loaded current frame mask(s)')
        else:
            self.object_mask = []
            print('No mask(s) loaded to current frame')
    
    
    def mask_load_external(self):
        print('Use Ctrl + Shift to select multiple files. \n' +
              'Load and apply external for current frame on.')
        files = filedialog.askopenfilenames(title = 'Mask Manager',
                                            multiple=True, 
                                            filetypes = ((".tif","*.tif"),))
        if len(files) > 0:
            for i in range(len(files)):
                if (i + self.index) <= (len(self.p['files_a']) - 1):
                    mask = piv_tls.imread(files[i])
                    mask = np.pad(mask, [(1,1), (1,1)], mode = 'constant') # pad image to find edges, will offset mask by 1 pixel
                    #mask = mask_coordinates(mask, 1, 10, False)
                    #flipped_mask = []
                    #for coords in mask:
                    #    flipped_mask.append((coords[1], coords[0])) # fixes a naughty bug, can't process arrays
                    mask = self.preprocessing_methods["generate_mask"](mask, self.p, self.p['use_dynamic_mask'])
                    self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords'] = str(mask)
                else:
                    break;
            self.check_exclusion()
            self.show(self.p['files_' + self.toggle][self.index]) 
            self.mask_load_applied()
            print('Applied external mask(s).')
    
        
    def apply_mask_all(self):   
        self.get_settings()
        clear = True
        if len(self.object_mask) == 0:
            clear = messagebox.askyesno(title = 'Mask Manager',
                    message = 'Do you want to clear the mask(s) for selected frames?')

        if clear:
            print('Applying mask to select frames..')
            for i in get_selection(self.p['mask_selection'], len(self.p['fnames'])):
                self.session['images']['settings'][f'frame_{i}'].attrs['mask_coords'] = str(self.object_mask)
            print('Applied mask to select frames')
        self.show(self.p['files_' + self.toggle][self.index])
        self.mask_load_applied()
            
    
    def preview_dynamic_mask(self):
        print('Extracting roi')
        try:
            xmin, xmax, ymin, ymax = self.session['images']['settings']['frame_0'].attrs['roi_coords']
            xmin, xmax = int(xmin), int(xmax)
            ymin, ymax = int(ymin), int(ymax)
        except Exception as e:
            print('Not able to extract roi, reverting to default (image shape)')
            ymin = 0
            xmin = 0
            ymax = self.img_shape[0] - 1
            xmax = self.img_shape[1] - 1
            
        print('Generating a frame mask')
        maskA = self.preprocessing_methods["generate_mask"](
            piv_tls.imread(self.p['files_a'][self.index])[ymin:ymax,xmin:xmax], 
            self.p, True
        )
        print('Generating b frame mask')
        maskB = self.preprocessing_methods["generate_mask"](
            piv_tls.imread(self.p['files_b'][self.index])[ymin:ymax,xmin:xmax], 
            self.p, True
        )
        print('Combining masks')
        mask = maskA + maskB
        mask = np.array(mask, dtype = 'object')
        for i in range(len(mask)):
            temp = np.array(mask[i])
            temp[:,0] += xmin
            temp[:,1] += ymin
            mask[i] = temp
        self.show(self.p['files_' + self.toggle][self.index], bypass = True)
        add_disp_mask(
            self.ax,
            mask,
            invert = False,
            color = self.p['mask_fill'],
            alpha = self.p['mask_alpha']
        )
        self.fig.canvas.draw()
        
        
        
    def terminate_mask_interact(self):
        self.disconnect(self.toggle_select)
        self.enable_widgets()
        
        
        
    '''~~~~~~~~~~~~~~~~~~~~~preprocessing~~~~~~~~~~~~~~~~~~~'''
    def __init_background_status(self):
        f = ttk.Frame(self.sub_lf)
        f.pack(fill = 'x')
        self.background_status_frame = tk.Frame(f)
        self.ttk_widgets['background_status'] = tk.Label(self.background_status_frame, 
                                                text = 'Background inactive')
        self.ttk_widgets['background_status'].pack(
            anchor = 'n', 
            fill = 'x',
            padx = 10, 
            pady = 3
        )
        self.background_status_frame.pack(fill='x', padx = 4, pady = 4)
        
            
    def load_background_img(self):
        files = filedialog.askopenfilenames(multiple=True, 
                                            filetypes = (
                                            (".bmp","*.bmp"),
                                            (".jpeg","*.jpeg"),
                                            (".jpg","*.jpg"),
                                            (".pgm","*.pgm"),
                                            (".png","*.png"),
                                            (".tif","*.tif")))
        if len(files) == 1:
            print('Applying background image to both A and B frames')
            img = piv_tls.imread(files[0])
            self.background_frame_a = img
            self.background_frame_b = img
        elif len(files) == 2:
            print('Applying backgrounds for A and B frames')
            self.background_frame_a = piv_tls.imread(files[0])
            self.background_frame_b = piv_tls.imread(files[1])
        else:
            print('Please select one or two background images')
            
        if len(self.background_frame_a) > 1:
            self.background_status_frame.config(bg = 'lime')
            self.ttk_widgets['background_status'].config(
                text = 'Background active',
                bg = 'lime'
            )
            print('Stored background image(s)')   
        else:
            self.background_status_frame.config(bg = self.b_color)
            self.ttk_widgets['background_status'].config(
                text = 'Background inactive',
                bg = self.b_color
            ) 
             
        
    def generate_background_images(self):
        print('Generating A frames background')
        self.background_frame_a = self.preprocessing_methods['generate_background'](
            self.p['files_a'][self.p['starting_frame']:self.p['ending_frame']], 
            self.p['background_type']
        )
        print('Generating B frames background')
        self.background_frame_b = self.preprocessing_methods['generate_background'](
            self.p['files_b'][self.p['starting_frame']:self.p['ending_frame']], 
            self.p['background_type'],
        )
        self.background_status_frame.config(bg = 'lime')
        self.ttk_widgets['background_status'].config(
            text = 'Background active',
            bg = 'lime'
        )
        print('Finished generating background images')
    
    
    def preview_background(self):
        if len(self.background_frame_a) > 1:
            if self.toggle == 'a':
                background = self.background_frame_a
            else:
                background = self.background_frame_b
            
            self.fig.clear()
            ax = self.fig.add_axes([0,0,1,1])
            ax.matshow(background, cmap=plt.cm.Greys_r,
                     vmax=background.max(),
                     vmin=background.min(),
                    )
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks([])
            ax.set_yticks([])
            print(f'Max background intensity (before compression): {background.max()}')
            print(f'Min background intensity (before compression): {background.max()}')
            self.fig.canvas.draw()

        else:
            print("Background could shown\nReason: No background stored")
        
            
    def save_background(self):
        if len(self.background_frame_a) > 1:
            dirr = filedialog.asksaveasfilename(
                title = 'Save background images',
                defaultextension = '.bmp',
                filetypes = [
                    ('bmp', '*.bmp'), 
                    ('tif', '*.tif'),
                ]
            )
            ext = dirr.split('.')[-1]
            if len(dirr) > 1:
                print('Saving background image a')
                piv_tls.imsave(
                    dirr[:-len(ext)-1] + 'a' + dirr[-len(ext)-1:],
                    self.background_frame_a
                )
                print('Saving background image b')
                piv_tls.imsave(
                    dirr[:-len(ext)-1] + 'b' + dirr[-len(ext)-1:],
                    self.background_frame_b
                )
        else:
            print("Background could not save\nReason: No background stored")
            
            
            
    '''~~~~~~~~~~~~~~~~~~~windowing/PIV~~~~~~~~~~~~~~~~~~~~~'''
    def __init_windowing_hint(self):
        padx = 3
        pady = 2
        f = ttk.Frame(self.sub_lf)
        F = ttk.Frame(f)
        F.pack(fill='x')
        F = ttk.Frame(f)
        keys = [
            ['corr_window_1', 'left'],
            ['overlap_window_1', 'right']
        ]
        for key in keys:
            self.tkvars[key[0]] = tk.StringVar()
            self.tkvars[key[0]].set(self.p[key[0]])
            self.tkvars.update({key[0]: self.tkvars[key[0]]})
            self.ttk_widgets[key[0]] = ttk.Combobox(F,
                                      textvariable = self.tkvars[key[0]],
                                      width = 10, justify = 'center')
            self.ttk_widgets[key[0]]['values'] = self.p.hint[key[0]]
            CreateToolTip(self.ttk_widgets[key[0]], self.p.help[key[0]])
            self.ttk_widgets[key[0]].pack(side=key[1], padx=padx + 2, pady=pady)
            self.generateOnChange(self.ttk_widgets[key[0]])
            self.ttk_widgets[key[0]].bind('<<Change>>', self.find_percentage)
            self.ttk_widgets[key[0]].bind('<FocusOut>', self.find_percentage)
        
        F.pack(fill='x')
        F = ttk.Frame(f)
        self.ttk_widgets['overlap_label_perc'] = ttk.Label(F, 
            text = '= 00%')
        self.ttk_widgets['overlap_label_perc'].pack(side = 'right',padx=20)
        self.find_percentage(0, get_settings = False)
        F.pack(fill='x', side='bottom')
        f.pack(fill='x')
    
    
    def preview_grid(self, auto = True, window_size = 64, overlap = 32):
        self.get_settings()
        self.find_overlap(1)
        if auto:
            try:
                window_size = [int(self.p['corr_window_1']),
                               int(self.p['corr_window_1'])]
            except:
                window_size = [
                    int(list(self.p['corr_window_1'].split(','))[0]),
                    int(list(self.p['corr_window_1'].split(','))[1])
                ]
            try:
                overlap = [int(self.p['overlap_window_1']),
                           int(self.p['overlap_window_1'])]
            except:
                overlap = [
                    int(list(self.p['overlap_window_1'].split(','))[0]),
                    int(list(self.p['overlap_window_1'].split(','))[1])
                ]
                
            for i in range(2, 7):
                if self.p['pass_%1d' % i]:
                    try:
                        window_size = [int(self.p[f'corr_window_{i}']),
                                       int(self.p[f'corr_window_{i}'])]
                    except:
                        window_size = [
                            int(list(self.p[f'corr_window_{i}'].split(','))[0]),
                            int(list(self.p[f'corr_window_{i}'].split(','))[1])
                        ]
                    if self.p['update_overlap']:
                        overlap = [int(_round((window_size[0] * self.overlap_percent[0]), 0)),
                                   int(_round((window_size[1] * self.overlap_percent[1]), 0))]
                    else:
                        try:
                            overlap = [int(self.p[f'overlap_window_{i}']),
                                       int(self.p[f'overlap_window_{i}'])]
                        except:
                            overlap = [
                                int(list(self.p[f'overlap_window_{i}'].split(','))[0]),
                                int(list(self.p[f'overlap_window_{i}'].split(','))[1])
                            ]
                else:
                    break
        print(f'Final grid calculated: \nwindow_size: {window_size}\noverlap: {overlap}' )
        roi_coords = self.session['images']['settings'][f'frame_{self.index}'].attrs['roi_coords']
        mask_coords = list(literal_eval(self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords']))
        img = piv_tls.imread(self.p['files_a'][self.index])
        try:
            xmin = int(roi_coords[0])
            xmax = int(roi_coords[1])
            ymin = int(roi_coords[2])
            ymax = int(roi_coords[3])
            img = img[ymin:ymax, xmin:xmax]
            offset = [xmin, ymin]
        except:
            offset = [0,0]
        x, y = get_rect_coordinates(
            img.shape,
            window_size,
            overlap
        )
        x += offset[0]
        y += offset[1]
        if len(mask_coords) != 0:
            mask = coords_to_xymask(x, y, mask_coords)
            mask = mask.reshape(x.shape)
        else:
            mask = np.full_like(x, 0)
            
        self.show(self.p['files_' + self.toggle][self.index], bypass = True)
        self.ax.plot(x[mask == 0], y[mask == 0], 'y+')  
        self.ax.plot(
                x[mask == 1],
                y[mask == 1],
                color = self.p['mask_vec'],
                marker = self.p['mask_vec_style'],
                linestyle = '',
            )
        self.fig.canvas.draw()
        self.ttk_widgets['preview_grid'].config(
            text = 'Clear grid',
            command = self.clear_grid
        )
    
    
    def clear_grid(self):
        self.show(self.p['files_' + self.toggle][self.index])
        self.ttk_widgets['preview_grid'].config(
            text = 'Preview grid',
            command = self.preview_grid
        )
        
        
    def stop_analysis(self, clear = True):
        self.p['analysis'] = False
        self.get_settings()
        
        
    def find_percentage(self, a, get_settings = True):
        if get_settings:
            self.get_settings()
        try:
            window_size = [int(self.p['corr_window_1']),
                           int(self.p['corr_window_1'])]
        except:
            window_size = [
                int(list(self.p['corr_window_1'].split(','))[0]),
                int(list(self.p['corr_window_1'].split(','))[1])
            ]

        try:
            overlap = [int(self.p['overlap_window_1']),
                       int(self.p['overlap_window_1'])]
        except:
            overlap = [
                int(list(self.p['overlap_window_1'].split(','))[0]),
                int(list(self.p['overlap_window_1'].split(','))[1])
            ]
            
        self.overlap_percent = [overlap[0] / window_size[0], overlap[1] / window_size[1]]
        if self.overlap_percent[0] == self.overlap_percent[1]:
            perc = f'= {_round(self.overlap_percent[0] * 100, 1)}%'
        else:
            perc = f'= {_round(self.overlap_percent[0] * 100, 1)},{_round(self.overlap_percent[1] * 100, 1)}%'
            
        self.ttk_widgets['overlap_label_perc'].config(text = perc)
    
    
    def find_overlap(self, a):
        self.get_settings()
        self.set_windowing(0)
        if self.p['update_overlap']:
            for i in range(2, 7):
                try:
                    window_size = [int(self.p[f'corr_window_{i}']),
                                   int(self.p[f'corr_window_{i}'])]
                except:
                    window_size = [
                        int(list(self.p[f'corr_window_{i}'].split(','))[0]),
                        int(list(self.p[f'corr_window_{i}'].split(','))[1])
                    ]
                overlap = [int(_round(self.overlap_percent[0] * window_size[0], 0)),
                           int(_round(self.overlap_percent[1] * window_size[1], 0))]

                if overlap[0] == overlap[1]:
                    overlap = str(overlap[0])
                else:
                    overlap = f'{overlap[0]},{overlap[1]}'
                self.p[f'overlap_window_{i}'] = overlap
                self.tkvars[f'overlap_window_{i}'].set(self.p[f'overlap_window_{i}'])
    
    
    def set_windowing(self, a):
        self.get_settings()
        for i in range(2, 7):
            try:
                if self.p['update_overlap'] == False:
                    self.ttk_widgets[f'overlap_window_{i}'].config(state = 'disabled')
                self.ttk_widgets[f'corr_window_{i}'].config(state = 'disabled')
                self.tkvars[f'corr_window_{i}'].set(self.p[f'corr_window_{i}'])
            except:
                pass # widget is not created yet 
        for i in range(2, 7):
            if self.p[f'pass_{i}']:
                try:
                    if self.p['update_overlap'] == False:
                        self.ttk_widgets[f'overlap_window_{i}'].config(state = 'normal')
                    self.ttk_widgets[f'corr_window_{i}'].config(state = 'normal')
                    self.tkvars[f'corr_window_{i}'].set(self.p[f'corr_window_{i}'])
                except:
                    pass # widget is not created yet
            else:
                break;
        if self.p['update_overlap']:
            for i in range(2, 7):
                self.ttk_widgets[f'overlap_window_{i}'].config(state = 'disabled')
        
        
    def other_set_first(self):
        self.get_settings()
        self.tkvars['sp_MinU'].set(self.p['fp_MinU'])
        self.tkvars['sp_MinV'].set(self.p['fp_MinV'])
        self.tkvars['sp_MaxU'].set(self.p['fp_MaxU'])
        self.tkvars['sp_MaxV'].set(self.p['fp_MaxV'])
    
    
    def set_first_pass(self):
        self.get_settings()
        self.tkvars['fp_MinU'].set(self.p['MinU'])
        self.tkvars['fp_MinV'].set(self.p['MinV'])
        self.tkvars['fp_MaxU'].set(self.p['MaxU'])
        self.tkvars['fp_MaxV'].set(self.p['MaxV'])
                
            
    def reset_vel_val(self, type_):
        if type_ == 'first pass':
            vel_list = ['fp_MinU', 'fp_MinV',
                        'fp_MaxU', 'fp_MaxV']
        elif type_ == 'other pass':
            vel_list = ['sp_MinU', 'sp_MinV',
                        'sp_MaxU', 'sp_MaxV']
        else:
            vel_list = ['MinU', 'MinV',
                        'MaxU', 'MaxV']
        vals = [-10.0, -10.0, 10.0, 10.0]
        for i in range(len(vel_list)):
            self.tkvars[vel_list[i]].set(vals[i])
            #self.p[vel_list[i]] = vals[i]
            
                    
                
    '''~~~~~~~~~~~~~~~~~~~~Calibration~~~~~~~~~~~~~~~~~~~~~~'''
    def calibration_load(self):
        calibration_image = filedialog.askopenfilenames(multiple=True, 
                                                        filetypes = (
                                                        (".bmp","*.bmp"),
                                                        (".jpeg","*.jpeg"),
                                                        (".jpg","*.jpg"),
                                                        (".pgm","*.pgm"),
                                                        (".png","*.png"),
                                                        (".tif","*.tif")))
        if len(calibration_image) > 0:
            if len(calibration_image) == 1:
                self.show(calibration_image[0], 
                              preproc=False, bypass = True)
            else:
                self.get_settings()
                warning = 'Please select only one calibration image.'
                if self.p['warnings']:
                    messagebox.showwarning(title='Error Message',
                                   message=warning)
                print(warning)
    
    
    def calibration_ref_dist(self):
        self.disable_widgets(exclude_tab = self.nb.index('current'))
        self.xy_coords = self.fig_canvas.mpl_connect('button_press_event', 
                                                      self.get_calib_coords)
        ax = self.fig.axes[0]
        self.toggle_selector = Cursor(
            ax,
            horizOn = True,
            vertOn = True,
            useblit = self.p['use_blit'],
            color = 'y',
            linestyle = '--',
        )
        self.coord_counter = 0
        self.coords = []
        
        
    def get_calib_coords(self, event):
        if event.inaxes is not None:
            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
            self.coords.append((int(_round(x, 0)), int(_round(y, 0))))
            #self.ax.plot(x, y, '.', color = 'red')
            #self.fig.canvas.draw()
            print('Selected point at ' + str(self.coord_counter))
            self.coord_counter += 1
            if self.coord_counter == 2:
                self.disconnect(self.xy_coords)
                self.enable_widgets()
                print('Exiting interactive calibration')
                self.tkvars['starting_ref_point'].set(f'{self.coords[0][0]},{self.coords[0][1]}')
                self.tkvars['ending_ref_point'].set(f'{self.coords[1][0]},{self.coords[1][1]}')
                delx = (self.coords[1][0] - self.coords[0][0])**2
                dely = (self.coords[1][1] - self.coords[0][1])**2
                self.p['reference_dist'] = _round((delx + dely)**0.5, 4)
                self.tkvars['reference_dist'].set(self.p['reference_dist'])
                self.toggle_selector.set_active(False)
                print('Set reference distance')
    
    
    def calibration_clear(self):
        self.coords = []
        self.coord_counter = 0
        self.p['starting_ref_point'] = '0,0'
        self.tkvars['starting_ref_point'].set(self.p['starting_ref_point'])
        self.p['ending_ref_point'] = '0,0'
        self.tkvars['ending_ref_point'].set(self.p['ending_ref_point'])
        self.p['reference_dist'] = 1
        self.tkvars['reference_dist'].set(self.p['reference_dist'])
        self.p['real_dist'] = 1000
        self.tkvars['real_dist'].set(self.p['real_dist'])
        self.p['time_step'] = 1000
        self.tkvars['time_step'].set(self.p['time_step'])
        self.p['scale'] = 1
        self.tkvars['scale'].set(self.p['scale'])
        self.p['scale_unit'] = 'm'
        self.tkvars['scale_unit'].set(self.p['scale_unit'])
        print('Cleared scaling parameters')
    
    
    def get_calibration_scale(self):
        try: # protects gui from permenantly being disabed
            self.get_settings()
            #self.disable_widgets(exclude_tab = self.nb.index('current'))
            if self.p['scale_unit'] == 'm':
                unit_dist = .001 # mm to meters
            else:
                unit_dist = 0.1 # mm to cm
            
            scale = _round(((self.p['real_dist'] * unit_dist) / self.p['reference_dist']), 5)
            print(f"Distance scale: {scale}\nVelocity scale: {self.p['time_step'] * 0.001}")
            self.p['scale'] = scale
            self.tkvars['scale'].set(self.p['scale'])
        except Exception as e:
            print('Could not calculate scale.\nReason: ' + str(e))
        
    
    def apply_calibration(self):
        '''Wrapper function to start calibration in a separate thread.'''
        start_processing = True
        start_processing = messagebox.askyesno(
            title = 'Batch Processing Manager',
            message = ('Do you want to perform calibration?\n'+
                'You will not be able to stop once the calibration starts.')
        )
        if start_processing:
            try:
                check_processing(self)
                self.calibration_thread = threading.Thread(
                    target = self.calibrate_results
                )
                self.calibration_thread.start()
            except Exception as e:
                print('Stopping current processing thread \nReason: ' + str(e))
                
                s
    def calibrate_results(self):
        self.get_settings()
        self.disable_widgets()
        try:
            print('Determining user scaling units')
            if self.p['scale_unit'] == 'm':
                dis = 'm'
            else:
                dis = 'cm'
            if self.p['scale'] == 1:
                scl = 'px'
            else:
                scl = dis
            if self.p['time_step'] == 1000:
                dt = 'dt'
            else:
                dt = 's'
            units = [scl, dt]
            time_dt = 0.001
            for i in range(len(self.p['files_a'])):
                self.session['results'][f'frame_{i}'].attrs['scale_dist'] = self.p['scale']
                self.session['results'][f'frame_{i}'].attrs['scale_vel'] = self.p['time_step'] * time_dt
                self.session['results'][f'frame_{i}'].attrs['units'] = units
                self.process_type.config(text = f'Calibrated frame {i}')
            print('Applied scaling to all frames')
        except:
            print('Failed to apply scaling to frames')
        self.enable_widgets()
    
    
    
    '''~~~~~~~~~~~~~~~~~~~~Validation~~~~~~~~~~~~~~~~~~~~~~'''
    def initialize_vel_interact(self):
        if self.results[0]:
            self.disable_widgets(exclude_tab = self.nb.index('current'))
            self.get_settings()
            self.fig.clear()
            self.ax = self.fig.add_axes([0,0,1,1])
            vec_plot.scatter(
                [self.results[1],
                 self.results[2],
                 self.results[3],
                 self.results[4]],
                self.fig,
                self.ax,
                mask_coords = self.results[7]
            )
            self.fig.canvas.draw()
            self.toggle_selector = RectangleSelector(self.ax, 
                                                     self.onselect_vel_limit,
                                                     drawtype='box',
                                                     button=[1],
                                                     rectprops = dict(facecolor='k', 
                                                                      edgecolor='k', 
                                                                      alpha=1, 
                                                                      fill=False))
            self.roi_rect = self.fig_canvas.mpl_connect('key_press_event', self.toggle_selector)
            #plt.show()
        else:
            self.enable_widgets()
            print('No results found')
    
    
    def onselect_vel_limit(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        x1 = (_round(eclick.xdata, 3))
        y1 = (_round(eclick.ydata, 3))
        x2 = (_round(erelease.xdata, 3))
        y2 = (_round(erelease.ydata, 3))
        print('startposition: (%f, %f)' % (x1, y1))
        print('endposition  : (%f, %f)' % (x2, y2))
        self.tkvars['MinU'].set(x1)
        self.tkvars['MaxU'].set(x2)
        self.tkvars['MinV'].set(y1)
        self.tkvars['MaxV'].set(y2)
        self.terminate_vel_interact()
        self.toggle_selector.set_active(False)
            
            
    def terminate_vel_interact(self):
        self.disconnect(self.roi_rect)
        self.enable_widgets()
        self.show(self.p['files_' + self.toggle][self.index])
        print('Exited interactive velocity limits')
    
    
    
    '''~~~~~~~~~~~~~~~~~~~~Exporting~~~~~~~~~~~~~~~~~~~~~~'''
    def export_current_plot(self, index = None):
        self.get_settings()
        dirr = filedialog.askdirectory()
        if len(dirr) > 1:
            for i in range(len(self.p['fnames'])):
                if index != None:
                    i = index
                print(f'Saving frame {i}')
                try:
                    sizeX, sizeY = (
                        int(list(self.p['export1_figsize'].split(','))[0]),
                        int(list(self.p['export1_figsize'].split(','))[1])
                    )
                    fig = plt.figure(figsize = (sizeX, sizeY))
                except: fig = plt.figure()
                
                self.show(
                    self.p[f'files_{self.toggle}'][self.index],
                    extFig = fig,
                    preview = self.p['export1_modified_img']
                )
                fname = os.path.join(
                        dirr, 
                        self.p['export1_fname'].format(
                            str(i).zfill(math.ceil(math.log10(len(self.p['fnames']))))
                        ) + '.' + self.p['export1_ext']
                )
                #if self.p['export1_tight_layout']:
                #    fig.tight_layout()
                plt.savefig(
                    fname,
                    dpi = self.p['export1_dpi'],
                )
                print(f'Saved frame {i}')
                if index != None:
                        break;
                    
                    
    def export_asci2(self, index = None):
        self.get_settings()
        dirr = filedialog.askdirectory()
        if len(dirr) > 1:
            for i in range(len(self.p['fnames'])):
                if index != None:
                    i = index
                results = self.session['results'][f'frame_{i}']
                if results.attrs['processed']:
                    x = np.array(results['x'])
                    y = np.array(results['y'])
                    s2n = np.array(results['s2n'])
                    roi_present  = results.attrs['roi_present']                             
                    roi_coords   = results.attrs['roi_coords']
                    
                    if 'u_vld' in results:
                        if self.p['debug']:
                            print('Show: Loaded validated vectors')
                        u = np.array(results['u_vld'])
                        v = np.array(results['v_vld'])
                        flag = np.array(results['tp_vld'])
                        
                    if 'u_mod' in results:
                        if self.p['debug']:
                            print('Show: Loaded modified vectors')
                        u = np.array(results['u_mod'])
                        v = np.array(results['v_mod'])
                        if 'u_vld' in results:
                            flag = np.array(results['tp_vld']) 
                        else:
                            flag = np.array(results['tp'])
                            
                    if 'u_vld' not in results and 'u_mod' not in results:
                        if self.p['debug']:
                            print('Show: Loaded unprocessed vectors')
                        u = np.array(results['u'])
                        v = np.array(results['v'])
                        flag = np.array(results['tp'])  
                    x = x.astype(np.int)
                    y = y.astype(np.int)
                    
                    if roi_present:
                        if self.p['debug']:
                            print('Show: Found region of interest')
                        x += int(roi_coords[0])
                        y += int(roi_coords[2])
                        
                    mask = list(literal_eval(results.attrs['mask_coords']))
                    if mask == []:
                        mask = np.full_like(x, False)
                    else:
                        mask = coords_to_xymask(x, y, mask)
                        mask = mask.astype(bool).reshape(u.shape)
                    
                    if roi_present:
                        x -= int(roi_coords[0])
                        y -= int(roi_coords[2])
                        
                    if self.p['export2_set_masked_values'] == 'NaN':
                        u[mask == True] = np.nan
                        v[mask == True] = np.nan
                    else:
                        u[mask == True] = 0
                        v[mask == True] = 0
                    flag[mask == True] = 1
                    
                    if self.p['asci2_delimiter'] == 'tab':
                        delimiter = '\t'
                    elif self.p['asci2_delimiter'] == 'space':
                        delimiter = ' '
                    else:
                        delimiter = self.p['asci2_delimiter']
                        
                    filename = os.path.join(
                        dirr, 
                        self.p['export2_fname'].format(
                            str(i).zfill(math.ceil(math.log10(len(self.p['fnames']))))
                        ) + self.p['asci2_extension']
                    )
                    
                    if self.p['export2_components'] == 'x,y,u,v,flag':
                        components = [x,y,u,v,flag]
                    elif self.p['export2_components'] == 'x,y,u,v,flag,s2n':
                        components = [x,y,u,v,flag,s2n]
                    elif self.p['export2_components'] == 'x,y,u,v,s2n,flag':
                        components = [x,y,u,v,s2n,flag]
                    else:
                        components = [x,y,u,v]
                        
                    save(components, filename = filename, delimiter = delimiter)
                    print(f'Saved frame {i}.')
                    if index != None:
                        break;
                else:
                    print(f'Frame {i} does not have stored results. Stopping function.')
                    break;
                    
                    
    def export_current_image(self, index = None):
        self.get_settings()
        dirr = filedialog.askdirectory()
        if len(dirr) > 1:
            for i in range(len(self.p['fnames'])):
                if index != None:
                    i = index
                print(f'Saving frame {i}')
                roi_xmin, roi_xmax, roi_ymin, roi_ymax = self.session['images']\
                    ['settings'][f'frame_{i}'].attrs['roi_coords']
                fname = os.path.join(
                    dirr, 
                    self.p['export3_fname'].format(
                        str(i).zfill(math.ceil(math.log10(len(self.p['fnames']))))
                    ) + 'a.' + self.p['export3_ext']
                )
                try:
                    mask_coords = list(literal_eval(self.session['images']['settings'][f'frame_{i}'].attrs['mask_coords']))
                except Exception as e:
                    print(str(e))
                    mask_coords = []
                if len(mask_coords) > 0:
                    for i in range(len(mask_coords)):
                        temp = np.array(mask_coords[i])
                        temp[:,0] = temp[:,0] - int(roi_xmin)
                        temp[:,1] = temp[:,1] - int(roi_ymin)
                        mask_coords[i] = temp
                img = piv_tls.imread(self.p['files_a'][i])
                maxVal = img.max()
                original_dtype = img.dtype
                img = img.astype(np.float64) # minimize compression loss
                img /= maxVal
                
                if len(self.background_frame_a) > 1:
                    print('Removing background for image 1')
                    img = self.preprocessing_methods['temporal_filters'](img, self.background_frame_a/maxVal, self.p)

                if self.p['apply_second_only'] != True:
                    print('Transforming image 1')
                    img = self.preprocessing_methods['transformations'](img, self.p)

                if self.p['do_phase_separation'] == True:
                    print('Separating phases for image 1')
                    img = self.preprocessing_methods['phase_separation'](img, self.p)

                print('Pre-processing image 1')
                img = self.preprocessing_methods['spatial_filters'](
                        img, 
                        self.p,
                        preproc            = True,
                        roi_xmin           = roi_xmin,
                        roi_xmax           = roi_xmax,
                        roi_ymin           = roi_ymin,
                        roi_ymax           = roi_ymax,
                )
                
                if len(mask_coords) >= 1:
                    print('Applying mask to image 1')
                    img = self.preprocessing_methods['apply_mask'](img, mask_coords, self.p)
                    
                if self.p['export3_convert_uint8'] == True:
                    img[img<0] = 0
                    img = np.uint8(img*2**8)
                else:
                    img[img<0] = 0
                    img = (img*maxVal).astype(original_dtype)
                    
                piv_tls.imsave(fname, img)
                
                fname = os.path.join(
                    dirr, 
                    self.p['export3_fname'].format(
                        str(i).zfill(math.ceil(math.log10(len(self.p['fnames']))))
                    ) + 'b.' + self.p['export3_ext']
                )
                
                img = piv_tls.imread(self.p['files_b'][i])
                maxVal = img.max()
                original_dtype = img.dtype
                img = img.astype(np.float64) # minimize compression loss
                img /= maxVal
                
                if len(self.background_frame_b) > 1:
                    print('Removing background for image 2')
                    img = self.preprocessing_methods['temporal_filters'](img, self.background_frame_b/maxVal, self.p)

                print('Transforming image 2')
                img = self.preprocessing_methods['transformations'](img, self.p)

                if self.p['do_phase_separation'] == True:
                    print('Separating phases for image 2')
                    img = self.preprocessing_methods['phase_separation'](img, self.p)

                print('Pre-processing image 2')
                img = self.preprocessing_methods['spatial_filters'](
                        img, 
                        self.p,
                        preproc            = True,
                        roi_xmin           = roi_xmin,
                        roi_xmax           = roi_xmax,
                        roi_ymin           = roi_ymin,
                        roi_ymax           = roi_ymax,
                )
                
                if len(mask_coords) >= 1:
                    print('Applying mask to image 2')
                    img = self.preprocessing_methods['apply_mask'](img, mask_coords, self.p)
                    
                if self.p['export3_convert_uint8'] == True:
                    img[img<0] = 0
                    img = np.uint8(img*2**8)
                else:
                    img[img<0] = 0
                    img = (img*maxVal).astype(original_dtype)
                    
                piv_tls.imsave(fname, img)
                print(f'Saved frame {i}')
                if index != None:
                    break;
                        
                        
    def generateOnChange(self, obj):
        # idea from https://stackoverflow.com/questions/3876229/how-to-run-a-code-whenever-a-tkinter-widget-value-changes
        obj.tk.eval('''
            proc widget_proxy {widget widget_command args} {

                # call the real tk widget command with the real args
                set result [uplevel [linsert $args 0 $widget_command]]

                # generate the event for certain types of commands
                if {([lindex $args 0] in {insert replace delete}) ||
                    ([lrange $args 0 2] == {mark set insert}) || 
                    ([lrange $args 0 1] == {xview moveto}) ||
                    ([lrange $args 0 1] == {xview scroll}) ||
                    ([lrange $args 0 1] == {yview moveto}) ||
                    ([lrange $args 0 1] == {yview scroll})} {

                    event generate  $widget <<Change>> -when tail
                }

                # return the result from the real widget command
                return $result
            }
            ''')
        obj.tk.eval('''
            rename {widget} _{widget}
            interp alias {{}} ::{widget} {{}} widget_proxy {widget} _{widget}
        '''.format(widget=str(obj)))
        
        
    def disconnect(self, connected):
        self.fig_canvas.mpl_disconnect(connected)
    
    
    def update_widget_state(self):
        self.get_settings()
        for key, keys in self.toggled_widgets.items():
            if self.p[key] == True:       
                for i in range(len(keys)):
                    self.ttk_widgets[keys[i]].config(state = 'normal') 
                    try:
                         self.ttk_widgets[keys[i]+ '_label'].config(state = 'normal')
                    except: pass # no label
                    if key == 'vld_global_thr':
                        self.ttk_widgets['set_vel_limits'].config(state = 'normal')
                        self.ttk_widgets['apply_glov_val_first_pass'].config(state = 'normal')
            else:
                for i in range(len(keys)):
                    self.ttk_widgets[keys[i]].config(state = 'disabled')
                    try:
                         self.ttk_widgets[keys[i]+ '_label'].config(state = 'disabled')
                    except: pass # no label
                    if key == 'vld_global_thr':
                        self.ttk_widgets['set_vel_limits'].config(state = 'disabled')
                        self.ttk_widgets['apply_glov_val_first_pass'].config(state = 'disabled')
                        
                        
    def update_buttons_state(self, state = 'normal', apply = True):
        for widget in self.toggled_buttons:
            self.ttk_widgets[widget].config(state = state)
        if apply:
            self.ttk_widgets['apply_frequence_button'].config(state = state)        
        
        
    def check_exclusion(self, ignore_blank = False):    
        self.get_settings()
        frame = self.session['images']['settings'][f'frame_{self.index}']
        try:
            roi = frame.attrs['roi_coords']
            xmin = roi[0]
            xmax = roi[1]
            ymin = roi[2]
            ymax = roi[3]
            
            if xmin and xmax and ymin and ymax != ('', ' '):
                self.ttk_widgets['roi_status'].config(bg = 'lime', text = 'ROI active')
                self.roi_status_frame.config(bg = 'lime')  
            else:
                self.roi_status_frame.config(bg = self.b_color) 
                self.ttk_widgets['roi_status'].config(bg = self.b_color,
                                                      text = 'ROI inactive')
                
            if self.p['roi-xmin'] and self.p['roi-ymin'] and self.p['roi-xmax'] and self.p['roi-ymax'] == ('', ' '):
                #and ignore_blank != True:
                print('Setting ROI widgets to selected ROI of image')
                self.tkvars['roi-xmin'].set(xmin)
                self.tkvars['roi-ymin'].set(ymin)
                self.tkvars['roi-xmax'].set(xmax)
                self.tkvars['roi-ymax'].set(ymax)
                self.ttk_widgets['roi_status'].config(bg = 'lime', text = 'ROI active')
                self.roi_status_frame.config(bg = 'lime') 
        except Exception as e:
            print('Could not check ROI status \nReason: '+str(e))  
            
        if len(list(literal_eval(frame.attrs['mask_coords']))) > 0:
            self.mask_status_frame.config(bg = 'lime')
            self.ttk_widgets['mask_status'].config(bg = 'lime', text = 'Mask active')
        else:
            self.mask_status_frame.config(bg = self.b_color)
            self.ttk_widgets['mask_status'].config(bg = self.b_color, text = 'Mask inactive'.format(self.mask_counter))
        
        if len(self.background_frame_a) > 1:
            self.background_status_frame.config(bg = 'lime')
            self.ttk_widgets['background_status'].config(
                text = 'Background active',
                bg = 'lime'
            )
        else:
            self.background_status_frame.config(bg = self.b_color)
            self.ttk_widgets['background_status'].config(
                text = 'Background inactive',
                bg = self.b_color
            ) 
        for widget in disabled_widgets():
            self.ttk_widgets[widget].config(state = 'disabled')

            
    def log(self, columninformation=None, timestamp=False, text=None,
            group=None):
        ''' Add an entry to the lab-book.

        The first initialized text-area is assumed to be the lab-book.
        It is internally accessible by self.ta[0].

        Parameters
        ----------
        timestamp : bool
            Print current time.
            Pattern: yyyy-mm-dd hh:mm:ss.
            (default: False)
        text : str
            Print a text, a linebreak is appended. 
            (default None)
        group : int
            Print group of parameters.
            (e.g. OpenPivParams.PIVPROC)
        columninformation : list
            Print column information of the selected file.

        Example
        -------
        log(text='processing parameters:', 
            group=OpenPivParams.POSTPROC)
        '''
        if text is not None:
            self.ta[0].insert(tk.END, text + '\n')
        if timestamp:
            td = datetime.today()
            s = '-'.join((str(td.year), str(td.month), str(td.day))) + \
                ' ' + \
                ':'.join((str(td.hour), str(td.minute), str(td.second)))
            self.log(text=s)
        if group is not None:
            self.log(text='Parameters:')
            for key in self.p.param:
                key_type = self.p.type[key]
                if key_type not in ['labelframe', 'sub_labelframe', 'h-spacer', 'label',
                                    'sub_h-spacer', 'post_button', 'dummy', 'button_static_c'
                                    'button_static_c2', 'sub_button_static_c', 'sub_button_static_c2']:
                    if group < self.p.index[key] < group+1000:
                        s = str(self.p.label[key]) + ': ' + str(self.p[key])
                        self.log(text=s)
        if columninformation is not None:
            self.ta[0].insert(tk.END, str(columninformation) + '\n')
            
            
    def show_informations(self, fname):
        ''' Shows the column names of the chosen file in the labbook.

        Parameters
        ----------
        fname : str
            A filename.
        '''
        data = self.load_pandas(fname)
        if isinstance(data, str) == True:
            self.log(text=data)
        else:
            self.log(columninformation=list(data.columns.values))
            
            
    def get_settings(self):
        '''Copy widget variables to the parameter object.'''
        for key in self.tkvars:
            if self.p.type[key] == 'str[]':
                try:
                    self.p[key] = str2list(self.tkvars[key].get())
                except Exception as e:
                    print(str(e))
            else:
                try:
                    self.p[key] = self.tkvars[key].get()
                except Exception as e:
                    print(str(e))
        self.__get_text('lab_book_content', self.ta[0])
        self.__get_text('user_func_def', self.ta[1])
        
    
    def load_session(self, default = False):
        if default == False:
            self.file = filedialog.askopenfilename(title = 'Session Manager',
                                              defaultextension = '.hdf5',
                                              filetypes = [('HDF5', '*.hdf5'), ])
        else:
            self.file = self.p.session_file
            
        if len(self.file) > 0:
            try:
                
                self.session.close()
                print('Closed session HDF5 file')
            except: pass # session not opened yet
            try:
                if os.path.exists(self.file): # check if file exists
                    self.session = h5py.File(self.file, 'a')
                else:
                    self.session = h5py.File(self.file, 'w')
                    images = self.session.create_group('images')
                    images.create_dataset('img_list', data = [])
                    images.create_dataset('files_a', data = [])
                    images.create_dataset('files_b', data = [])
                    images.create_dataset('frames', data = [])
                    self.session.create_group('results')

                print('Loading image list into memory..')
                try:
                    self.p['img_list'] = list(self.session['images']['img_list'].asstr()[:])
                except: self.p['img_list'] = []
        
                print('Loading frames into memory..')
                try:
                    self.p['files_a'] = list(self.session['images']['files_a'].asstr()[:])
                    self.p['files_b'] = list(self.session['images']['files_b'].asstr()[:])
                    self.p['fnames'] = list(self.session['images']['frames'].asstr()[:])
                except:
                    print('Could not load frames')
                    self.p['files_a'] = []
                    self.p['files_b'] = []
                    self.p['fnames'] = []
                
                self.tkvars['img_list'].set(self.p['img_list'])
                self.tkvars['fnames'].set(self.p['fnames'])
        
                self.num_of_frames.config(text = '0/'+str(len(self.p['fnames'])-1))
                self.num_of_files.config(text = str(len(self.p['img_list'])))
                self.toggle = 'a'
                
                if len(self.p['files_a']) == 0 and len(self.p['img_list']) == 0:
                    self.update_buttons_state(state = 'disabled')

                elif len(self.p['files_a']) == 0 and len(self.p['img_list']) != 0:
                    self.update_buttons_state(state = 'disabled')
                    self.ttk_widgets['apply_frequence_button'].config(state = 'normal')
                    self.ttk_widgets['remove_current_image'].config(state = 'normal')
                    self.xy_connect = self.fig_canvas.mpl_connect(
                        'motion_notify_event', 
                        self.change_xy_current
                    )

                elif len(self.p['files_a']) != 0:
                    self.update_buttons_state(state = 'normal')
                    self.xy_connect = self.fig_canvas.mpl_connect(
                        'motion_notify_event', 
                        self.change_xy_current
                    )
                    self.index = 0
                    self.toggle = 'a'
                try:
                    if self.p['files_a'][0] == 'none':
                        self.update_buttons_state(state = 'normal')
                except: pass
            except Exception as e:
                print('Could not load session\nReason: ' + str(e))        
        
        
    def set_settings(self):
        '''Copy values of the parameter object to widget variables.'''
        for key in self.tkvars:
            if key not in ['img_list', 'fnames', 'files_a', 'files_b', 'frames']:
                try: 
                    self.tkvars[key].set(self.p[key])
                except Exception as e: # if an error occurs, shorten the output
                    print(str(e))
        self.ta[0].delete('1.0', tk.END)
        self.ta[0].insert('1.0', self.p['lab_book_content'])
        self.ta[1].delete('1.0', tk.END)
        self.ta[1].insert('1.0', self.p['user_func_def'])
    
    
    def clear_results(self, update_plot = True): 
        n = 0
        if 'Average' in self.p['fnames']:
            n += 1
            self.p['fnames'].pop(self.p['fnames'].index('Average'))
            del self.session['results']['average']
        if 'Ensemble' in self.p['fnames']:
            n += 1
            self.p['fnames'].pop(self.p['fnames'].index('Ensemble'))
            del self.session['results']['ensemble']
            
        self.tkvars['fnames'].set(self.p['fnames'])
        for i in range(len(self.p['fnames'])):
            try:
                scale_dist = self.session['results'][f'frame_{i}'].attrs['scale_dist']
                scale_vel = self.session['results'][f'frame_{i}'].attrs['scale_vel']
                units = self.session['results'][f'frame_{i}'].attrs['units']
            except: # no calibration has been applied
                scale_dist = 1
                scale_vel  = 1
                units = ['px', 'dt']
            del self.session['results'][f'frame_{i}']     
            frame = self.session['results'].create_group(f'frame_{i}')
            frame.attrs['processed'] = False
            frame.attrs['scale_dist'] = scale_dist
            frame.attrs['scale_vel'] = scale_vel
            frame.attrs['units'] = units

        if n != 0:
            del self.session['images']['fnames']
            self.session['images'].create_dataset('fnames', data = self.p['fnames'])
        if update_plot:
            self.show(self.p['files_' + self.toggle][self.index])
        print('Cleared results for all frames')
    
    
    def plot_window_function(self):
        frame = tk.Toplevel(
            self,
            width = 600,
            height = 600
        )
        frame.attributes('-topmost', 'true')
        fig = Fig()
        fig_frame = ttk.Frame(frame)
        fig_frame.pack(side = 'left',
                       fill = 'both',
                       expand =True)
        fig_canvas = FigureCanvasTkAgg(
            fig,
            master = fig_frame
        )
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(
            side='left',
            fill='x',
            expand='True'
        )
        
        cmap = self.p['func_cmap'] 
        if self.p['reverse_func_cmap']:
            cmap = cmap + '_r'
        
        if self.p['window_weighting'] == 'gaussian':
            weight = get_window(
                ('gaussian', self.p['window_weighting_sigma']), 
                self.p['window_size_func'])
        else:
            weight = get_window(
                self.p['window_weighting'], 
                self.p['window_size_func'])
        
        if self.p['plot_func_3d']:
            weight = np.outer(weight, weight)
            ax = fig.add_subplot(111, projection = '3d')
            x, y = np.mgrid[0:weight.shape[1]:1, 0:weight.shape[0]:1]
            ax.plot_surface(
                x, y, 
                weight, 
                cmap=cmap
            )
            ax.set_title(self.p['window_weighting'] + ' window function')
        else:
            ax = fig.add_subplot(111)
            ax.plot(weight)
            ax.set_title(self.p['window_weighting'] + ' window function')
        fig.canvas.draw()
        
    
    def initiate_img_intensity_plot(self):
        if self.inten_frame != None:
            if self.inten_frame.winfo_exists() == 1:
                raise Exception('Intensity viewer already exists')
        
        self.inten_frame = tk.Toplevel(
            self,
            #width = 450,
            height = 600
        )
        self.inten_frame.attributes('-topmost', 'true')
        self.inten_frame.protocol("WM_DELETE_WINDOW", self.dummy_callback)
        self.inten_fig = Fig()
        fig_frame = ttk.Frame(self.inten_frame)
        fig_frame.pack(side='top',
                            fill='both',
                            expand='True')
        fig_canvas = FigureCanvasTkAgg(
            self.inten_fig,
            master=fig_frame
        )
        
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(
            side='top',
            fill='x',
            expand='True'
        )

        fig_canvas._tkcanvas.pack(
            side='top',
            fill='both',
            expand='True'
        )

        ttk.Button(
            self.inten_frame, 
            command = self.save_intensity_hist_values,
            text = 'Export intensity values',
            style = 'h12.TButton',
            width = 40
        ).pack(fill = 'x')
        
        ttk.Button(
            self.inten_frame, 
            command = self.terminate_inten_interact,
            text = 'Close',
            style = 'h12.TButton',
            width = 40
        ).pack(fill = 'x', side = 'bottom')
        
        self.xy_connect_inten = self.fig_canvas.mpl_connect(
            'button_press_event', 
             self.plot_intensity_values
        )
        print('Initialized pixel intensity viewer')
        
        self.img_inten = None
        
    
    def plot_intensity_values(self, event):
        if event.inaxes is not None:
            self.get_settings() # incase 3d plot is enabled
            self.inten_fig.clear()
            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
            img = piv_tls.imread(self.p[f'files_{self.toggle}'][self.index])
            ax = self.inten_fig.add_subplot(111)
            
    
            self.inten_fig.canvas.draw()
            print('Updated pixel intensity histogram')
            
            
    def save_intensity_hist_values(self):
        filename = filedialog.asksaveasfilename(
                title = 'Save histogram intensity values',
                defaultextension = '.txt',
                filetypes = [('text file', '*.txt'), ]
        )
        if len(filename) > 1:
            np.savetxt(
                filename,
                self.img_inten,
                delimiter = ' '
            )
            
            
    def terminate_inten_interact(self):
        self.disconnect(self.xy_connect_inten)
        tk.Tk.destroy(self.inten_frame)
        del self.img_inten
    
    
    def initiate_correlation_plot(self):
        if self.corr_frame != None: 
            if self.corr_frame.winfo_exists() == 1:
                raise Exception('Correlation viewer already exists')
        self.corr_frame = tk.Toplevel(
            self,
            #width = 450,
            height = 600
        )
        self.corr_frame.attributes('-topmost', 'true')
        self.corr_frame.protocol("WM_DELETE_WINDOW", self.dummy_callback)
        self.corr_fig = Fig(figsize = (3,3))
        corr_fig_frame = ttk.Frame(self.corr_frame)
        corr_fig_frame.pack(side='top',
                            fill='both',
                            expand='True')
        corr_fig_canvas = FigureCanvasTkAgg(
            self.corr_fig,
            master=corr_fig_frame
        )
        
        corr_fig_canvas.draw()
        corr_fig_canvas.get_tk_widget().pack(
            side='top',
            fill='x',
            expand='True'
        )

        corr_fig_canvas._tkcanvas.pack(
            side='top',
            fill='both',
            expand='True'
        )

        fig_toolbar = NavigationToolbar2Tk(
            corr_fig_canvas,
            corr_fig_frame
        )
        
        fig_toolbar.update()  

        ttk.Button(
            self.corr_frame, 
            command = self.save_corr_sect,
            text = 'Export correlation plane',
            style = 'h12.TButton',
            width = 40
        ).pack(fill = 'x')
        
        ttk.Button(
            self.corr_frame, 
            command = self.terminate_corr_interact,
            text = 'Close',
            style = 'h12.TButton',
            width = 40
        ).pack(fill = 'x', side = 'bottom')
        
        self.corr_sect = None
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'Frame:').pack(side = 'left')
        self.corr_frame_num = ttk.Label(f, text = 'N/A')
        self.corr_frame_num.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'X ['+str(self.units[0])+']:').pack(side = 'left')
        self.corr_x = ttk.Label(f, text = 'N/A')
        self.corr_x.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'Y ['+str(self.units[0])+']:').pack(side = 'left')
        self.corr_y = ttk.Label(f, text = 'N/A')
        self.corr_y.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'U ['+str(self.units[1])+']:').pack(side = 'left')
        self.corr_u = ttk.Label(f, text = 'N/A')
        self.corr_u.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'V ['+str(self.units[1])+']:').pack(side = 'left')
        self.corr_v = ttk.Label(f, text = 'N/A')
        self.corr_v.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'index:').pack(side = 'left')
        self.corr_index = ttk.Label(f, text = 'N/A')
        self.corr_index.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'max corr height:').pack(side = 'left')
        self.corr_peak_height = ttk.Label(f, text = 'N/A')
        self.corr_peak_height.pack(side = 'right')
        f.pack(anchor = 'w')
        
        f = tk.Frame(self.corr_frame)
        ttk.Label(f, text = 'peak to mean ratio:').pack(side = 'left')
        self.corr_peak2mean = ttk.Label(f, text = 'N/A')
        self.corr_peak2mean.pack(side = 'right')
        f.pack(anchor = 'w')
        
        self.xy_connect_corr = self.fig_canvas.mpl_connect(
            'button_press_event', 
             self.plot_correlation_plane
        )
        print('Initialized correlation viewer')
        
        
    def plot_correlation_plane(self, event):
        if event.inaxes is not None:
            self.get_settings() # incase 3d plot is enabled
            self.corr_fig.clear()
            px, py = event.inaxes.transData.inverted().transform((event.x, event.y))
            x, y = self.results[1], self.results[2]
            corr = np.array(self.session['results'][f'frame_{self.index}']['corr'])
            indx = np.abs(x[0,:] / px - 1) # find index from closest distance to cursor point
            indy = np.abs(y[:,0] / py - 1)
            indx = int(np.where(indx == indx.min())[0])
            indy = int(np.where(indy == indy.min())[0])
            self.corr_sect = corr[indy*x.shape[1]+indx, :, :]
            self.corr_frame_num.config(text = str(self.index))
            self.corr_index.config(text = f'({indx}, {indy})')
            self.corr_x.config(text = str(x.reshape(-1)[indy*x.shape[1]+indx]))
            self.corr_y.config(text = str(y.reshape(-1)[indy*x.shape[1]+indx]))
            self.corr_u.config(text = str(self.results[3].reshape(-1)[indy*x.shape[1]+indx]))
            self.corr_v.config(text = str(self.results[4].reshape(-1)[indy*x.shape[1]+indx]))
            self.corr_peak_height.config(text = str(self.corr_sect.max()))
            self.corr_peak2mean.config(text = str(self.corr_sect.max() / np.absolute(self.corr_sect).mean()))
            cmap = self.p['func_cmap'] 
            if self.p['reverse_func_cmap']:
                cmap = cmap + '_r'
            if self.p['plot_func_3d']:
                ax = self.corr_fig.add_subplot(111, projection = '3d')
                x, y = np.mgrid[0:self.corr_sect.shape[0]:1, 0:self.corr_sect.shape[1]:1]
                ax.plot_surface(
                    x, y, 
                    self.corr_sect, 
                    cmap=cmap
                )
            else:
                ax = self.corr_fig.add_subplot(111)
                ax.imshow(self.corr_sect, cmap = cmap)
    
            self.corr_fig.canvas.draw()
            print('Updated correlation plot')
            
            
    def save_corr_sect(self):
        filename = filedialog.asksaveasfilename(
                title = 'Save correlation matix',
                defaultextension = '.txt',
                filetypes = [('text file', '*.txt'), ]
        )
        if len(filename) > 1:
            np.savetxt(
                filename,
                self.corr_sect,
                delimiter = ' '
            )
            
            
    def terminate_corr_interact(self):
        self.disconnect(self.xy_connect_corr)
        tk.Tk.destroy(self.corr_frame)
        del self.corr_sect
        
    
    def plot_scatter(self):
        self.scatter_frame = tk.Toplevel(
            self,
            width = 600,
            height = 600
        )
        self.scatter_frame.attributes('-topmost', 'true')
        self.scatter_fig = Fig()
        scatter_fig_frame = ttk.Frame(self.scatter_frame)
        scatter_fig_frame.pack(side='left',
                            fill='both',
                            expand='True')
        scatter_fig_canvas = FigureCanvasTkAgg(
            self.scatter_fig,
            master=scatter_fig_frame
        )
        
        scatter_fig_canvas.draw()
        scatter_fig_canvas.get_tk_widget().pack(
            side='left',
            fill='x',
            expand='True'
        )
        
        scatter_fig_toolbar = NavigationToolbar2Tk(
            scatter_fig_canvas,
            scatter_fig_frame
        )
        
        scatter_fig_toolbar.update()  
        scatter_fig_canvas._tkcanvas.pack(
            side='top',
            fill='both',
            expand='True'
        )
        #self.fig_canvas.mpl_connect(
        #    "key_press_event",
        #    lambda: key_press_handler(
        #        event,
        #        self.fig_canvas,
        #        self.fig_toolbar
        #    )
        #)
        self.update_scatter_plt()
        
        
    def update_scatter_plt(self):
        x, y, u, v = self.results[1:5]
        self.scatter_fig.clear()
        vec_plot.scatter(
            [x,y,u,v],
            self.scatter_fig,
            units = self.units,
            mask_coords = self.results[7],
            title = self.p['fnames'][self.index]
        )
        self.scatter_fig.canvas.draw()
        print("Updated scatter plot frame.")
        
        
    def plot_histogram(self):
        self.hist_frame = tk.Toplevel(
            self,
            width = 600,
            height = 600
        )
        self.hist_frame.attributes('-topmost', 'true')
        self.hist_fig = Fig()
        hist_fig_frame = ttk.Frame(self.hist_frame)
        hist_fig_frame.pack(side='left',
                            fill='both',
                            expand='True')
        hist_fig_canvas = FigureCanvasTkAgg(
            self.hist_fig,
            master=hist_fig_frame
        )
        
        hist_fig_canvas.draw()
        hist_fig_canvas.get_tk_widget().pack(
            side='left',
            fill='x',
            expand='True'
        )
        
        fig_toolbar = NavigationToolbar2Tk(
            hist_fig_canvas,
            hist_fig_frame
        )
        
        fig_toolbar.update()  
        hist_fig_canvas._tkcanvas.pack(
            side='top',
            fill='both',
            expand='True'
        )
        #self.fig_canvas.mpl_connect(
        #    "key_press_event",
        #    lambda: key_press_handler(
        #        event,
        #        self.fig_canvas,
        #        self.fig_toolbar
        #    )
        #)
        self.update_histogram_plt()
        
        
    def update_histogram_plt(self):
        x, y, u, v = self.results[1:5]
        self.hist_fig.clear()
        vec_plot.hist2(
            [x,y,u,v],
            self.p,
            self.hist_fig,
            units = self.units,
            title = self.p['fnames'][self.index],
            mask_coords = self.results[7]
        )
        self.hist_fig.canvas.draw()
        print("Updated histogram plot frame")
    
    
    def plot_statistics_table(self, overide = False):
        self.stats_frame = tk.Toplevel(
            self,
            width = 600,
            height = 600
        )
        self.stats_frame.attributes('-topmost', 'true')
        column_names = [
            'Component', 'Units', 'STD', 'Min', 'Mean', 'Max'
        ]
        components = [
            'u-component', 'v-component', 'magnitude', 
            'vorticity', 'enstrophy', 'shear strain', 'normal strain'
        ]
        f = ttk.Frame(self.stats_frame)
        for name in column_names:
            if name == 'Component':
                width = 20
            else:
                width = 15
            e = ttk.Entry(
                f, 
                width=width, 
                justify = 'center',
            )
            e.insert(tk.END, name)
            e.config(state = 'disabled')
            e.pack(side = 'left', anchor = 'nw')
        f.pack(side = 'top', anchor = 'w')
        self.stats = {}
        for component in components:
            f = ttk.Frame(self.stats_frame)
            for name in column_names:
                
                if name == 'Component':
                    width = 20
                else:
                    width = 15 
                self.stats[component + '_' + name] = ttk.Entry(
                    f,
                    width = width,
                    justify = 'center'
                )
                if name == 'Component':
                    self.stats[component + '_' + name].insert(tk.END, component)
                self.stats[component + '_' + name].config(state = 'disabled')
                self.stats[component + '_' + name].pack(side = 'left')
            f.pack(side = 'top', anchor = 'w')
        self.update_statistics()

        
    def update_statistics(self):
        units_d, units_t = self.units
        x, y, u, v = self.results[1:5]
        mask_coords = self.results[7]
        if len(mask_coords) > 0:
            mask = coords_to_xymask(x, y, mask_coords)
        else:
            mask = np.ma.nomask
        for stat in self.stats:
            self.stats[stat].config(state = 'normal')
        for component in ['u-component', 'v-component', 'magnitude', 
                          'vorticity', 'enstrophy', 'shear strain', 'normal strain']:
            if component in ['u-component', 'v-component', 'magnitude']:
                units = units_d + '/' + units_t
            elif component in ['vorticity', 'shear strain', 'normal strain']:
                units = '1/' + units_t
            elif component == 'enstrophy':
                units = units_d+'^2/' + units_t+'^2'
            comp = vec_plot.get_component(x, y, u, v, component = component) 
            comp = np.ma.masked_array(comp, mask = mask)
            c_std = _round(np.nanstd(comp), 4)
            c_min = _round(np.nanmin(comp), 4)
            c_mean = _round(np.nanmean(comp), 4)
            c_max = _round(np.nanmax(comp), 4)
            self.stats[component + '_Units'].delete(0, tk.END)
            self.stats[component + '_STD'].delete(0, tk.END)
            self.stats[component + '_Min'].delete(0, tk.END)
            self.stats[component + '_Mean'].delete(0, tk.END)
            self.stats[component + '_Max'].delete(0, tk.END)
            self.stats[component + '_Units'].insert(tk.END, units)
            self.stats[component + '_STD'].insert(tk.END, c_std)
            self.stats[component + '_Min'].insert(tk.END, c_min)
            self.stats[component + '_Mean'].insert(tk.END, c_mean)
            self.stats[component + '_Max'].insert(tk.END, c_max)
        for stat in self.stats:
            self.stats[stat].config(state = 'disabled')
        print('Updated statistics table')
        
        
    def dummy_callback(self):
        print('Please use »Close« button')
    
    
    def initialize_dist_interact(self):
        pass
        print('Initialized interactive point selection')
    
    
    def initiate_area_interact(self):
        pass
        print('Initialized interactive area selection')
    
    
    def show(self, fname, extFig = None, bypass = False, preview = False, preproc = True, show_results=True,
             results = None, show_mask = True, perform_check = True, ignore_blank = False):
        '''Display a file.

        This method distinguishes vector data (file extensions
        txt, dat, jvc,vec and csv) and images (all other file extensions).

        Parameters
        ----------
        fname : str
            A filename.
        '''
        self.get_settings()
        if extFig == None:
            fig = self.fig
            fig.clear()
        else:
            fig = extFig
        ax = fig.add_axes([0,0,1,1])
        #ax = fig.add_subplot(111)
        if self.rasterize:
            ax.set_rasterized(True)

        if bypass != True:
            if perform_check:
                self.check_exclusion(ignore_blank = ignore_blank)
            if results == None:
                results = self.session['results'][f'frame_{self.index}']
                
            processed_frame = results.attrs['processed']
            if processed_frame: 
                x = np.array(results['x']) + results.attrs['offset_x']
                y = np.array(results['y']) + results.attrs['offset_y']
                s2n = np.array(results['s2n'])
                if 'u_vld' in results:
                    if self.p['debug']:
                        print('Show: Loaded validated vectors')
                    u = np.array(results['u_vld'])
                    v = np.array(results['v_vld'])
                    tp = np.array(results['tp_vld']) 
                if 'u_mod' in results:
                    if self.p['debug']:
                        print('Show: Loaded modified vectors')
                    u = np.array(results['u_mod'])
                    v = np.array(results['v_mod'])
                    if 'u_vld' in results:
                        tp = np.array(results['tp_vld']) 
                    else:
                        tp = np.array(results['tp']) 
                if 'u_vld' not in results and 'u_mod' not in results:
                    if self.p['debug']:
                        print('Show: Loaded unprocessed vectors')
                    u = np.array(results['u'])
                    v = np.array(results['v'])
                    tp = np.array(results['tp'])  
                u_or, v_or = u, v
                x = x.astype(np.int)
                y = y.astype(np.int)
                mask_coords  = list(literal_eval(results.attrs['mask_coords']))
                roi_present  = results.attrs['roi_present']                             
                roi_coords   = results.attrs['roi_coords']
                process_time = results.attrs['process_time']
            
                self.units      = results.attrs['units']
                self.scale_dist = results.attrs['scale_dist']
                self.scale_vel  = results.attrs['scale_vel']
                    
                if roi_present:
                    if self.p['debug']:
                        print('Show: Found region of interest')
                    self.xlim = [int(roi_coords[0]), int(roi_coords[1])]
                    self.ylim = [int(roi_coords[2]), int(roi_coords[3])]
                    x += int(roi_coords[0])
                    y += int(roi_coords[2])
                else:
                    self.xlim = [None, None]
                    self.ylim = [None, None]
                    
                u, v = u / self.scale_vel * self.scale_dist, v /  self.scale_vel * self.scale_dist
                
                self.results = [
                    process_time, #processed_frame,
                    x, y, u, v,
                    tp, s2n,
                    mask_coords,
                    roi_present,
                    roi_coords,
                    process_time,
                    self.scale_dist,
                    self.scale_vel,                        
                ] 
                if self.p['debug']:
                    print('Show: Saved current results')
                if self.stats_frame != None and\
                        self.stats_frame.winfo_exists() == 1:
                    self.update_statistics()
                if self.scatter_frame != None and\
                        self.scatter_frame.winfo_exists() == 1:
                    self.update_scatter_plt()
                if self.hist_frame != None and\
                        self.hist_frame.winfo_exists() == 1:
                    self.update_histogram_plt()
                
            else:
                roi_coords = ['', '', '', '']
                mask_coords = []
                self.xlim = [None, None]
                self.ylim = [None, None]
                #self.units = ['px', 'dt']
                #self.scale_dist  = 1
                #self.scale_vel   = 1
                self.units      = results.attrs['units']
                self.scale_dist = results.attrs['scale_dist']
                self.scale_vel  = results.attrs['scale_vel']

            if processed_frame == True and show_results == True:
                self.calculate_statistics(self.results)
                xmin = roi_coords[0]
                xmax = roi_coords[1]
                ymin = roi_coords[2]
                ymax = roi_coords[3]
                if self.p['autoscale_vectors']:
                    autoden = np.hypot(
                        x.max(), y.max()
                    )
                    delx = (x.min() - x.max())/x.shape[1] # following 3 lines were borrowed from PIVlab
                    dely = (y.min() - y.max())/x.shape[0]
                    del_ = delx**2 + dely**2
                    meanLen = np.nanmean(
                        ((u_or**2 + v_or**2)/del_)**.5
                    )
                    scale = 1/meanLen
                    if self.p['nthArrX'] == self.p['nthArrY']:
                        skip = self.p['nthArrX']
                    else:
                        skip = 1 # too lazy to add on...
                    scale = autoden / scale / skip
                    
                    try:
                        disp_scale = _round(scale, 1)
                    except: disp_scale = scale
                    
                    self.tkvars['vec_scale'].set(disp_scale)
                    if self.p['debug']:
                        print('Show: Calculated scale and width for vectors')
                else:
                    scale = self.p['vec_scale']
                scale = scale / self.scale_vel * self.scale_dist
                    
                if self.p['autowidth_vectors']:
                    dx = x[0, :][-1] - x[0, :][0]
                    dy = y[:, 0][-1] - y[:, 0][0]
                    d = (dx**2 + dy**2)** 0.5
                    width = _round((1/d), 5)
                    self.tkvars['vec_width'].set(width)
                else:
                    width = self.p['vec_width']
                
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                try:
                    img = self.show_img(
                        fname,
                        ax, 
                        preproc = True,
                        preview = False, 
                        roi_coords = ['', '', '', ''], 
                        show_mask = False,
                        mask_coords = [],
                        plot = False)
                    alpha = 1
                except Exception as e:
                    print("Could not read image file\nReason: " + str(e))
                    img = np.zeros((y.max() + y.min(), x.max() + x.min()))
                    self.img_shape = img.shape
                    alpha = 0
                ax.matshow(img, 
                      cmap=plt.cm.gray,
                      alpha = alpha,
                      vmax=self.p['matplot_intensity_max'],
                      vmin=self.p['matplot_intensity_min'],)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim([0, img.shape[1]])
                ax.set_ylim([0, img.shape[0]])

                for ax in fig.get_axes():
                    ax.invert_yaxis()

                if len(mask_coords) > 0:
                    xymask = coords_to_xymask(x, y, mask_coords)
                else:
                    xymask = np.ma.nomask
                        
                if self.p['contours_show']:
                    borders = np.min(x[0, :]), np.max(x[0, :]), np.min(y[:, 0]), np.max(y[:, 0])
                        
                    vec_plot.contour(
                        [x, y], 
                        self.p,
                        fig,
                        ax,
                        color_values = vec_plot.get_component(
                            x, y, u, -v, self.p['velocity_color']
                        ),
                        borders = borders,
                        mask = xymask,
                        )
                    if self.p['debug']:
                        print('Show: Generated contours')
        
                if self.p['streamlines_show']:
                    vec_plot.streamlines(
                        [x, y, u, -v],
                        ax,
                        self.p,
                        mask = xymask,
                    )
                    if self.p['debug']:
                        print('Show: Generated streamlines')
                    
                add_disp_mask(
                    ax,
                    mask_coords,
                    invert = False,
                    color = self.p['mask_fill'],
                    alpha = self.p['mask_alpha']
                )
                if self.p['debug']:
                    print('Show: Generated mask object')
                if self.p['vectors_show']:
                    vec_plot.vector(
                        [x, y, u, v, tp],
                        fig,
                        ax,
                        self.p,
                        mask_coords = mask_coords,
                        scale = scale,
                        width = width,
                    )
                    if self.p['debug']:
                        print('Show: Generated vectors')

                if self.p['show_colorbar']:
                    try: vmin = float(self.p['vmin'])
                    except: vmin = None
                    try: vmax = float(self.p['vmax'])
                    except: vmax = None
                            
                    vec_plot.plot_colorbar(
                        fig,
                        vec_plot.get_component(
                            x, y, u, -v, self.p['velocity_color']
                        ),
                        #cbaxis = ax,
                        cmap = vec_plot.get_cmap(self.p['color_map']),
                        vmin = vmin,
                        vmax = vmax,
                    )
                try:     
                    if xmin and xmax and ymin and ymax != ('', ' '):
                        add_disp_roi(
                            ax,
                            int(xmin), int(ymin), int(xmax), int(ymax),
                            linewidth = self.p['roi_border_width'],
                            edgecolor = self.p['roi_border'],
                            linestyle = self.p['roi_line_style']
                        )
                    if self.p['debug']:
                        print('Show: Generated roi display')
                except Exception as e:
                    print('Ignoring roi exclusion objects \nReason: ' + str(e))

            else:
                if self.p['debug']:
                    print('Show: Bypassed all vector plotting functions')
                if fname != 'none':
                    self.img_shape = piv_tls.imread(fname).shape
                    self.show_img(fname, 
                                  axes = ax,
                                  preview = preview,
                                  roi_coords = self.session['images']['settings'][f'frame_{self.index}'].attrs['roi_coords'],
                                  preproc = preproc,
                                  show_mask = show_mask,
                                  mask_coords = list(
                                      literal_eval(
                                          self.session['images']['settings'][f'frame_{self.index}'].attrs['mask_coords']
                                      )
                                  )
                    )
        if bypass == True and fname != 'none':
            if self.p['debug']:
                print('Show: Bypassed all plotting functions')
            self.show_img(
                fname, 
                axes = ax,
                preview = preview,
                preproc = preproc,
                show_mask = False,
                mask_coords = []
            )
        if extFig == None:
            self.ax = ax # used for GUI interactivity
            fig.canvas.draw()

        
    def show_img(self, 
                 fname,
                 axes, 
                 array = [],
                 preproc = True,
                 preview = False, 
                 roi_coords = ['', '', '', ''], 
                 show_mask = True,
                 mask_coords = [],
                 plot = True):
        '''Display an image.

        Parameters
        ----------
        fname : str
            Pathname of an image file.
        '''
        if len(array) == 0:
            img = piv_tls.imread(fname)
        else:
            img = array
        print('\nimage data type: {}'.format(img.dtype))
        if 'int' not in str(img.dtype):
            print('Warning: For PIV processing, ' +
                  'image will be compressed to [0,1] float32, processed, and converted to int16' +
                  '\nThis may cause a loss of precision')
            
        print('Processing image')
        #maxVal = img.max()
        #if img.max() > 2**8:
        #    maxVal = 2**16
        #else:
        #    maxVal = 2**8
        maxVal = img.max()
        img = img.astype(np.float32) # minimize compression loss
        img /= maxVal
        if len(self.background_frame_a) > 1:
            if self.toggle == 'a':
                bg = self.background_frame_a
            else:
                bg = self.background_frame_b
            img = self.preprocessing_methods['temporal_filters'](img, bg/maxVal, self.p)
            
        img = np.int16(self.process_disp_img(
            img = img,
            axes = axes,
            roi_coords = [
                roi_coords[0],
                roi_coords[1],
                roi_coords[2],
                roi_coords[3],
            ], 
            mask_coords = mask_coords,
            invert_mask = self.p['invert_mask'],
            preproc = preproc,
            preview = preview, 
            show_mask = show_mask
        )*2**8)
        print('Processed image')
        if plot == True:
            self.img_shape = img.shape
            axes.matshow(img, cmap=plt.cm.Greys_r,
                         vmax=self.p['matplot_intensity_max'],
                         vmin=self.p['matplot_intensity_min'],
                        )
            axes.xaxis.set_major_formatter(plt.NullFormatter())
            axes.yaxis.set_major_formatter(plt.NullFormatter())
            axes.set_xticks([])
            axes.set_yticks([])
            self.fig.canvas.draw()
        else:
            return img
    
    
    
    def process_disp_img(self, 
                         img,
                         axes,
                         roi_coords,
                         mask_coords,
                         invert_mask = False,
                         offset_x = 0,
                         offset_y = 0,
                         stretch_x = 0,
                         stretch_y = 0,
                         preview = False,
                         preproc = True,
                         show_mask = True):
        if self.p['apply_second_only'] == True:
            if self.toggle == 'b':
                if self.p['debug']:
                    print('Show_img: Performming transformation')
                img = self.preprocessing_methods['transformations'](img, self.p)
        else:
            if self.p['debug']:
                print('Show_img: Performing transformation')
            img = self.preprocessing_methods['transformations'](img, self.p)
                
        raw_img = self.preprocessing_methods['spatial_filters'](
            img, 
            self.p,
            preproc = False,
            preview = False,
            roi_xmin = '',
            roi_xmax = '',
            roi_ymin = '',
            roi_ymax = '',
        )  
        
        if preproc == True:
            img = raw_img.copy()
            try:
                xmin = roi_coords[0]
                xmax = roi_coords[1]
                ymin = roi_coords[2]
                ymax = roi_coords[3]
            
            except:
                xmin = ''
                xmax = ''
                ymin = ''
                ymax = ''
                
            if preview:
                if self.p['debug']:
                    print('Show_img: Generating display image')
                    
                if self.p['do_phase_separation'] == True:
                    if self.p['debug']:
                        print('Show_img: Performing phase separation')
                    img = self.preprocessing_methods['phase_separation'](img, self.p)

                    
            img = self.preprocessing_methods['spatial_filters'](
                img, 
                self.p,
                preproc = True,
                preview = preview,
                roi_xmin = xmin,
                roi_xmax = xmax,
                roi_ymin = ymin,
                roi_ymax = ymax,
            )
            
            try:
                if xmin and xmax and ymin and ymax != ('', ' '):
                    xmin, xmax = int(xmin), int(xmax)
                    ymin, ymax = int(ymin), int(ymax)

                    raw_img[ymin:ymax, xmin:xmax] = img
                    img = raw_img 
                    add_disp_roi(axes, 
                                 xmin, ymin, xmax, ymax,
                                 linewidth = self.p['roi_border_width'],
                                 edgecolor = self.p['roi_border'],
                                 linestyle = self.p['roi_line_style'])
                if show_mask:
                    add_disp_mask(
                        axes,
                        mask_coords,
                        invert = invert_mask,
                        color = self.p['mask_fill'],
                        alpha = self.p['mask_alpha']
                    )
            except Exception as e:
                print('Ignoring image exclusion preview \nReason: '+str(e))

        else:
            img = raw_img
        return(img)
            
                        
    def destroy(self):
        '''Destroy the OpenPIV GUI.

        Settings are automatically saved.
        '''
        if messagebox.askyesno('Exit Manager', 'Are you sure you want to exit?'):
            
            self.get_settings()
            if self.p['save_on_exit']:
                print('Saving settings.')
                self.p.dump_settings(self.p.params_fname)
            print('Closing session HDF5 file..')
            self.session.close()
            print('Closed session HDF5 file.')
            tk.Tk.quit(self)
            tk.Tk.destroy(self)
            # sometimes the GUI closes, but the main thread still runs
            print('Destorying main thread')
            sys.exit()
            print('Destoryed main thread.') # This should not execute if the thread is destroyed. 
                                            # Could cause possible issue in the future.

if __name__ == '__main__':
    openPivGui = OpenPivGui()
    openPivGui.geometry("1200x750") # a good starting size for the GUI
    openPivGui.mainloop()
