#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Parallel Processing of PIV images.'''
import os
os.environ['OPENBLAS_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

from openpivgui.open_piv_gui_tools import _round, coords_to_xymask, normalize_array

import multiprocessing
import numpy as np
import time
try:
    import pyfftw as FFTW
except: pass

import openpiv.windef as piv_wdf
import openpiv.pyprocess as piv_prc
import openpiv.preprocess as piv_pre
import openpiv.tools as piv_tls

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


class MultiProcessing(piv_tls.Multiprocesser):
    '''Parallel processing, based on the corrresponding OpenPIV class.

    Do not run from the interactive shell or within IDLE! Details at:
    https://docs.python.org/3.6/library/multiprocessing.html#using-a-pool-of-workers

    Parameters
    ----------
    params : OpenPivParams
        A parameter object.
    '''

    def __init__(
        self,
        params,
        settings = None, 
        session = None,
        files_a = None,
        files_b = None,
        disp_off = 0,
        bg_a = [],
        bg_b = [],
        ensemble = False,
        parallel = False, 
    ):
        '''Standard initialization method.
        For parallel, this class can be used on its own.
        '''
        self.parameter = params.p
        self.preproc = params.preprocessing_methods
        self.postproc = params.postprocessing_methods
        self.settings = settings
        self.session = session
        self.files_a = files_a
        self.files_b = files_b
        self.bg_a = bg_a
        self.bg_b = bg_b
        self.disp_off = disp_off
        self.parallel = parallel
        self.file_path = params.path
        self.rfft_plans = {}
        self.irfft_plans = {}
        print('Generating FFT object')
        self.parameter['pass_1'] = True
        try: # example of using different fft libraries, is also done inefficiently
            if self.parameter['use_FFTW'] != True or self.parallel == True:
                raise Exception('Disabled FFTW via exception')
            frame_a = piv_tls.imread(self.files_a[0])
            #FFTW.config.NUM_THREADS = os.cpu_count() 
            FFTW.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
            for i in range(1, 7):
                if self.parameter[f'pass_{i}'] == True:
                    # setting up the windowing of each pass
                    try:
                        corr_window = [int(self.parameter[f'corr_window_{i}']),
                                       int(self.parameter[f'corr_window_{i}'])]
                    except:
                        corr_window = [
                                int(list(self.parameter[f'corr_window_{i}'].split(','))[0]),
                                int(list(self.parameter[f'corr_window_{i}'].split(','))[1])
                        ]
                        
                    if self.parameter['update_overlap'] == False or i == 1:
                        try:
                            overlap = [int(self.parameter[f'overlap_window_{i}']),
                                       int(self.parameter[f'overlap_window_{i}'])]
                        except:
                            overlap = [
                                int(list(self.parameter[f'overlap_window_{i}'].split(','))[0]),
                                int(list(self.parameter[f'overlap_window_{i}'].split(','))[1])
                            ]
                        overlap_percent = [overlap[0] / corr_window[0], overlap[1] / corr_window[1]]
                    else:
                        overlap = [int(corr_window[0] * overlap_percent[0]),
                                   int(corr_window[1] * overlap_percent[1])]
                        
                    n_rows, n_cols = piv_prc.get_field_shape(
                        frame_a.shape, 
                        corr_window, 
                        overlap
                    )
                    
                    s = np.array([corr_window, corr_window])
                    if self.parameter['corr_method'] == 'linear':
                        size = s[0] + s[1] - 1
                        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
                    else:
                        fsize = None
                    
                    if i == 1:
                        global rfft2_1
                        fft_dtype = 'float16'
                        fft_mem_1 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_1.shape}')
                        def rfft2_1(aa, s=None, axes = (-2,-1)):
                            fft_forward_1 = FFTW.builders.rfft2(fft_mem_1, fsize, axes = (-2,-1))
                            return fft_forward_1(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_1
                    elif i == 2:
                        global rfft2_2
                        fft_mem_2 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_2.shape}')
                        fft_forward_2 = FFTW.builders.rfft2(fft_mem_2, fsize, axes = (-2,-1))
                        def rfft2_2(aa, s=None, axes = (-2,-1)):
                            return fft_forward_2(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_2
                    elif i == 3:
                        global rfft2_3
                        fft_mem_3 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_3.shape}')
                        fft_forward_3 = FFTW.builders.rfft2(fft_mem_3, fsize, axes = (-2,-1))
                        def rfft2_3(aa, s=None, axes = (-2,-1)):
                            return fft_forward_3(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_3
                    elif i == 4:
                        global rfft2_4
                        fft_mem_4 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_4.shape}')
                        fft_forward_4 = FFTW.builders.rfft2(fft_mem_4, fsize, axes = (-2,-1))
                        def rfft2_4(aa, s=None, axes = (-2,-1)):
                            return fft_forward_4(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_4
                    elif i == 5:
                        global rfft2_5
                        fft_mem_5 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_5.shape}')
                        fft_forward_5 = FFTW.builders.rfft2(fft_mem_5, fsize, axes = (-2,-1))
                        def rfft2_5(aa, s=None, axes = (-2,-1)):
                            return fft_forward_5(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_5
                    elif i == 6:
                        global rfft2_6
                        fft_mem_6 = FFTW.empty_aligned(
                            [n_rows*n_cols, corr_window[0], corr_window[1]], 
                            dtype = fft_dtype
                        )
                        print(f'Pass {i} RFFT object shape: {fft_mem_6.shape}')
                        fft_forward_6 = FFTW.builders.rfft2(fft_mem_6, fsize, axes = (-2,-1))
                        def rfft2_6(aa, s=None, axes = (-2,-1)):
                            return fft_forward_6(aa)
                        self.rfft_plans[f'pass_{i}'] = rfft2_6
                    
                    if self.parameter['corr_method'] == 'linear':
                        n = 2
                    else:
                        n = 1
                        
                    ifft_dtype = 'complex64'
                    if i == 1:
                        global irfft2_1
                        ifft_mem_1 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_1).shape,
                            dtype = ifft_dtype
                        )
                        print(f'Pass {i} IRFFT object shape: {ifft_mem_1.shape}')
                        fft_backward_1 = FFTW.builders.irfft2(ifft_mem_1, axes = (-2,-1))
                        def irfft2_1(aa, s=None, axes = (-2,-1)):
                            return fft_backward_1(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_1
                    elif i == 2:
                        global irfft2_2
                        ifft_mem_2 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_2).shape,
                            dtype = ifft_dtype
                        )
                        print(f'Pass {i} IRFFT object shape: {ifft_mem_2.shape}')
                        fft_backward_2 = FFTW.builders.irfft2(ifft_mem_2, axes = (-2,-1))
                        def irfft2_2(aa, s=None, axes = (-2,-1)):
                            return fft_backward_2(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_2
                    elif i == 3:
                        global irfft2_3
                        ifft_mem_3 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_3).shape,
                            dtype = ifft_dtype
                        )
                        print(f'Pass {i} IRFFT object shape: {ifft_mem_3.shape}')
                        fft_backward_3 = FFTW.builders.irfft2(ifft_mem_3, axes = (-2,-1))
                        def irfft2_3(aa, s=None, axes = (-2,-1)):
                            return fft_backward_3(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_3
                    elif i == 4:
                        global irfft2_4
                        ifft_mem_4 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_4).shape,
                            dtype = ifft_dtype
                        )
                        print(f'Pass {i} IRFFT object shape: {ifft_mem_4.shape}')
                        fft_backward_4 = FFTW.builders.irfft2(ifft_mem_4, axes = (-2,-1))
                        def irfft2_4(aa, s=None, axes = (-2,-1)):
                            return fft_backward_4(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_4
                    elif i == 5:
                        global irfft2_5
                        ifft_mem_5 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_5).shape,
                            dtype = ifft_dtype
                        )
                        print(f'Pass {i} IRFFT object shape: {ifft_mem_5.shape}')
                        fft_backward_5 = FFTW.builders.irfft2(ifft_mem_5, axes = (-2,-1))
                        def irfft2_5(aa, s=None, axes = (-2,-1)):
                            return fft_backward_5(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_5
                    elif i == 6:
                        global irfft2_6
                        ifft_mem_6 = FFTW.empty_aligned(
                            #[n_rows * n_cols, corr_window[0]*n, int(corr_window[1]/2 * n) + 1],
                            self.rfft_plans[f'pass_{i}'](fft_mem_6).shape,
                            dtype = ifft_dtype
                        )
                        fft_backward_6 = FFTW.builders.irfft2(ifft_mem_6, axes = (-2,-1))
                        def irfft2_6(aa, s=None, axes = (-2,-1)):
                            return fft_backward_6(aa)
                        self.irfft_plans[f'pass_{i}'] = irfft2_6

                else:
                    break;
            print('Using python wrapper of FFTW library')
        #'''
        except:
            from scipy.fft import rfft2, irfft2
            for i in range(1, 7):
                if self.parameter[f'pass_{i}'] == True:
                    self.rfft_plans[f'pass_{i}'] = rfft2
                    self.irfft_plans[f'pass_{i}'] = irfft2
                else:
                    break;
            print('Using PocketFFT library')
        #'''
        if files_a != None:
            #if self.parameter['swap_files']:
            #    self.files_a, self.files_b = self.files_b, self.files_a

            self.n_files = len(self.files_a)
                
    
    def process(self, args):
        '''Process chain as configured in the GUI.

        Parameters
        ----------
        args : tuple
            Tuple as expected by the inherited run method:
            file_a (str) -- image file a
            file_b (str) -- image file b
            counter (int) -- index pointing to an element of the filename list
        '''
        file_a, file_b, counter = args
        frame_a = piv_tls.imread(file_a)
        frame_b = piv_tls.imread(file_b)

        s = self.parameter['smoothn_val1']
        if self.parameter['analysis'] != True:
            raise Exception('Cancled analysis via exception')
                
        # preprocessing
        print('\nPre-pocessing frame: {}'.format(counter + self.disp_off))
        
        roi_coords = self.settings[f'{counter}'][0]
        mask_coords = self.settings[f'{counter}'][1]
        mask_c = mask_coords.copy()
        try:
            roi_xmin = int(roi_coords[0])
            roi_xmax = int(roi_coords[1])
            roi_ymin = int(roi_coords[2])
            roi_ymax = int(roi_coords[3])
        except:
            roi_xmin = 0
            roi_ymin = 0
            roi_ymax, roi_xmax = frame_a.shape
            
        for i in range(len(mask_coords)):
            temp = np.array(mask_coords[i])
            temp[:,0] -= roi_xmin
            temp[:,1] -= roi_ymin
            mask_coords[i] = temp

        maxVal = frame_a.max()
        frame_a = frame_a.astype('float32')
        frame_a /= maxVal
        if len(self.bg_a) > 1:
            print('Removing background for image 1')
            frame_a = self.preproc['temporal_filters'](frame_a, self.bg_a/maxVal, self.parameter)
        if self.parameter['apply_second_only'] != True:
            print('Transforming image 1')
            frame_a = self.preproc['transformations'](frame_a, self.parameter)
        if self.parameter['do_phase_separation'] == True:
            print('Separating phases for image 1')
            frame_a = self.preproc['phase_separation'](frame_a, self.parameter)
                
        print('Pre-processing image 1')
        frame_a = self.preproc['spatial_filters']( # float32 takes less space for FFTs
                frame_a, 
                self.parameter,
                preproc            = True,
                roi_xmin           = roi_xmin,
                roi_xmax           = roi_xmax,
                roi_ymin           = roi_ymin,
                roi_ymax           = roi_ymax,
                )*2**8
        
        if len(mask_coords) >= 1:
            print('Applying mask to image 1')
            frame_a = self.preproc['apply_mask'](frame_a, mask_coords, self.parameter)
            
        frame_a = frame_a.astype('float32')    
        
        maxVal = frame_b.max()
        frame_b = frame_b.astype('float32')
        frame_b /= maxVal
        
        if len(self.bg_a) > 1:
            print('Removing background for image 2')
            frame_b = self.preproc['temporal_filters'](frame_b, self.bg_b/maxVal, self.parameter)
            
        print('Transforming image 2')
        frame_b = self.preproc['transformations'](frame_b, self.parameter)

        if self.parameter['do_phase_separation'] == True:
            print('Separating phases for image 2')
            frame_b = self.preproc['phase_separation'](frame_b, self.parameter)
                
        print('Pre-processing image 2')
        frame_b = self.preproc['spatial_filters'](
                frame_b, 
                self.parameter,
                preproc            = True,
                roi_xmin           = roi_xmin,
                roi_xmax           = roi_xmax,
                roi_ymin           = roi_ymin,
                roi_ymax           = roi_ymax,
                )*2**8
        
        if len(mask_coords) >= 1:
            print('Applying mask to image 2')
            frame_b = self.preproc['apply_mask'](frame_b, mask_coords, self.parameter)
            
        frame_b = frame_b.astype('float32')   
        
        if self.parameter['analysis'] != True:
            raise Exception('Cancled analysis via exception')
            
        if self.parameter['window_weighting'] == 'gaussian':
            corr_windowing = ('gaussian', self.parameter['window_weighting_sigma'])
        else:
            corr_windowing = self.parameter['window_weighting']
            
        if self.parameter['algorithm'] not in [
            'I messed up here'
        ]:

            # setup custom windowing
            try:
                corr_window = [int(self.parameter['corr_window_1']),
                               int(self.parameter['corr_window_1'])]
            except:
                corr_window = [
                        int(list(self.parameter['corr_window_1'].split(','))[0]),
                        int(list(self.parameter['corr_window_1'].split(','))[1])
                ]
            try:
                overlap = [int(self.parameter['overlap_window_1']),
                           int(self.parameter['overlap_window_1'])]
            except:
                overlap = [
                    int(list(self.parameter['overlap_window_1'].split(','))[0]),
                    int(list(self.parameter['overlap_window_1'].split(','))[1])
                ]
            passes = 1
            for i in range(2, 7):
                if self.parameter['pass_%1d' % i]:
                    passes += 1
                else:
                    break;
            
            overlap_percent = [overlap[0] / corr_window[0], overlap[1] / corr_window[1]]
            
            print('Evaluating frame: {}'.format(counter + self.disp_off))
            # evaluation first pass
            start = time.time() 
            limit_peak_search = False
            peak_distance = None
            if self.parameter['limit_peak_search_each']:
                limit_peak_search = True
                if self.parameter['limit_peak_search_auto_each'] != True:
                    peak_distance = self.parameter['limit_peak_search_distance_each']

            if passes == 1 and self.parameter['limit_peak_search_last'] == True:
                limit_peak_search = True
                peak_distance = self.parameter['limit_peak_search_distance_last']
              
            if self.parameter['do_s2n']:
                if passes == 1:
                    do_s2n = True
                else:
                    do_s2n = False
            else:
                do_s2n = False

            x, y, u, v, s2n, corr = pivware.firstpass(
                frame_a.astype('float32'), frame_b.astype('float32'),
                window_size                = corr_window,
                overlap                    = overlap,
                normalize_intensity        = self.parameter['normalize_intensity'],
                algorithm                  = self.parameter['algorithm'],
                subpixel_method            = self.parameter['subpixel_method'],
                offset_correlation         = self.parameter['offset_corr_subpix'],
                correlation_method         = self.parameter['corr_method'],
                weight                     = corr_windowing,
                disable_autocorrelation    = self.parameter['disable_autocorrelation'],
                autocorrelation_distance   = self.parameter['disable_autocorrelation_distance'],
                limit_peak_search          = limit_peak_search,
                limit_peak_search_distance = peak_distance,
                do_sig2noise               = do_s2n,
                sig2noise_method           = self.parameter['s2n_method'],
                sig2noise_mask             = self.parameter['s2n_mask'],
                rfft2  = self.rfft_plans['pass_1'],
                irfft2 = self.irfft_plans['pass_1'],
            )

            # validating first pass, signal to noise calc.
            if passes != 1 or self.parameter['validate_last_pass'] == True:
                startn = time.time()
                if self.parameter['exclude_masked_regions'] == True:
                    # applying mask(s)
                    if len(mask_coords) > 0:
                        xymask = coords_to_xymask(x, y, mask_coords).reshape(x.shape)
                    print('Created mask')
                else:
                    xymask = np.ma.nomask
                mask = np.full_like(x, 0)
                if self.parameter['fp_peak2peak_validation'] == True:
                    sig2noise = piv_prc.vectorized_sig2noise_ratio(
                        corr, 
                        sig2noise_method='peak2peak', 
                        width = self.parameter['fp_peak2peak_mask_width']
                    ).reshape(u.shape)
                    u, v, mask, _ = self.postproc['validate_results'](
                        u, v, 
                        mask = xymask,
                        flag = mask,
                        s2n = sig2noise, 
                        s2n_val = True,
                        s2n_thresh = self.parameter['fp_peak2peak_threshold'],
                        global_thresh = False,
                        global_std = False,
                        z_score = False,
                        local_median = False,
                        replace = False,
                    )
                    print('Mean peak-to-peak ratio: '+str(sig2noise.mean()))
                if self.parameter['fp_peak2mean_validation'] == True:
                    sig2noise = piv_prc.vectorized_sig2noise_ratio(
                        corr, 
                        sig2noise_method = 'peak2mean', 
                        width = self.parameter['fp_peak2peak_mask_width']
                    ).reshape(u.shape)
                    u, v, mask, _ = self.postproc['validate_results'](
                        u, v, 
                        mask = xymask,
                        flag = mask,
                        s2n = sig2noise, 
                        s2n_val = True,
                        s2n_thresh = self.parameter['fp_peak2mean_threshold'],
                        global_thresh = False,
                        global_std = False,
                        z_score = False,
                        local_median = False,
                        replace = False,
                    )
                    print('Mean peak-to-mean ratio: '+str(sig2noise.mean()))
                # validate other passes
                u, v, mask, _ = self.postproc['validate_results'](
                    u, v,
                    xymask,
                    flag = mask,
                    global_thresh       = self.parameter['fp_vld_global_threshold'],
                    global_minU         = self.parameter['fp_MinU'],
                    global_maxU         = self.parameter['fp_MaxU'],
                    global_minV         = self.parameter['fp_MinV'],
                    global_maxV         = self.parameter['fp_MaxV'],
                    global_std          = self.parameter['fp_vld_global_threshold'],
                    global_std_thresh   = self.parameter['fp_std_threshold'],
                    z_score             = self.parameter['fp_zscore'],
                    z_score_thresh      = self.parameter['fp_zscore_threshold'],
                    local_median        = self.parameter['fp_local_med_threshold'],
                    local_median_thresh = self.parameter['fp_local_med'],
                    local_median_kernel = self.parameter['fp_local_med_size'],
                    replace             = self.parameter['pass_repl'],
                    replace_method      = self.parameter['pass_repl_method'],
                    replace_inter       = self.parameter['pass_repl_iter'],
                    replace_kernel      = self.parameter['pass_repl_kernel'],
                )
                print(f'Validated pass 1 of frame: {counter + self.disp_off} '+ 
                      f'({_round(time.time() - startn, 3)} second(s))')           

                # smoothning  before deformation if 'each pass' is selected
                startn = time.time()
                if self.parameter['smoothn_each_pass']:
                    if self.parameter['smoothn_first_more']:
                        s *= 1.5
                    _, _, u, v, _ = self.postproc['modify_results'](
                        x, y, u, v,
                        smooth = True,
                        strength = s,
                        robust = self.parameter['robust1']
                    ) 
                    print(f'Smoothned pass 1 for frame: {counter + self.disp_off} '+
                          f'({_round(time.time() - startn, 3)} second(s))')  
                    s = self.parameter['smoothn_val1']

            print(f'Finished pass 1 for frame: {counter + self.disp_off}')
            print(f"window size (y,x): {corr_window[0]}x{corr_window[1]} pixels")
            print(f'overlap (y,x): {overlap[0]}x{overlap[1]} pixels' , '\n')  

            # evaluation of all other passes
            if passes != 1:
                iterations = passes - 1
                for i in range(2, passes + 1):
                    if self.parameter['analysis'] != True:
                        raise Exception('Cancled analysis via exception')
                    # setting up the windowing of each pass
                    try:
                        corr_window = [int(self.parameter[f'corr_window_{i}']),
                                       int(self.parameter[f'corr_window_{i}'])]
                    except:
                        corr_window = [
                                int(list(self.parameter[f'corr_window_{i}'].split(','))[0]),
                                int(list(self.parameter[f'corr_window_{i}'].split(','))[1])
                        ]
                    if self.parameter['update_overlap']:
                        overlap = [int(corr_window[0] * overlap_percent[0]),
                                   int(corr_window[1] * overlap_percent[1])]
                    else:
                        try:
                            overlap = [int(self.parameter[f'overlap_window_{i}']),
                                       int(self.parameter[f'overlap_window_{i}'])]
                        except:
                            overlap = [
                                int(list(self.parameter[f'overlap_window_{i}'].split(','))[0]),
                                int(list(self.parameter[f'overlap_window_{i}'].split(','))[1])
                            ]
                    
                    limit_peak_search = False
                    peak_distance = None
            
                    if self.parameter['limit_peak_search_each']:
                        limit_peak_search = True
                        if self.parameter['limit_peak_search_auto_each'] != True:
                            peak_distance = self.parameter['limit_peak_search_distance_each']
                            
                    if i == passes and self.parameter['limit_peak_search_last'] == True:
                        limit_peak_search = True
                        peak_distance = self.parameter['limit_peak_search_distance_last']
                            
                    if self.parameter['do_s2n']:
                        if passes == i:
                            do_s2n = True
                        else:
                            do_s2n = False
                    else:
                        do_s2n = False
                    x, y, u, v, s2n, corr = pivware.multipass(
                        frame_a.astype('float32'), frame_b.astype('float32'),
                        corr_window,
                        overlap,
                        passes, # number of iterations 
                        i, # current iteration 
                        x, y, u, v,
                        normalize_intensity        = self.parameter['normalize_intensity'],
                        algorithm                  = self.parameter['algorithm'],
                        interpolation_order1       = self.parameter['grid_interpolation_order'],
                        interpolation_order2       = self.parameter['image_interpolation_order'],
                        interpolation_order3       = self.parameter['image_deformation_order'],
                        correlation_method         = self.parameter['corr_method'],
                        subpixel_method            = self.parameter['subpixel_method'],
                        offset_correlation         = self.parameter['offset_corr_subpix'],
                        deformation_method         = self.parameter['deformation_method'],
                        weight                     = corr_windowing,
                        disable_autocorrelation    = self.parameter['disable_autocorrelation'],
                        autocorrelation_distance   = self.parameter['disable_autocorrelation_distance'],
                        limit_peak_search          = limit_peak_search,
                        limit_peak_search_distance = peak_distance,
                        do_sig2noise               = do_s2n,
                        sig2noise_method           = self.parameter['s2n_method'],
                        sig2noise_mask             = self.parameter['s2n_mask'],
                        rfft2 = self.rfft_plans[f'pass_{i}'],
                        irfft2 = self.irfft_plans[f'pass_{i}'],
                    )
                    if i != passes or self.parameter['validate_last_pass'] == True:
                        startn = time.time()
                        if self.parameter['exclude_masked_regions'] == True:
                            # applying mask(s)
                            if len(mask_coords) > 0:
                                xymask = coords_to_xymask(x, y, mask_coords)
                                xymask.reshape(x.shape)
                                print('Created mask')
                        else:
                            xymask = np.ma.nomask
                        mask = np.full_like(x, 0)
                        if self.parameter['sp_peak2peak_validation'] == True:
                            sig2noise = piv_prc.vectorized_sig2noise_ratio(
                                corr, 
                                sig2noise_method='peak2peak', 
                                width = self.parameter['sp_peak2peak_mask_width']
                            ).reshape(u.shape)
                            u, v, mask, _ = self.postproc['validate_results'](
                                u, v, 
                                mask = xymask,
                                flag = mask,
                                s2n = sig2noise, 
                                s2n_val = True,
                                s2n_thresh = self.parameter['sp_peak2peak_threshold'],
                                global_thresh = False,
                                global_std = False,
                                z_score = False,
                                local_median = False,
                                replace = False,
                            )
                            print('Mean peak-to-peak ratio: '+str(sig2noise.mean()))
                        if self.parameter['sp_peak2mean_validation'] == True:
                            sig2noise = piv_prc.vectorized_sig2noise_ratio(
                                corr, 
                                sig2noise_method = 'peak2mean', 
                                width = self.parameter['sp_peak2peak_mask_width']
                            ).reshape(u.shape)
                            u, v, mask, _ = self.postproc['validate_results'](
                                u, v, 
                                mask = xymask,
                                flag = mask,
                                s2n = sig2noise, 
                                s2n_val = True,
                                s2n_thresh = self.parameter['sp_peak2mean_threshold'],
                                global_thresh = False,
                                global_std = False,
                                z_score = False,
                                local_median = False,
                                replace = False,
                            )
                            print('Mean peak-to-mean ratio: '+str(sig2noise.mean()))
                        # validate other passes
                        u, v, mask, _ = self.postproc['validate_results'](
                            u, v,
                            xymask,
                            flag = mask,
                            global_thresh       = self.parameter['sp_vld_global_threshold'],
                            global_minU         = self.parameter['sp_MinU'],
                            global_maxU         = self.parameter['sp_MaxU'],
                            global_minV         = self.parameter['sp_MinV'],
                            global_maxV         = self.parameter['sp_MaxV'],
                            global_std          = self.parameter['sp_vld_global_threshold'],
                            global_std_thresh   = self.parameter['sp_std_threshold'],
                            z_score             = self.parameter['sp_zscore'],
                            z_score_thresh      = self.parameter['sp_zscore_threshold'],
                            local_median        = self.parameter['sp_local_med_validation'],
                            local_median_thresh = self.parameter['sp_local_med'],
                            local_median_kernel = self.parameter['sp_local_med_size'],
                            replace             = self.parameter['pass_repl'],
                            replace_method      = self.parameter['pass_repl_method'],
                            replace_inter       = self.parameter['pass_repl_iter'],
                            replace_kernel      = self.parameter['pass_repl_kernel'],
                        )
                        print(f'Validated pass {i} of frame: {counter + self.disp_off} '+ 
                              f'({_round(time.time() - startn, 3)} second(s))')           
                        startn = time.time()
                        # smoothning each individual pass if 'each pass' is selected
                        if self.parameter['smoothn_each_pass']:
                            _, _, u, v, _ = self.postproc['modify_results'](
                                x, y, u, v,
                                smooth = True,
                                strength = s,
                                robust = self.parameter['robust1']
                            ) 
                            print(f'Smoothned pass {i} for frame: {counter + self.disp_off} '+
                                  f'({_round(time.time() - startn, 3)} second(s))')   
                    print(f'Finished pass {i} for frame: {counter + self.disp_off}')
                    print(f"window size (y,x): {corr_window[0]}x{corr_window[1]} pixels")
                    print(f'overlap (y,x): {overlap[0]}x{overlap[1]} pixels', '\n')  
                    iterations -= 1
            if self.parameter['validate_last_pass'] != True:
                mask = np.full_like(x, 0)
            mask[u == np.nan] = 1
            mask[v == np.nan] = 1
                    
        typevector = mask
        if self.parameter['algorithm'].lower() == 'fft-based convolution':
            u *= -1
            v *= -1
            
        if self.parameter['analysis'] != True:
            raise Exception('Cancled analysis via exception')
        # applying mask(s)
        if len(mask_coords) > 0:
            xymask = coords_to_xymask(x, y, mask_coords).reshape(x.shape)
            u = np.ma.masked_array(u, xymask)
            v = np.ma.masked_array(v, xymask) 

        end = time.time() 
        # save data to dictionary.
        try:
            int(roi_xmin) 
            int(roi_xmax)
            int(roi_ymin)
            int(roi_ymax)
            roi_present = True
        except:
            roi_present = False
    
        results = {}
        results['processed'] = True
        results['roi_present'] = roi_present
        results['roi_coords'] = [roi_xmin, roi_xmax, roi_ymin, roi_ymax]
        results['mask_coords'] = mask_c
        results['process_time'] = _round((end - start), 3)
        results['x'] = x
        results['y'] = y
        results['u'] = u
        results['v'] = -v
        results['tp'] = typevector
        results['s2n'] = s2n
            
        # additional information of evaluation
        time_per_vec = _round((((end - start) * 1000) / u.size), 3)
        print('Processed frame: {}'.format(counter))
        print('Process time: {} second(s)'.format((_round((end - start), 3))))
        print('Number of vectors: {}'.format(u.size))
        print('Time per vector: {} millisecond(s)'.format(time_per_vec), '\n')
            
        if self.parallel == True:
            np.save(
                os.path.join(
                    self.file_path,
                    f'tmp/frame_{counter}.npy'), 
                results
            )
        else:
            results['corr'] = corr
            return results
    
    
    def ensemble_solution(self):
        passes = 1
        for i in range(2, 7):
            if self.parameter['pass_%1d' % i]:
                passes += 1
            else:
                break;
        if self.parameter['window_weighting'] == 'gaussian':
            corr_windowing = ('gaussian', self.parameter['window_weighting_sigma'])
        else:
            corr_windowing = self.parameter['window_weighting']

        # setup custom windowing
        try:
            corr_window = [int(self.parameter['corr_window_1']),
                           int(self.parameter['corr_window_1'])]
        except:
            corr_window = [
                int(list(self.parameter['corr_window_1'].split(','))[0]),
                int(list(self.parameter['corr_window_1'].split(','))[1])
            ]
        try:
            overlap = [int(self.parameter['overlap_window_1']),
                       int(self.parameter['overlap_window_1'])]
        except:
            overlap = [
                int(list(self.parameter['overlap_window_1'].split(','))[0]),
                int(list(self.parameter['overlap_window_1'].split(','))[1])
            ]

        overlap_percent = [overlap[0] / corr_window[0], overlap[1] / corr_window[1]]
        s = self.parameter['smoothn_val1']
            
        limit_peak_search = False
        peak_distance = None
        if self.parameter['limit_peak_search_each']:
            limit_peak_search = True
            if self.parameter['limit_peak_search_auto_each'] != True:
                 peak_distance = self.parameter['limit_peak_search_distance_each']

        if passes == 1 and self.parameter['limit_peak_search_last'] == True:
            limit_peak_search = True
            peak_distance = self.parameter['limit_peak_search_distance_last']

        do_s2n = False
        start = time.time()
        for counter in range(len(self.files_a)):
            if self.parameter['analysis'] != True:
                raise Exception('Cancled analysis via exception')
                
            print('Evaluating frame: {}'.format(counter + self.disp_off))
            frame_a = piv_tls.imread(self.files_a[counter])
            frame_b = piv_tls.imread(self.files_b[counter])

            # preprocessing
            print('\nPre-pocessing frame: {}'.format(counter + self.disp_off))

            roi_coords = self.settings[f'{counter}'][0]
            mask_coords = self.settings[f'{counter}'][1]
            try:
                roi_xmin = int(roi_coords[0])
                roi_xmax = int(roi_coords[1])
                roi_ymin = int(roi_coords[2])
                roi_ymax = int(roi_coords[3])
            except:
                roi_xmin = 0
                roi_ymin = 0
                roi_ymax, roi_xmax = frame_a.shape

            maxVal = frame_a.max()
            frame_a = frame_a.astype('float32')
            frame_a /= maxVal
            if len(self.bg_a) > 1:
                print('Removing background for image 1')
                frame_a = self.preproc['temporal_filters'](frame_a, self.bg_a/maxVal, self.parameter)
            if self.parameter['apply_second_only'] != True:
                print('Transforming image 1')
                frame_a = self.preproc['transformations'](frame_a, self.parameter)
            if self.parameter['do_phase_separation'] == True:
                print('Separating phases for image 1')
                frame_a = self.preproc['phase_separation'](frame_a, self.parameter)

            print('Pre-processing image 1')
            frame_a = self.preproc['spatial_filters']( # float32 takes less space for FFTs
                    frame_a, 
                    self.parameter,
                    preproc            = True,
                    roi_xmin           = roi_xmin,
                    roi_xmax           = roi_xmax,
                    roi_ymin           = roi_ymin,
                    roi_ymax           = roi_ymax,
                    )*2**8

            if len(mask_coords) >= 1:
                print('Applying mask to image 1')
                frame_a = self.preproc['apply_mask'](frame_a, mask_coords, self.parameter)

            frame_a = frame_a.astype('float32')    

            maxVal = frame_b.max()
            frame_b = frame_b.astype('float32')
            frame_b /= maxVal

            if len(self.bg_a) > 1:
                print('Removing background for image 2')
                frame_b = self.preproc['temporal_filters'](frame_b, self.bg_b/maxVal, self.parameter)

            print('Transforming image 2')
            frame_b = self.preproc['transformations'](frame_b, self.parameter)

            if self.parameter['do_phase_separation'] == True:
                print('Separating phases for image 2')
                frame_b = self.preproc['phase_separation'](frame_b, self.parameter)

            print('Pre-processing image 2')
            frame_b = self.preproc['spatial_filters'](
                    frame_b, 
                    self.parameter,
                    preproc            = True,
                    roi_xmin           = roi_xmin,
                    roi_xmax           = roi_xmax,
                    roi_ymin           = roi_ymin,
                    roi_ymax           = roi_ymax,
                    )*2**8

            if len(mask_coords) >= 1:
                print('Applying mask to image 2')
                frame_b = self.preproc['apply_mask'](frame_b, mask_coords, self.parameter)

            frame_b = frame_b.astype('float32')

            if self.parameter['analysis'] != True:
                raise Exception('Cancled analysis via exception')

            corr = pivware.firstpass(
                frame_a.astype('float32'), frame_b.astype('float32'),
                window_size                = corr_window,
                overlap                    = overlap,
                normalize_intensity        = self.parameter['normalize_intensity'],
                algorithm                  = self.parameter['algorithm'],
                subpixel_method            = self.parameter['subpixel_method'],
                correlation_method         = self.parameter['corr_method'],
                weight                     = corr_windowing,
                disable_autocorrelation    = self.parameter['disable_autocorrelation'],
                autocorrelation_distance   = self.parameter['disable_autocorrelation_distance'],
                limit_peak_search          = limit_peak_search,
                limit_peak_search_distance = peak_distance,
                do_sig2noise               = do_s2n,
                sig2noise_method           = self.parameter['s2n_method'],
                sig2noise_mask             = self.parameter['s2n_mask'],
            )[5]
            
            if counter == 0:
                corr_avg = corr
            else:
                corr_avg += corr
            print(f'Finished frame {counter} pass 1')
            
        corr_avg /= len(self.files_a)
        
        x, y = piv_prc.get_rect_coordinates(
            frame_a.shape,
            corr_window,
            overlap
        )
        n_rows, n_cols = piv_prc.get_field_shape(
            frame_a.shape, 
            corr_window, 
            overlap
        )
        u, v = piv_prc.vectorized_correlation_to_displacements(
            corr_avg, 
            n_rows,
            n_cols,
            subpixel_method=self.parameter['subpixel_method'],
            offset_minimum = True
        )
        
        # validating first pass, signal to noise calc.
        if passes != 1 or self.parameter['validate_last_pass'] == True:
            #if self.parameter['exclude_masked_regions'] == True:
            #    # applying mask(s)
            #    if len(mask_coords) > 0:
            #        xymask = coords_to_xymask(x, y, mask_coords).reshape(x.shape)
            #        print('Created mask')
            #else:
            xymask = np.ma.nomask
                
            mask = np.full_like(x, 0)
            if self.parameter['fp_peak2peak_validation'] == True:
                sig2noise = piv_prc.vectorized_sig2noise_ratio(
                    corr, 
                    sig2noise_method='peak2peak', 
                    width = self.parameter['fp_peak2peak_mask_width']
                ).reshape(u.shape)
                u, v, mask, _ = self.postproc['validate_results'](
                    u, v, 
                    mask = xymask,
                    flag = mask,
                    s2n = sig2noise, 
                    s2n_val = True,
                    s2n_thresh = self.parameter['fp_peak2peak_threshold'],
                    global_thresh = False,
                    global_std = False,
                    z_score = False,
                    local_median = False,
                    replace = False,
                )
                print('Mean peak-to-peak ratio: '+str(sig2noise.mean()))
                
            if self.parameter['fp_peak2mean_validation'] == True:
                sig2noise = piv_prc.vectorized_sig2noise_ratio(
                    corr, 
                    sig2noise_method = 'peak2mean', 
                    width = self.parameter['fp_peak2peak_mask_width']
                ).reshape(u.shape)
                u, v, mask, _ = self.postproc['validate_results'](
                    u, v, 
                    mask = xymask,
                    flag = mask,
                    s2n = sig2noise, 
                    s2n_val = True,
                    s2n_thresh = self.parameter['fp_peak2mean_threshold'],
                    global_thresh = False,
                    global_std = False,
                    z_score = False,
                    local_median = False,
                    replace = False,
                )
                print('Mean peak-to-mean ratio: '+str(sig2noise.mean()))
                
            # validate other passes
            u, v, mask, _ = self.postproc['validate_results'](
                u, v,
                xymask,
                flag = mask,
                global_thresh       = self.parameter['fp_vld_global_threshold'],
                global_minU         = self.parameter['fp_MinU'],
                global_maxU         = self.parameter['fp_MaxU'],
                global_minV         = self.parameter['fp_MinV'],
                global_maxV         = self.parameter['fp_MaxV'],
                global_std          = self.parameter['fp_vld_global_threshold'],
                global_std_thresh   = self.parameter['fp_std_threshold'],
                z_score             = self.parameter['fp_zscore'],
                z_score_thresh      = self.parameter['fp_zscore_threshold'],
                local_median        = self.parameter['fp_local_med_threshold'],
                local_median_thresh = self.parameter['fp_local_med'],
                local_median_kernel = self.parameter['fp_local_med_size'],
                replace             = self.parameter['pass_repl'],
                replace_method      = self.parameter['pass_repl_method'],
                replace_inter       = self.parameter['pass_repl_iter'],
                replace_kernel      = self.parameter['pass_repl_kernel'],
            )
            print('Validated pass 1') 

            # smoothning  before deformation if 'each pass' is selected
            if self.parameter['smoothn_each_pass']:
                if self.parameter['smoothn_first_more']:
                    s *= 1.5
                _, _, u, v, _ = self.postproc['modify_results'](
                    x, y, u, v,
                    smooth = True,
                    strength = s,                        
                    robust = self.parameter['robust1']
                ) 
                
                print('Smoothned pass 1 for frame: {}'.format(counter + self.disp_off))
                s = self.parameter['smoothn_val1']

        print('Finished pass 1')
        print(f"window size (y,x): {corr_window[0]}x{corr_window[1]} pixels")
        print(f'overlap (y,x): {overlap[0]}x{overlap[1]} pixels' , '\n')  
            
        if passes != 1:
            iterations = passes - 1
            for i in range(2, passes + 1):
                if self.parameter['analysis'] != True:
                    raise Exception('Cancled analysis via exception')
                # setting up the windowing of each pass
                try:
                    corr_window = [int(self.parameter[f'corr_window_{i}']),
                                   int(self.parameter[f'corr_window_{i}'])]
                except:
                    corr_window = [
                        int(list(self.parameter[f'corr_window_{i}'].split(','))[0]),
                        int(list(self.parameter[f'corr_window_{i}'].split(','))[1])
                    ]
                if self.parameter['update_overlap']:
                    overlap = [int(corr_window[0] * overlap_percent[0]),
                               int(corr_window[1] * overlap_percent[1])]
                else:
                    try:
                         overlap = [int(self.parameter[f'overlap_window_{i}']),
                                   int(self.parameter[f'overlap_window_{i}'])]
                    except:
                        overlap = [
                            int(list(self.parameter[f'overlap_window_{i}'].split(','))[0]),
                            int(list(self.parameter[f'overlap_window_{i}'].split(','))[1])
                        ]
                    
                limit_peak_search = False
                peak_distance = None
            
                if self.parameter['limit_peak_search_each']:
                    limit_peak_search = True
                    if self.parameter['limit_peak_search_auto_each'] != True:
                        peak_distance = self.parameter['limit_peak_search_distance_each']
                        
                if i == passes and self.parameter['limit_peak_search_last'] == True:
                    limit_peak_search = True
                    peak_distance = self.parameter['limit_peak_search_distance_last']
                                
                for counter in range(len(self.files_a)):
                    if self.parameter['analysis'] != True:
                        raise Exception('Cancled analysis via exception')

                    print('Evaluating frame: {}'.format(counter + self.disp_off))
                    frame_a = piv_tls.imread(self.files_a[counter])
                    frame_b = piv_tls.imread(self.files_b[counter])

                    # preprocessing
                    print('\nPre-pocessing frame: {}'.format(counter + self.disp_off))

                    roi_coords = self.settings[f'{counter}'][0]
                    mask_coords = self.settings[f'{counter}'][1]
                    try:
                        roi_xmin = int(roi_coords[0])
                        roi_xmax = int(roi_coords[1])
                        roi_ymin = int(roi_coords[2])
                        roi_ymax = int(roi_coords[3])
                    except:
                        roi_xmin = 0
                        roi_ymin = 0
                        roi_ymax, roi_xmax = frame_a.shape

                    maxVal = frame_a.max()
                    frame_a = frame_a.astype('float32')
                    frame_a /= maxVal
                    if len(self.bg_a) > 1:
                        print('Removing background for image 1')
                        frame_a = self.preproc['temporal_filters'](frame_a, self.bg_a/maxVal, self.parameter)
                    if self.parameter['apply_second_only'] != True:
                        print('Transforming image 1')
                        frame_a = self.preproc['transformations'](frame_a, self.parameter)
                    if self.parameter['do_phase_separation'] == True:
                        print('Separating phases for image 1')
                        frame_a = self.preproc['phase_separation'](frame_a, self.parameter)

                    print('Pre-processing image 1')
                    frame_a = self.preproc['spatial_filters']( # float32 takes less space for FFTs
                            frame_a, 
                            self.parameter,
                            preproc            = True,
                            roi_xmin           = roi_xmin,
                            roi_xmax           = roi_xmax,
                            roi_ymin           = roi_ymin,
                            roi_ymax           = roi_ymax,
                            )*2**8

                    if len(mask_coords) >= 1:
                        print('Applying mask to image 1')
                        frame_a = self.preproc['apply_mask'](frame_a, mask_coords, self.parameter)

                    frame_a = frame_a.astype('float32')    

                    maxVal = frame_b.max()
                    frame_b = frame_b.astype('float32')
                    frame_b /= maxVal

                    if len(self.bg_a) > 1:
                        print('Removing background for image 2')
                        frame_b = self.preproc['temporal_filters'](frame_b, self.bg_b/maxVal, self.parameter)

                    print('Transforming image 2')
                    frame_b = self.preproc['transformations'](frame_b, self.parameter)

                    if self.parameter['do_phase_separation'] == True:
                        print('Separating phases for image 2')
                        frame_b = self.preproc['phase_separation'](frame_b, self.parameter)

                    print('Pre-processing image 2')
                    frame_b = self.preproc['spatial_filters'](
                            frame_b, 
                            self.parameter,
                            preproc            = True,
                            roi_xmin           = roi_xmin,
                            roi_xmax           = roi_xmax,
                            roi_ymin           = roi_ymin,
                            roi_ymax           = roi_ymax,
                            )*2**8

                    if len(mask_coords) >= 1:
                        print('Applying mask to image 2')
                        frame_b = self.preproc['apply_mask'](frame_b, mask_coords, self.parameter)

                    frame_b = frame_b.astype('float32')

                    if self.parameter['analysis'] != True:
                        raise Exception('Cancled analysis via exception')
                    
                    corr = pivware.multipass(
                        frame_a.astype('float32'), frame_b.astype('float32'),
                        corr_window,
                        overlap,
                        passes, # number of iterations 
                        i, # current iteration 
                        x, y, u, v,
                        normalize_intensity        = self.parameter['normalize_intensity'],
                        algorithm                  = self.parameter['algorithm'],
                        interpolation_order1       = self.parameter['grid_interpolation_order'],
                        interpolation_order2       = self.parameter['image_interpolation_order'],
                        interpolation_order3       = self.parameter['image_deformation_order'],
                        correlation_method         = self.parameter['corr_method'],
                        subpixel_method            = self.parameter['subpixel_method'],
                        deformation_method         = self.parameter['deformation_method'],
                        weight                     = corr_windowing,
                        disable_autocorrelation    = self.parameter['disable_autocorrelation'],
                        autocorrelation_distance   = self.parameter['disable_autocorrelation_distance'],
                        limit_peak_search          = limit_peak_search,
                        limit_peak_search_distance = peak_distance,
                        do_sig2noise               = do_s2n,
                        sig2noise_method           = self.parameter['s2n_method'],
                        sig2noise_mask             = self.parameter['s2n_mask'],
                    )[5]
                    if counter == 0:
                        corr_avg = corr
                    else:
                        corr_avg += corr
                    print(f'Finished frame {counter} pass {i}')
                    
                corr_avg /= len(self.files_a)
                
                x, y = piv_prc.get_rect_coordinates(
                    frame_a.shape,
                    corr_window,
                    overlap
                )
                n_rows, n_cols = piv_prc.get_field_shape(
                    frame_a.shape, 
                    corr_window, 
                    overlap
                )

                u, v = piv_prc.vectorized_correlation_to_displacements(
                    corr_avg, 
                    n_rows,
                    n_cols,
                    subpixel_method=self.parameter['subpixel_method'],
                    offset_minimum = True
                )
                
                if i != passes or self.parameter['validate_last_pass'] == True:
                    # applying mask(s)
                    xymask = np.ma.nomask
                    mask = np.full_like(x, 0)
                    if self.parameter['sp_peak2peak_validation'] == True:
                        sig2noise = piv_prc.vectorized_sig2noise_ratio(
                            corr, 
                            sig2noise_method='peak2peak', 
                            width = self.parameter['sp_peak2peak_mask_width']
                        ).reshape(u.shape)
                        u, v, mask, _ = self.postproc['validate_results'](
                            u, v, 
                            mask = xymask,
                            flag = mask,
                            s2n = sig2noise, 
                            s2n_val = True,
                            s2n_thresh = self.parameter['sp_peak2peak_threshold'],
                            global_thresh = False,
                            global_std = False,
                            z_score = False,
                            local_median = False,
                            replace = False,
                        )
                        print('Mean peak-to-peak ratio: '+str(sig2noise.mean()))
                        
                    if self.parameter['sp_peak2mean_validation'] == True:
                        sig2noise = piv_prc.vectorized_sig2noise_ratio(
                            corr, 
                            sig2noise_method = 'peak2mean', 
                            width = self.parameter['sp_peak2peak_mask_width']
                        ).reshape(u.shape)
                        u, v, mask, _ = self.postproc['validate_results'](
                            u, v, 
                            mask = xymask,
                            flag = mask,
                            s2n = sig2noise, 
                            s2n_val = True,
                            s2n_thresh = self.parameter['sp_peak2mean_threshold'],
                            global_thresh = False,
                            global_std = False,
                            z_score = False,
                            local_median = False,
                            replace = False,
                        )
                        print('Mean peak-to-mean ratio: '+str(sig2noise.mean()))
                        
                    # validate other passes
                    u, v, mask, _ = self.postproc['validate_results'](
                        u, v,
                        xymask,
                        flag = mask,
                        global_thresh       = self.parameter['sp_vld_global_threshold'],
                        global_minU         = self.parameter['sp_MinU'],
                        global_maxU         = self.parameter['sp_MaxU'],
                        global_minV         = self.parameter['sp_MinV'],
                        global_maxV         = self.parameter['sp_MaxV'],
                        global_std          = self.parameter['sp_vld_global_threshold'],
                        global_std_thresh   = self.parameter['sp_std_threshold'],
                        z_score             = self.parameter['sp_zscore'],
                        z_score_thresh      = self.parameter['sp_zscore_threshold'],
                        local_median        = self.parameter['sp_local_med_validation'],
                        local_median_thresh = self.parameter['sp_local_med'],
                        local_median_kernel = self.parameter['sp_local_med_size'],
                        replace             = self.parameter['pass_repl'],
                        replace_method      = self.parameter['pass_repl_method'],
                        replace_inter       = self.parameter['pass_repl_iter'],
                        replace_kernel      = self.parameter['pass_repl_kernel'],
                    )
                    print('Validated pass {} of frame: {}'.format(i,counter + self.disp_off))             

                    # smoothning each individual pass if 'each pass' is selected
                    if self.parameter['smoothn_each_pass']:
                        _, _, u, v, _ = self.postproc['modify_results'](
                            x, y, u, v,
                            smooth = True,
                            strength = s,
                            robust = self.parameter['robust1']
                        ) 
                        print('Smoothned pass {} for frame: {}'.format(i,counter + self.disp_off))
                print(f'Finished pass {i}')
                print(f"window size (y,x): {corr_window[0]}x{corr_window[1]} pixels")
                print(f'overlap (y,x): {overlap[0]}x{overlap[1]} pixels', '\n')  
                iterations -= 1
                
        ttf = 0
        print('Generating mask')
        for i in range(len(self.files_a)):
            mask_coords = self.settings[f'{i}'][1]
            if len(mask_coords) > 0:
                if ttf == 0:
                    xymask = coords_to_xymask(x, y, mask_coords)
                else:
                    xymask += coords_to_xymask(x, y, mask_coords)
                    ttf = 1
            else:
                if ttf == 0:
                    xymask = np.zeros(n_rows * n_cols)
        xymask = xymask.astype('float32') / len(self.files_a)
        xymask[xymask < 0.5] = 0
        xymask[xymask > 0.5] = 1
        xymask = xymask.reshape(x.shape)
        print('Generated mask')
             
            
        if self.parameter['do_s2n']:
            sig2noise = piv_prc.vectorized_sig2noise_ratio(
                corr, 
                sig2noise_method=self.parameter['s2n_method'],
                width=self.parameter['s2n_mask']
            )
        else:
            sig2noise = np.zeros(n_rows * n_cols)
        sig2noise = sig2noise.reshape(n_rows, n_cols)
        
        try:
            int(roi_xmin) 
            int(roi_xmax)
            int(roi_ymin)
            int(roi_ymax)
            roi_present = True
        except:
            roi_present = False
            
        end = time.time()
        results = {}
        results['processed'] = True
        results['roi_present'] = roi_present
        results['roi_coords'] = [roi_xmin, roi_xmax, roi_ymin, roi_ymax]
        results['mask_coords'] = []
        results['process_time'] = _round((end - start), 3)
        results['x'] = x
        results['y'] = y
        results['u'] = u
        results['v'] = -v
        results['tp'] = np.zeros_like(x)
        results['s2n'] = sig2noise
        return results
                
                
        
        
import scipy.ndimage as scn
from scipy import fft
from scipy.signal import convolve2d, get_window
from scipy.interpolate import RectBivariateSpline
    
class pivware():    
    def firstpass(
        frame_a, frame_b,
        window_size = 64,
        overlap = 32,
        algorithm = 'Standard FFT correlation',
        normalize_intensity = False,
        normalized_correlation = False,
        correlation_method = "circular",
        weight = 'boxcar',
        disable_autocorrelation = False,
        autocorrelation_distance = 1,
        limit_peak_search = False,
        limit_peak_search_distance = None,
        subpixel_method = "gaussian",
        offset_correlation = True,
        do_sig2noise = False,
        sig2noise_method = 'peak2mean',
        sig2noise_mask = 2,
        rfft2 = fft.rfft2,
        irfft2 = fft.irfft2,
    ):      
        start = time.time()
        x, y = piv_prc.get_rect_coordinates(frame_a.shape,
                                       window_size,
                                       overlap)
        print(f'Generated grid ({_round(time.time() - start, 6)*1000} millisecond(s))') 
        
        aa = piv_prc.sliding_window_array(
            frame_a, 
            window_size, 
            overlap
        )
        bb = piv_prc.sliding_window_array(
            frame_b, 
            window_size,
            overlap
        )

        n_rows, n_cols = piv_prc.get_field_shape(
            frame_a.shape, 
            window_size, 
            overlap
        )
        
        weight_y = get_window(weight, aa.shape[-2])
        weight_x = get_window(weight, aa.shape[-1])
        weight = np.outer(weight_y, weight_x).astype('float32')
        aa = aa * weight[np.newaxis,:,:]
        bb = bb * weight[np.newaxis,:,:]
        
        start = time.time()
        if algorithm.lower() == 'minimum quadratic differences':
            corr = pivware.mqd_correlation(
                aa, bb, 
                correlation_method,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        elif algorithm.lower() == 'phase correlation':
            corr = pivware.phase_correlation(
                aa, bb, 
                correlation_method=correlation_method,
                normalized_correlation=True,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        elif algorithm.lower() == 'fft-based convolution':
            corr = pivware.convolution(
                aa - aa.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
                bb - bb.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
            ) / (aa.shape[1] * aa.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] *
                 aa.shape[2] * bb.std(axis = (-2,-1))[:, np.newaxis, np.newaxis]).astype('float32')
        #elif algorithm.lower() == 'normalized fft correlation':
        #    corr = piv_prc.fft_correlate_images(
        #        aa,bb, 
        #        correlation_method=correlation_method,
        #        normalized_correlation=True)
            
        elif algorithm.lower() == 'normalized fft correlation':
            corr = pivware.fft_norm_correlate_images(
                aa, bb, 
                correlation_method=correlation_method,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        else:
            if correlation_method == 'linear':
                aa -= aa.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
                bb -= bb.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
            corr = piv_prc.fft_correlate_images(
                aa, bb,
                correlation_method=correlation_method,
                normalized_correlation=False,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        print(f'Correlated images ({_round(time.time() - start, 3)} second(s))')  
        print(f'Memory allocated towards correlation matrix: {corr.nbytes / 1e6} MB')
        
        if do_sig2noise:
            sig2noise = piv_prc.vectorized_sig2noise_ratio(
                corr, 
                sig2noise_method=sig2noise_method, 
                width=sig2noise_mask
            )
        else:
            sig2noise = np.zeros(n_rows * n_cols)
        sig2noise = sig2noise.reshape(n_rows, n_cols)
        
        if corr[0, :, :].shape[0] >= 8:
            if disable_autocorrelation:
                corr = pivware.remove_autocorrelation_peak(
                    corr, 
                    autocorrelation_distance
                )
            if limit_peak_search:
                corr = pivware.limit_peak_search_area(
                    corr, 
                    limit_peak_search_distance
                )
        
        start = time.time()
        #if offset_minimum == True:
        corr_min = corr.min(axis = (-2,-1)) # avoid negative peaks 
        corr_min[corr_min > 0] = 0
        corr -= corr_min[:, np.newaxis, np.newaxis]
        u, v = piv_prc.vectorized_correlation_to_displacements(
            corr, 
            n_rows,
            n_cols,
            subpixel_method=subpixel_method,
            #offset_minimum = offset_correlation
        )
        print(f'Found peaks/displacements ({_round(time.time() - start, 6)*1000} milliseconds)')   
        
        u = u.reshape(n_rows, n_cols)
        v = v.reshape(n_rows, n_cols)
        
        # applying blank mask
        u = np.ma.masked_array(u, np.ma.nomask)
        v = np.ma.masked_array(v, np.ma.nomask) 
        
        return x, y, u, v, sig2noise, corr
    
    
    
    def multipass(
        frame_a,frame_b,
        window_size,
        overlap,
        iterations,
        current_iteration,
        x_old, y_old, u_old, v_old,
        algorithm = 'Standard FFT Correlation',
        normalize_intensity = False,
        normalized_correlation = False,
        correlation_method = "circular",
        deformation_method = "second image",
        interpolation_order1 = 3,
        interpolation_order2 = 3,
        interpolation_order3 = 3,
        weight = 'boxcar',
        disable_autocorrelation = False,
        autocorrelation_distance = 1,
        limit_peak_search = False,
        limit_peak_search_distance = None,
        subpixel_method = "gaussian",
        offset_correlation = True,
        do_sig2noise = False,
        sig2noise_method = 'peak2mean',
        sig2noise_mask = 2,
        rfft2 = fft.rfft2,
        irfft2 = fft.irfft2,
        ):
        """
        A slighly modified version of the original multipass_window_deform
        algorithm. For more information on the algorithm, please refer to 
        the windef file located in the OpenPIV package. For copyright information,
        please check the OpenPIV GitHub repository.
        """
            
        start = time.time()
        x, y = piv_prc.get_rect_coordinates(
            frame_a.shape,
            window_size,
            overlap)
        print(f'\nGenerated grid ({_round(time.time() - start, 6)*1000} millisecond(s))')
        
        start = time.time()
        y_old = y_old[:, 0]
        x_old = x_old[0, :]
        y_int = y[:, 0]
        x_int = x[0, :]      

        # interpolating the displacements from the old grid onto the new grid
        # y befor x because of numpy works row major
        start = time.time()
        ip = RectBivariateSpline(
            y_old, x_old, u_old.filled(0.),
            kx = interpolation_order1,
            ky = interpolation_order1,
        )
        u_pre = ip(y_int, x_int)

        ip2 = RectBivariateSpline(
            y_old, x_old, v_old.filled(0.),
            kx = interpolation_order1,
            ky = interpolation_order1,
        )
        v_pre = ip2(y_int, x_int)
        print(f'Interpolated old grid onto new grid ({_round(time.time() - start, 6)*1000} milliseconds)')

        if deformation_method != 'none':
            if deformation_method == "symmetric":
                x_new, y_new, ut, vt = piv_wdf.create_deformation_field(
                    frame_a, 
                    x, y, u_pre, v_pre,
                    kx = interpolation_order3,
                    ky = interpolation_order3,
                )
                frame_a = scn.map_coordinates(
                    frame_a, ((y_new - vt / 2, x_new - ut / 2)),
                    order=interpolation_order2, 
                    mode='nearest'
                )
                frame_b = scn.map_coordinates(
                    frame_b, ((y_new + vt / 2, x_new + ut / 2)),
                    order=interpolation_order2, 
                    mode='nearest'
                )
                
            elif deformation_method == "second image":
                frame_b = piv_wdf.deform_windows(
                    frame_b, x, y, u_pre, -v_pre,
                    interpolation_order=interpolation_order2,
                    kx = interpolation_order3,
                    ky = interpolation_order3,
                )
            else:
                raise Exception("Deformation method is not valid.")
        
            print(f'Deformed images ({_round(time.time() - start, 6)*1000} milliseconds)') 
        
        aa = piv_prc.sliding_window_array(
            frame_a, 
            window_size, 
            overlap
        )
        bb = piv_prc.sliding_window_array(
            frame_b, 
            window_size,
            overlap
        )
        
        n_rows, n_cols = piv_prc.get_field_shape(
            frame_a.shape, 
            window_size, 
            overlap
        )
        
        weight_y = get_window(weight, aa.shape[-2])
        weight_x = get_window(weight, aa.shape[-1])
        weight = np.outer(weight_y, weight_x).astype('float32')
        aa = aa * weight[np.newaxis,:,:]
        bb = bb * weight[np.newaxis,:,:]
        
        start = time.time()
        if algorithm.lower() == 'minimum quadratic differences':
            corr = pivware.mqd_correlation(
                aa, bb, 
                correlation_method,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )

        elif algorithm.lower() == 'phase correlation':
            corr = pivware.phase_correlation(
                aa, bb, 
                correlation_method=correlation_method,
                normalized_correlation=True,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        elif algorithm.lower() == 'fft-based convolution':
            corr = pivware.convolution(
                aa - aa.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
                bb - bb.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
            ) / (aa.shape[1] * aa.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] *
                 aa.shape[2] * bb.std(axis = (-2,-1))[:, np.newaxis, np.newaxis]).astype('float32')
        #elif algorithm.lower() == 'normalized fft correlation':
        #    corr = piv_prc.fft_correlate_images(
        #        aa,bb, 
        #        correlation_method=correlation_method,
        #        normalized_correlation=True)
            
        elif algorithm.lower() == 'normalized fft correlation':
            corr = pivware.fft_norm_correlate_images(
                aa, bb, 
                correlation_method=correlation_method,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
            
        else:
            if correlation_method == 'linear':
                aa -= aa.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
                bb -= bb.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
            corr = piv_prc.fft_correlate_images(
                aa, bb,
                correlation_method=correlation_method,
                normalized_correlation=False,
                rfft2 = rfft2,
                irfft2 = irfft2,
            )
        if weight.mean() != 1.0:
            weight = piv_prc.fft_correlate_images(
                weight[np.newaxis, :, :], 
                weight[np.newaxis, :, :],
                'circular', True
            )  
            corr *= (weight / weight.max())
        print(f'Correlated images ({_round(time.time() - start, 3)} second(s))') 
        print(f'Memory allocated towards correlation matrix: {corr.nbytes / 1e6} MB')
        
        if do_sig2noise:
            sig2noise = piv_prc.vectorized_sig2noise_ratio(
                corr, 
                sig2noise_method=sig2noise_method, 
                width=sig2noise_mask
            )
        else:
            sig2noise = np.zeros(n_rows * n_cols)
        sig2noise = sig2noise.reshape(n_rows, n_cols)
        
        if corr[0, :, :].shape[0] >= 8:
            if disable_autocorrelation:
                corr = pivware.remove_autocorrelation_peak(
                            corr, 
                            autocorrelation_distance
                )
            if limit_peak_search:
                corr = pivware.limit_peak_search_area(
                            corr, 
                            limit_peak_search_distance
                        )

        # normalize correlation to 0..1 if no normalization is already done
        #if normalized_correlation == False:
            #corr = (corr - corr.min()) / (corr.max() - corr.min())
        
        start = time.time()
        #if offset_minimum == True:
        corr_min = corr.min(axis = (-2,-1)) # avoid negative peaks 
        corr_min[corr_min > 0] = 0
        corr -= corr_min[:, np.newaxis, np.newaxis]
        u, v = piv_prc.vectorized_correlation_to_displacements(
            corr, 
            n_rows,
            n_cols,
            subpixel_method=subpixel_method,
            #offset_minimum = offset_correlation
        )
        print(f'Found peaks/displacements ({_round(time.time() - start, 6)*1000} milliseconds)') 
        
        u = u.reshape(n_rows, n_cols)
        v = v.reshape(n_rows, n_cols)

        # adding or averaging the recent displacment on to the displacment of the previous pass
        if deformation_method != 'none':
            u += u_pre
            v += v_pre
            
        else:
            u = (u + u_pre) / 2
            v = (v + v_pre) / 2
    
        # applying blank mask
        u = np.ma.masked_array(u, np.ma.nomask)
        v = np.ma.masked_array(v, np.ma.nomask) 
            
        return x, y, u, v, sig2noise, corr
    
    
    def fft_norm_correlate_images(image_a, image_b,
                                  correlation_method="circular",
                                  conj = np.conj,
                                  rfft2 = fft.rfft2,
                                  irfft2 = fft.irfft2,
                                  fftshift = fft.fftshift):
        """ FFT based normalized cross correlation
        of two images with multiple views of np.stride_tricks()
        The 2D FFT should be applied to the last two axes (-2,-1) and the
        zero axis is the number of the interrogation window
        This should also work out of the box for rectangular windows.
        Parameters
        ----------
        image_a : 3d np.ndarray, first dimension is the number of windows,
            and two last dimensions are interrogation windows of the first image

        image_b : similar

        correlation_method : string
            one of the three methods implemented: 'circular' or 'linear'
            [default: 'circular].

        conj : function
            function used for complex conjugate

        rfft2 : function
            function used for rfft2

        irfft2 : function
            function used for irfft2

        fftshift : function
            function used for fftshift

        """
        
        return piv_prc.fft_correlate_images(
            image_a - image_a.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
            image_b - image_b.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis],
            correlation_method=correlation_method,
            normalized_correlation=False,
            rfft2 = rfft2,
            irfft2 = irfft2,
            conj = conj,
            fftshift = fftshift,
        ) / (
            image_b.shape[-2] * image_a.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] * 
            image_b.shape[-1] * image_b.std(axis = (-2,-1))[:, np.newaxis, np.newaxis]
        ).astype('float32')


    def phase_correlation(image_a, image_b,
                          correlation_method = 'circular',
                          normalized_correlation = True,
                          conj = np.conj,
                          rfft2 = fft.rfft2,
                          irfft2 = fft.irfft2,
                          fftshift = fft.fftshift):
        '''
        Phase filtering to produce a phase-only correlation. Two methods
        are implemented here: Phase-only correlation and "symmetric" phase 
        correlation, which is supposedly more robust.
        Parameters
        ----------
        image_a : 3d np.ndarray, first dimension is the number of windows,
            and two last dimensions are interrogation windows of the first image

        image_b : similar

        correlation_method : string
            one of the two methods implemented: 'circular' or 'linear'
            [default: 'circular].

        normalized_correlation : bool
            decides wether normalized correlation is done or not: True or False
            [default: True].

        Returns
        -------
        corr : 3d np.ndarray
            a three dimensions array for the correlation function.
        '''
        if correlation_method not in ['circular', 'linear']:
            raise ValueError(f'Correlation method not supported {correlation_method}')

        s1 = np.array(image_a.shape[-2:])
        s2 = np.array(image_b.shape[-2:])
        if normalized_correlation == True:
            norm = (s2[0] * image_a.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] * 
                    s2[1] * image_b.std(axis = (-2,-1))[:, np.newaxis, np.newaxis])
            image_a -= image_a.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
            image_b -= image_b.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis]
        else:
            norm = 1

        if correlation_method == 'circular':
            f2a = conj(rfft2(image_a))
            f2b = rfft2(image_b)
            r = f2a * f2b
            r /= (np.sqrt(np.absolute(f2a) * np.absolute(f2b)) + 1e-10)
            corr = fftshift(irfft2(r).real, axes=(-2, -1)) * 100 # so it would be in range of float32
        else:
            size = s1 + s2 - 1
            fsize = 2 ** np.ceil(np.log2(size)).astype(int)
            fslice = (slice(0, image_a.shape[0]),
                      slice((fsize[0]-s1[0])//2, (fsize[0]+s1[0])//2),
                      slice((fsize[1]-s1[1])//2, (fsize[1]+s1[1])//2))
            f2a = conj(rfft2(image_a, fsize, axes=(-2, -1)))
            f2b = rfft2(image_b, fsize, axes=(-2, -1))
            r = f2a * f2b
            r /= (np.sqrt(np.absolute(f2a) * np.absolute(f2b)) + 1e-10)
            corr = fftshift(irfft2(r), axes=(-2, -1)).real[fslice] * 100
        return corr / norm.astype('float32')
    
    
    def convolution(aa, bb):
        corrMat = []
        for i in range(aa.shape[0]):
            corr = piv_prc.fft_correlate_windows(
                aa[i,:,:], 
                bb[i,:,:],
            ).astype('float32')
            corrMat.append(corr)
        return np.array(corrMat)
    
    
    def remove_autocorrelation_peak(corr, distance = 1, offset = 0):
        y, x = corr.shape[1:3]
        if isinstance(distance, list) != True and isinstance(distance, list) != True: 
            distance = [distance, distance]
        meanCorr = np.nanmean(corr, axis = (-2,-1))[:, np.newaxis, np.newaxis]
        corr[:, y//2 - distance[0] : y//2 + distance[0], x//2 - distance[1] : x//2 + distance[1]] = meanCorr
        return corr
    
    
    def limit_peak_search_area(corr, distance = None):
        y, x = corr.shape[1:3]
        if distance == None:
            distance = [y // 4, x//4]
        if isinstance(distance, list) != True and isinstance(distance, list) != True: 
            distance = [distance, distance]
        if distance[0] >= 2 or distance[1] >= 2:
            meanCorr = np.nanmean(corr, axis = (-2,-1))[:, np.newaxis, np.newaxis]
            corrNew = np.zeros(corr.shape) + meanCorr
            corrNew[:, y//2 - distance[0] : y//2 + distance[0], x//2 - distance[1] : x//2 + distance[1]] = corr[
                    :, y//2 - distance[0] : y//2 + distance[0], x//2 - distance[1] : x//2 + distance[1]]
            corr = corrNew
        else:
            print('Ignoring peak search area limit due to search area size being less than 4 pixels in either x or y direction')
        return corr    