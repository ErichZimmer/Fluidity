#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''A class for simple parameter handling.

This class is also used as a basis for automated widget creation
by OpenPivGui.
'''

import os
import json
from numpy import save, load
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

example_user_function = '''
print(
self.object_mask
)
'''

class OpenPivParams():
    '''A class for convenient parameter handling.

    Widgets are automatically created based on the content of the
    variables in the dictionary OpenPivParams.default.

    The entries in OpenPivParams.default are assumed to follow this
    pattern:

    (str) key:
        [(int) index, 
         (str) type, 
         value,
         (tuple) hints,
         (str) label,
         (str) help]

    The index is used for sorting and grouping, because Python 
    dictionaries below version 3.7 do not preserve their order. A 
    corresponding input widged ist chosen based on the type string:

        None:                    no widget, no variable, but a rider
        boolean:                 checkbox
        str[]:                   listbox
        text:                    text area
        other (float, int, str): entry (if hints not None: option menu)

    A label is placed next to each input widget. The help string is
    displayed as a tooltip.

    The parameter value is directly accessible via indexing the base
    variable name. For example, if your OpenPivParams object variable
    name is »my_settings«, you can access a value by typing:

    my_settings[key] 

    This is a shortcut for my_settings.param[key]. To access other 
    fields, use my_settings.label[key], my_settings.help[key] and so on.
    '''

    def __init__(self):
        # hard coded location of the parameter file in the home dir:
        self.params_fname = os.path.expanduser('~' + os.sep +
                                               'openpivgui_settings.json')
        self.session_file= os.path.expanduser(r'~' + os.sep +
                                                 'openpivgui_session.hdf5')
        # grouping and sorting based on an index:
        self.GENERAL = 1000
        self.PREPROC = 2000
        self.PIVPROC = 3000
        self.CALIBRATION = 5000
        self.VALIDATION = 6000
        self.POSTPROC = 7000
        self.PLOTTING = 8000
        self.LOGGING = 9000
        self.USER = 10000
        self.navi_position = 0
        # these are the default parameters, basis for widget creation:
        self.default = {
            ##########################################################
            # Place additional variables in the following sections.  #
            # Widgets are created automatically. Don't care about    #
            # saving and restoring - new variables are included      #
            # automatically.                                         #
            #                                                        #
            # most variable types:                                   #
            # dummy: dummy widget ignored in documentation           #
            # dummy2: dummy widget that is documented                #
            # None: creates notebook rider                           #
            # labelframe: creates a labeled frame                    #
            # sub_labelframe: creates a nested label frame           # 
            # Note to most widgets: if there is 'sub_' in the        #
            #   beginning, it will be placed in a nested label frame #
            # button_static_c: creates single button                 #
            # button_static_c2: creates two buttons side by size     #
            # h-spacer: horizontal spacer                            #
            # str: string widget                                     #
            # sub: string for nested label frames                    #
            # int: integer widget                                    # 
            # sub_int integer widget for nested frames               #
            # sub_int2: integer widget used for windowing            #
            # float: floating point widget                           #
            # sub_float: floating point widget for nested frames     #
            # bool: boolean checkbox widget                          #
            # sub_bool: boolean checkbox widget for nested frames    #
            # str[]: listbox                                         #
            ##########################################################
            
            # general and image import
            'load':
                [1000,
                 None,        # type None: This will create a rider.
                 None,
                 None,
                 'Load Menu',
                 None],
            
            'analysis': # if set false, the analysis will be stopped and cleared
                [1005, 'dummy', True,
                None,
                None,
                None],
            
            'files_a':
                [1006, 'dummy', [],
                None,
                None,
                None],
            
            'files_b':
                [1007, 'dummy', [],
                None,
                None,
                None],
            
            #'frames':
            #    [1008, 'dummy', [],
            #    None,
            #    None,
            #    None],
            
            'background_img':
                [1012, 'dummy', None,
                None,
                None,
                None],
            
            'loading_frame':
                [1015, 'labelframe', None,
                 None,
                 'Load images',
                 None],
            
            'load_img_button':
                [1020, 'button_static_c', None, 
                 "self.select_image_files()",
                 'Load images',
                 None],
            
            'img_list':
                [1022, 'str[]', [],
                None,
                'image list',
                None,
                ],
         
            'fnames':
                [1023,        # index, here: group GENERAL
                 'str[]',     # type
                 [],          # value
                 None,        # hint (used for option menu, if not None)
                 'filenames',  # label
                 '            '+
                 'Number of images:'
                ],       # help (tooltip)
            
            'remove_current_image':
                [1030, 'button_static_c', None, 
                 "self.remove_image(self.index_or)",
                 'Remove current image',
                 None],
            
            'sequence':
                [1060, 'str', '(1+2),(3+4)',
                 ('(1+2),(1+3)','(1+2),(2+3)', '(1+2),(3+4)'),
                 'sequence order',
                 'Select sequence order for evaluation.'],

            'skip':
                [1065, 'int', 1,
                 None,
                 'jump',
                 'Select sequence order jump for evaluation.' +
                 '\nEx: (1+(1+x)),(2+(2+x))'],
            
            'apply_frequence_button':
                [1070, 'button_static_c', None, 
                 "self.apply_frequencing()",
                 'Apply frequencing',
                 None],
            
            'load_results':
                [1100, None, None,
                None,
                'Load results',
                None],
            
            'loading_results_frame':
                [1115, 'labelframe', None,
                 None,
                 'Load results',
                 None],
            
            'select_results_button':
                [1120, 'button_static_c', None, 
                 "self.select_result_files()",
                 'Select files',
                 None],
            
            'skiprows':
                [1131, 'str', '0', None,
                 'skip rows',
                 'Number of rows skipped at the beginning of the file.'],

            'decimal':
                [1132, 'str', '.', None,
                 'decimal separator',
                 'Decimal separator for floating point numbers.'],

            'sep':
                [1133, 'str', 'tab', (',', ';', 'space', 'tab'),
                 'column separator',
                 'Column separator.'],
            
            'header_names':
                [1134, 'str', 'x,y,u,v,flag,s2n', ('x,y,u,v,flag,s2n',
                                                   'x,y,u,v,s2n,flag'),
                 'header names',
                 'Header names for loading the data in the correct order.'],
            
            'loaded_units_dist':
                [1135, 'str', 'px', ('px', 'm', 'cm'),
                 'distance units',
                 'Distance units of the loaded results.'],
            
            'loaded_units_vel':
                [1136, 'str', 'dt', ('dt', 's'),
                 'time units',
                 'Time units of the loaded results.'],
            
            'flip_spacer':
                [1140, 'h-spacer', None,
                 None,
                 None,
                 None],

            'flip_u':
                [1141, 'bool', 'False', None,
                 'flip u-component',
                 'flip u-component array when saving RAW results.'],

            'flip_v':
                [1142, 'bool', 'False', None,
                 'flip v-component',
                 'flip v-component array when saving RAW results.'],

            'invert_spacer':
                [1145, 'h-spacer', None,
                 None,
                 None,
                 None],

            'invert_u':
                [1150, 'bool', 'False', None,
                 'invert u-component',
                 'Invert (negative) u-component when saving RAW results.'],

            'invert_v':
                [1151, 'bool', 'False', None,
                 'invert v-component',
                 'Invert (negative) v-component when saving RAW results.'],
            
            'General':
                [1200, None, None,
                None,
                'General',
                None],
            
            'general_frame':
                [1215, 'labelframe', None,
                 None,
                 'General',
                 None],
            
            'warnings':
                [1220, 'dummy', True, None,
                 'Enable popup warnings',
                 'Enable popup warning messages (recommended).'],
            
            'use_blit':
                [1225, 'dummy', False, None,
                 'fast render',
                 'Use blitting to fast render artists.'],
            
            'pop_up_info':
                [1230, 'dummy', True, None,
                 'Enable popup info',
                 'Enable popup information messages (recommended).'],
            
            'save_on_exit':
                [1235, 'dummy', True, None,
                 'save on exit',
                 'Save current settings and session to the Users folder when exiting.'],
            
            'debug':
                [1235, 'bool', False, None,
                 'enable debug log',
                 'Used for debugging'],
            
            'preview_all':
                [1250, 'dummy', False, None,
                 'preprocess images at all times (very slow)',
                 'Not recommended. Preprocess images with selected settings at all times.'],
            
            # PIV rider
            'piv':
                [3000, None, None, None,
                 'PIV',
                 None],
            
            'piv_frame':
                [3005, 'labelframe', None,
                 None,
                 'PIV settings/analyze',
                 None],

            'first_pass_sub_frame':
                [3010, 'sub_labelframe', None,
                 None,
                 'Pass 1',
                 None],
            
            'pass_1':
                [3051, 'dummy', True, None,
                 'pass 1',
                 'Is always enabled.'],
            
            'first_pass_label':
                [3012, 'sub_label', None, None,
                ' Interrogration window [px]      Overlap [px]',
                None],
            
            'windowing_hint_label':
                [3015, 'dummy', None, None,
                None,
                None],
            
            'corr_window_1':
                [3025, 'dummy', '64', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the first pass.'],

            'overlap_window_1':
                [3030, 'dummy', '32', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap',
                 'Size of the overlap of the first pass in pixels. The overlap will then be ' +
                 'calculated for the following passes.'],
            
            'other_pass_sub_frame':
                [3035, 'sub_labelframe', None,
                 None,
                 'Pass 2...6',
                 None],

            'other_pass_label':
                [3036, 'sub_label', None, None,
                ' Interrogration window [px]      Overlap [px]',
                None],
            
            'pass_2':
                [3037, 'sub_bool2', True, None,
                 'pass 2',
                 'Enable a second pass.'],

            'corr_window_2':
                [3040, 'sub_int2', '32', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the second pass.'],

            'overlap_window_2':
                [3041, 'dummy', '16', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap size',
                 'Overlap size for the second pass.'],
            
            'pass_3_spacer':
                [3045, 'sub_h-spacer', None,
                 None,
                 None,
                 None],

            'pass_3':
                [3046, 'sub_bool2', True, None,
                 'pass 3',
                 'Enable a third pass.'],

            'corr_window_3':
                [3047, 'sub_int2', '16', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the third pass.'],

            'overlap_window_3':
                [3048, 'dummy', '8', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap size',
                 'Overlap size for the second pass.'],
            
            'pass_4_spacer':
                [3050, 'sub_h-spacer', None,
                 None,
                 None,
                 None],

            'pass_4':
                [3051, 'sub_bool2', False, None,
                 'pass 4',
                 'Enable a fourth pass.'],

            'corr_window_4':
                [3052, 'sub_int2', '16', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the fourth pass.'],
            
            'overlap_window_4':
                [3053, 'dummy', '8', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap size',
                 'Overlap size for the second pass.'],

            'pass_5_spacer':
                [3060, 'sub_h-spacer', None,
                 None,
                 None,
                 None],

            'pass_5':
                [3061, 'sub_bool2', False, None,
                 'pass 5',
                 'Enable a fifth pass.'],

            'corr_window_5':
                [3062, 'sub_int2', '16', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the fifth pass.'],
            
            'overlap_window_5':
                [3063, 'dummy', '8', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap size',
                 'Overlap size for the second pass.'],
            
            'pass_6_spacer':
                [3065, 'sub_h-spacer', None,
                 None,
                 None,
                 None],

            'pass_6':
                [3066, 'sub_bool2', False, None,
                 'pass 6',
                 'Enable a fifth pass.'],

            'corr_window_6':
                [3067, 'sub_int2', '16', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'interrogation window',
                 'Interrogation window for the fifth pass.'],
            
            'overlap_window_6':
                [3068, 'dummy', '8', ('128', '96', '64', '48', '32', '24', '16', '12', '8'),
                 'overlap size',
                 'Overlap size for the second pass.'],
            
            'algorithm': # Direct Cross Correlation broke on me :(
                [3072, 'str', 'Standard FFT Correlation', (
                    'Standard FFT Correlation',
                    'Normalized FFT Correlation',
                    'Phase Correlation',
                    #'Minimum Quadratic Differences',
                    'FFT-based Convolution',
                    ),
                 'algorithm',
                 'Algorithm used for the correlation.'#+
                 #'»Minimum Quadratic Difference« is experimental and should be used '+
                 #'in only one pass.'
                ],
            
            'gaussian_corr_sigma':
                [3083, 'dummy2', 1, None,
                 'gaussian phase correlation sigma',
                 'Used for gaussian phase correlation. Should be greater than 1, but not too big.'],
            
            'corr_method':
                [3075, 'str', 'circular',
                 ('circular', 'linear'),
                 'correlation method',
                 'Circular is no padding and linear is zero padding.'],

            'subpixel_method':
                [3080, 'str', 'gaussian',
                 ('centroid', 'gaussian', 'parabolic'),
                 'subpixel method',
                 'Three point function for subpixel approximation.'],
            
            'normalize_intensity':
                [3083, 'dummy', False, None,
                 'normalize intensity',
                 'Normalization by removing the mean intensity value per window and '+
                 'clipping the negative values to zero.'],
            
            'update_overlap':
                [3084, 'bool', True, 'bind3',
                 'uniform overlap',
                 'Updates overlap for visual purposes. When disabled, the individual overlap '+
                 'entries will then be used.'],
            
            'ensemble_correlation':
                [3085, 'bool', False, None,
                 'ensemble correlation (experimental)',
                 'Warning: Only single pass works properly.\n'+
                 'Average the correlation of all frames to obtain better results.'],
            
            'offset_corr_subpix':
                [3092, 'dummy', True, None, 
                 None,
                 None],
            
            'preview_grid':
                [3086, 'button_static_c', None, 
                 "self.preview_grid()",
                 'Preview grid',
                 None],
            
            'analyze_current':
                [3087, 'button_static_c', None, 
                 "self.start_processing(self.index)",
                 'Analyze current frame',
                 None],
            
            'analyze_all':
                [3088, 'button_static_c', None, 
                 "self.start_processing()",
                 'Analyze all frames',
                 None],
            
            'clear_results':
                [3089, 'button_static_c', None, 
                 "self.clear_results()",
                 'Clear all results',
                 None],

            'analyze_frame_index':
                [3095, 'dummy', None, None, # None = all frames
                 None,
                 None],
            
            'Advanced_algs':
                [3100, None, None, None,
                 'alg',
                 None],
            
            'alg_frame':
                [3105, 'labelframe', None,
                 None,
                 'Advanced settings',
                 None],
            
            'deform_sub_frame':
                [3150, 'sub_labelframe', None,
                 None,
                 'deformation/interpolation',
                 None],
            
            'deformation_method':
                [3151, 'sub', 'second image', (
                    'none',
                    'second image',
                    'symmetric'),
                 'deformation method',
                 'Window deformation method. '+
                 '»symmetric« deforms both first and second images (uses symmetric deformation). '+
                 '\n»second image« deforms the second image only.'],
            
            'grid_interpolation_order':
                [3152, 'sub_int', 3, (1, 2, 3, 4, 5),
                 'grid interpolation order',
                 'Interpolation order of the bivariate spline interpolator. \n' +
                 '»1« yields first order bi-linear interpolation \n'+
                 '»2« yields second order quadratic interpolation \n'+
                 'and so on...'],   
            
            'image_interpolation_order':
                [3153, 'sub_int', 1, (1, 2, 3, 4, 5),
                 'image interpolation order',
                 'Interpolation order of the bivariate spline interpolator. \n' +
                 '»1« yields first order bi-linear interpolation \n'+
                 '»2« yields second order quadratic interpolation \n'+
                 'and so on...\n»3« yields best results with expense of slower computation.'], 
            
            'image_deformation_order':
                [3154, 'sub_int', 1, (1, 2, 3, 4, 5),
                 'image deformation order',
                 'Deformation order of the spline interpolator. \n' +
                 '»1« yields first order linear interpolation \n'+
                 '»2« yields second order quadratic interpolation \n'+
                 'and so on...'], 
            
            
            'peak_search_sub_frame':
                [3155, 'sub_labelframe', None,
                 None,
                 'peak search',
                 None],
            
            'disable_autocorrelation':
                [3160, 'sub_bool', False, 'bind',
                 'disable autocorrelation (experimental)',
                 'Remove the autocorrelation peak by setting the center of' +
                 'the correlation plane to the correlation mean.'],
            
            'disable_autocorrelation_distance':
                [3161, 'sub_int', 1, None,
                 'distance from center [px]',
                 'Set the selected distance from center to the mean of the correlation plane.'],
            
            'limit_peak_search_all_spacer':
                [3165, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'limit_peak_search_each':
                [3170, 'sub_bool', False, 'bind',
                 'limit peak search (each pass)',
                 'Limit peak search area to theoretically obtain a better result.'],
            
            'limit_peak_search_auto_each':
                [3171, 'sub_bool', True, None,
                 'auto-size distance',
                 'Automatically size the distance from the center of '+
                 'the correlation plane to 1/4 of the correlation shape.'],
            
            'limit_peak_search_distance_each':
                [3172, 'sub_int', 8, None,
                 'distance from (0,0) [px]',
                 'The distance from the center of the correlation field used to search '+
                 'for the highest correlation peak.'],
            
            'limit_peak_search_last_spacer':
                [3175, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'limit_peak_search_last':
                [3180, 'sub_bool', False, 'bind',
                 'limit peak search (last pass)',
                 'Limit peak search area to theoretically obtain a better result. '+
                 'Automatically overides limit peak search for last pass. '],
            
            'limit_peak_search_distance_last':
                [3182, 'sub_int', 8, None,
                 'distance from (0,0) [px]',
                 'The distance from the center of the correlation field used to search '+
                 'for the highest correlation peak.'],
            
            'window_weigting_spacer':
                [3185, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'window_weighting':
                [3190, 'sub', 'boxcar', (
                    'boxcar',
                    'gaussian',
                    #'blackman',
                    'bartlett',
                    #'bohman',
                    'cosine',
                    #'hann',
                    'hamming',
                  ),
                 'window weighting',
                 'Window weighting function applied to the interrogation windows and '+
                 'correlation matrix (multiplied by autocorrelation of weighting function).'],
            
            'window_weighting_sigma':
                [3195, 'sub_float', 8.0, None,
                 'gaussian weight sigma',
                 'Sigma for gaussian window weighting.'],
            
            'extract_window_func':
                [3196, 'sub_button_static_c', None, 
                 "self.plot_window_function()",
                 'Plot window function',
                 None],
            
            'window_size_func':
                [3197, 'sub_int', 32, None,
                 'interrogation window [px]',
                 'Interrogation window size in pixels.'],
            
            'signal2noise_sub_h-frame':
                [3200, 'sub_labelframe', None,
                 None,
                 'signal to noise ratio',
                 None],
            
            'do_s2n':
                [3205, 'sub_bool', True, 'bind',
                 'calculate signal to noise ratio',
                 'Calculate signal to noise ratio. (slow)'],

            's2n_method':
                [3210, 'sub', 'peak2mean', ('peak2mean','peak2peak'),
                 'method:',
                 'Method used to calculate signal to noise ratio.'],
            
            's2n_mask':
                [3215, 'sub_int', 2, None,
                 'mask width [px from origin]',
                 'Used to find the second peak when doing peak to peak '
                 'signal to noise calculation.'],
            
            'multicore_frame':
                [3255, 'sub_labelframe', None,
                 None,
                 'batch processing',
                 None],

            'manual_select_cores':
                [3260, 'sub_bool', True, 'bind',
                 'manually select cores',
                 'Manually select cores. ' +
                 'If not selected, all available cores will be used.'],

            'cores':
                [3261, 'sub_int', 1, None,
                 'number of cores',
                 'Select amount of cores to be used for PIV evaluations.'],
            
            'use_FFTW':
                [3262, 'sub_bool', False, None,
                 'enable FFTW',
                 'Use FFTW to accelarate processing with precomputed fft objects '+
                 '(only avaliable in serial (1 core) processing.'],
            
            'batch_size':
                [3263, 'dummy', 100, (10, 50, 100, 250, 500, 1000),
                 'ensemble batch size [frames]',
                 'Batch size, in frames, for parallization. \nBatches are '+
                 'loaded in memory, so for a weaker the computer, the smaller '+
                 'the better.'],
            
            'piv_sub_frame5':
                [3865, 'sub_labelframe', None,
                 None,
                 'Other',
                 None],
            
            'validate_last_pass':
                [3870, 'sub_bool', False, None,
                 'validate last pass',
                 'Validate last pass.'],
            
            'exclude_masked_regions':
                [3890, 'sub_bool', False, None,
                 'exclude masked regions',
                 'Exclude masked regions for all passes.'],
            
            'data_probe':
                [3900, None, None, None,
                 'dp',
                 None],
            
            'data_probe_frame':
                [3905, 'labelframe', None,
                 None,
                 'Data probe',
                 None],
            
            'data_inten_frame':
                [3910, 'sub_labelframe', None,
                 None,
                 'extract image intensities',
                 None],
            
            #'data_image_axis':
            #    [3911, 'sub', 'x-axis', ('x-axis', 'y-axis'),
            #     'image axis',
            #     'Axis to extract image intensities.'],
                 
            #'view_image_inten_plot':
            #    [3912, 'sub_button_static_c', None, 
            #     "print('Not implemented')",
            #     'plot histogram',
            #     None],
            
            #'data_probe_extract_frame':
            #    [3921, 'sub_labelframe', None,
            #     None,
            #     'probe area',
            #     None],
            
            #'data_probe_window_size':
            #    [3922, 'sub', '32', None,
            #     'window size',
            #     'Data probe window size (effectively a single pass PIV evaluation)'],
                 
            #'view_image_statistics_plot':
            #    [3930, 'sub_button_static_c', None, 
            #     "print('Not implemented')",
            #     'enable data probe',
            #     None],
            
            'corr_sub_window':
                [3931, 'sub_labelframe', None,
                 None,
                 'extract correlation',
                 None],
            
            'extract_select_corr':
                [3932, 'sub_button_static_c', None, 
                 "self.initiate_correlation_plot()",
                 'select correlation window',
                 None],
            
            # calibration
            'calib':
                [5000, None, None, None,
                 'Calibtration',
                 None],
            
            'calib_frame':
                [5005, 'labelframe', None, None,
                 'Calibration',
                 None],
            
            #'dewarp':
            #    [5008, 'sub_labelframe', None,
            #     None,
            #     'vector correction',
            #     None],
            
            #'dewarp_select_roi,dewarp_reset_roi':
            #    [5010, 
            #    'sub_button_static_c2',
            #    None, 
            #    ["print('Not implemented')", "print('Not implemented')"],
            #    ['Select ROI', 'Clear ROI'],
            #    None],
            
            #'dewarp_min_dist':
            #    [5011, 'sub_int', 50, None,
            #     'min distance [px]',
            #     'Minimum distance between points.'],
            
            
            'pixel_to_real_cali':
                [5208, 'sub_labelframe', None,
                 None,
                 'pixel to object conversion',
                 None],
            
            'load_calib_img_button':
                [5210, 'sub_button_static_c', None, 
                 "self.calibration_load()",
                 'Load calibration image',
                 None],
            
            'sel_ref_distance_button':
                [5215, 'sub_button_static_c', None, 
                 "self.calibration_ref_dist()",
                 'Select reference distance',
                 None],
            
            'starting_ref_point':
                [5220, 'sub', ('0,0'), None,
                 'starting point (x,y)',
                 'Not recommended to modify.\n'+
                 'Starting point of the reference distance.'],
            
            'ending_ref_point':
                [5225, 'sub', ('0,0'), None,
                 'ending point (x,y)',
                 'Not recommended to modify.\n'+
                 'Ending point of the reference distance.'],
            
            'reference_dist':
                [5230, 'sub_float', 1, None,
                 'reference distance [px]',
                 'The reference distance, in pixels, between two points ' +
                 'on a reference image.'],
            
            'real_dist':
                [5235, 'sub_float', 1000, None,
                 'real distance [mm]',
                 'The real distance, in millimeters, between the two points selected '+
                 'in the reference distance.'],
            
            'time_step':
                [5240, 'sub_float', 1000, None,
                 'time step [ms]',
                 'The time step, in milliseconds, between two images.'],
            
            'scale':
                [5245, 'sub_float', 1, None,
                 'scale [m or cm/px]',
                 'Not recommended to modify.\n'+
                 'Calculated or input scale in m/px or cm/px.'],
            
            'scale_unit':
                [5248, 'sub', 'm', ('m', 'cm'),
                 'scale units',
                 'Scale calculation in m/dt or cm/dt.'],
            
            'clear_calib_img_button':
                [5250, 'sub_button_static_c', None, 
                 "self.calibration_clear()",
                 'Reset calibration',
                 None],
            
            'get_calib_button':
                [5255, 'sub_button_static_c', None, 
                 "self.get_calibration_scale()",
                 'Calculate scale',
                 None],
            
            'apply_calib_button':
                [5260, 'sub_button_static_c', None, 
                 "self.apply_calibration()",
                 'Apply to all frames',
                 None],
            
            'preview_cali':
                [5270, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            # plotting vectors
            'plt_vectors':
                [8000, None, None, None,
                 'Plot',
                 None],
            
            'plt_vec_frame':
                [8005, 'labelframe', None, 
                 None,
                 'Vectors',
                 None],
            
            'vectors_show':
                [8007, 'bool', True, None,
                 'show vectors',
                 'Plot vectors ontop of current plot.'],
            
            'vector_appearance_frame':
                [8010, 'sub_labelframe', None,
                 None,
                 'appearance',
                 None],
            
            'nthArrX':
                [8015, 'sub_int', 1, None,
                 'plot nth vector (x-axis)',
                 'Plot every nth vector for the x axis.'],
            
            'nthArrY':
                [8020, 'sub_int', 1, None,
                 'plot nth vector (y-axis)',
                 'Plot every nth vector for the y axis.'], 
            
            'vector_appearance_frame1.25':
                [8022, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'autoscale_vectors': # will need A LOT of working
                [8024, 'sub_bool', False, None,
                 'auto-scale vectors (experimental)',
                 'Autoscale width and scale of vectors.'],
            
            'vec_scale':
                [8025, 'sub_int', 100, None,
                 'scale',
                 'Velocity as a fraction of the plot width, e.g.: ' +
                 'm/s per plot width. Large values result in shorter ' +
                 'vectors.'],
            
            'vector_appearance_frame1.5':
                [8027, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'autowidth_vectors': # will need A LOT of working
                [8028, 'sub_bool', False, None,
                 'auto-scale vector width',
                 'Autoscale width of vectors.'],
            
            'vec_width':
                [8030, 'sub_float', 0.00085, None,
                 'shaft width',
                 'Line width as a fraction of the plot width.'],
            
            'vec_head_width':
                [8035, 'sub_float', 3.0, None,
                 'head width',
                 'Vector head width as a multiple of vector shaft width'],
            
            'vec_head_len':
                [8036, 'sub_float', 4.0, None,
                 'head length',
                 'Vector head length as a multiple of vector shaft width'],
            
            'vec_pivot':
                [8037, 'sub', 'tail', ('tip', 'middle', 'tail'),
                 'pivot',
                 'The part of the vector anchored to the (x, y) coordinate '+
                 'and rotates about this point.'],
            
            'vector_appearance_frame2':
                [8040, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'uniform_vector_color':
                [8044, 'sub_bool', True, None,
                 'uniform color',
                 'Make all vectors a uniform color.'],
            
            'valid_color': # hardcoded button
                [8045, 'dummy', '#00ff00', None,
                 None,
                 'Choose the color of the valid vectors'],
            
            'invalid_color':
                [8050, 'dummy', 'red', None,
                 None,
                 'Choose the color of the invalid vectors'],
            
            'vector_appearance_frame3':
                [8055, 'sub_h-spacer', None,
                 None,
                 None,
                 None],
            
            'show_masked_vectors': # algorithm didn't work, temporarly disabled
                [8060, 'sub_bool', False, None,
                'show masked vectors',
                'If enabled, masked vectors are shown in the selected visuals.'],
            
            'mask_vec_style':
                [8065, 'sub', 'x', ('x', '+', '.'),
                 'masked vector style',
                 'Define the style/visuals of masked vectors.'],
            
            'mask_vec':
                [8066, 'dummy', 'red', None,
                 None,
                 'Choose the color of the masked vectors'],
            
            'preview_vectors':
                [8070, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            #plot contours
            'plt_contours':
                [8200, None, None, None,
                 'Plot',
                 None],
            
            'plt_contours_frame':
                [8205, 'labelframe', None, 
                 None,
                 'Contours',
                 None],
            
            'contours_show':
                [8210, 'bool', False, None,
                 'show contours',
                 'Plot contours ontop of current plot.'],
            
            'contours_type':
                [8215, 'str', 'filled', ('unfilled', 'filled'),
                 'contour type',
                 'Fill contours with current color map.'],
            
            'contours_uniform':
                [8217, 'bool', False, None,
                 'interpolate levels (filled only)',
                 'Create leveled contours on filled contours.'],
            
            'contours_alpha':
                [8220, 'float', 1, None,
                 'contours transparency [0-1]',
                 'Define how transparent contour objects are where »0« is ' + 
                 'completley clear and »1« is completely opaque.'],
            
            'contours_thickness':
                [8225, 'float', 1, (0.5, 1, 1.5, 2, 2.5),
                 'line thickness [px]',
                 'Contour line thickness in pixels.'],
            
            'contours_line_style':
                [8230, 'str', 'solid', ('solid', 'dashed', 'dotted'),
                 'line style',
                 'Fill contours with current color map.'],
            
            'contours_custom_density':
                [8235, 'bool', False, 'bind',
                 'enable custom density',
                 'Enable independent density control.'],
            
            'contours_density':
                [8240, 'int', 10, None,
                 'contour density',
                 'The number of contour lines that should be plotted.'],
            
            'contours_uniform_color':
                [8245, 'bool', False, 'bind',
                 'uniform color',
                 'Uniform color.'],
            
            'contour_color':
                [8250, 'str', 'y', ('y', 'k', 'b', 'r', 'w'),
                 'contour color',
                 'Contour color.'],
            
            'preview_contours':
                [8255, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            #'contours_labels':
            #    [8250, 'bool', False, None,
            #     'show velocity labels',
            #     'Show velocity labels.'],
            
            
            
            # plot streamlines
            'plt_strmlines':
                [8300, None, None, None,
                 'Plot',
                 None],
            
            'plt_strmlines_frame':
                [8305, 'labelframe', None, 
                 None,
                 'Streamlines',
                 None],
            
            'streamlines_show':
                [8310, 'bool', False, None,
                 'show streamlines',
                 'Plot streamlines on top of current plot.'],
            
            'streamlines_density':
                [8320, 'str', '0.5, 1', None, 
                 'streamline density',
                 'Streamline density. Can be one value (e.g. 1) or a couple' +
                 ' of values for a range (e.g. 0.5, 1).'],
            
            'streamlines_thickness':
                [8325, 'float', 1, (0.25, 0.5, 1, 1.5, 2, 2.5),
                 'streamlines thickness [px]',
                 'Streamlines thickness in pixels.'],
            
            'integrate_dir':
                [8330, 'str', 'both', ('both', 'forward','backward'),
                 'streamline direction',
                 'Integrate the streamline in forward, backward or both ' +
                 'directions. default is both.'],
            
            'streamlines_arrow_style':
                [8335, 'str', '-', ('-', 'simple', 'fancy', 'wedge', '|-|', '-|>'),
                 'arrow style',
                 'Style of the arrow on the streamlines.'],
            
            'streamlines_arrow_width':
                [8340, 'float', '1.0', None,
                 'arrow size',
                 'Scaling factor for the arrow size.'],
            
            'streamlines_color':
                [8350, 'str', 'y', ('y', 'k', 'b', 'r', 'w'),
                 'streamlines color',
                 'Streamlines color.'],
            
            'preview_streamlines':
                [8355, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            
            # plot statistics 
            'plt_statistics':
                [8400, None, None, None,
                 'Plot',
                 None],
            
            'plt_statistics_frame':
                [8405, 'labelframe', None, 
                 None,
                 'Statistics',
                 None],
            
            'statistics_sub_frame':
                [8410, 'sub_labelframe', None,
                 None,
                 'vector statistics',
                 None],
            
            'statistics_vec_amount':
                [8420, 'sub_int', 0, None, 
                 'amount of vectors',
                 'Amount of vectors.'],
            
            'statistics_vec_time':
                [8425, 'sub_float', 0, None, 
                 'process time [s]',
                 'Process time in seconds.'],
            
            'statistics_vec_time2':
                [8430, 'sub_float', 0, None, 
                 'time per vector [ms]',
                 'Time per vector in milliseconds.'],
            
            'statistics_vec_valid':
                [8435, 'sub_float', 0, None, 
                 '% valid vectors',
                 'Percent of valid vectors excluding mask(s).'],
            
            'statistics_vec_invalid':
                [8440, 'sub_float', 0, None, 
                 '% invalid vectors',
                 'Percent of invalid vectors excluding mask(s).'],
            
            'statistics_vec_masked':
                [8445, 'sub_float', 0, None, 
                 '% masked vectors',
                 'Percent of masked vectors.'],
            
            'statistics_s2n_mean':
                [8447, 'sub_float', 0, None, 
                 'sig2noise ratio mean',
                 'Mean signal to noise ratio value.'],
            
            'view_statistics_plot':
                [8455, 'button_static_c', None, 
                 "self.plot_statistics_table()",
                 'Generate statistics chart',
                 None],
            
            'view_scatter_plot':
                [8466, 'button_static_c', None, 
                 "self.plot_scatter()",
                 'Generate scatter plot',
                 None],
            
            'view_histogram_plot':
                [8467, 'button_static_c', None, 
                 "self.plot_histogram()",
                 'Generate histogram plot',
                 None],
            
            'plot_grid':
                [8475, 'dummy', True, None, 
                 'grid', 
                 'adds a grid to the diagram.'],
            
            'plot_legend':
                [8480, 'dummy', True, None,
                 'legend', 
                 'adds a legend to the diagram.'],
            
            'statistics_subframe':
                [8485, 'sub_labelframe', None, 
                 None,
                 'histogram',
                 None],
            
            'plot_scaling': 
                [8486, 'sub', 'None', ('None',
                                      # 'logx',
                                       'logy',
                                      # 'loglog'
                                      ),
                 'axis scaling', 
                 'scales the axes. '+
                 #'logarithm scaling x-axis --> logx; '+
                 'logarithm scaling y-axis --> logy. ' #+
                 #'logarithm scaling both axes --> loglog.'
                ],
            
            'histogram_type':
                [8487, 'sub', 'bar', ('bar','step'), 
                 'histogram type', 
                 'Choose histogram type.'],
            
            'histogram_quantity':
                [8488, 'sub', 'magnitude', ('u-component','v-component','magnitude'),
                 'component',
                 'Component used for histogram plotting.'], 
            
            'histogram_bins':
                [8489, 'sub', 20, None,
                 'histogram number of bins',
                 'Number of bins (bars) in the histogram.'],
            
            'preview_statistics':
                [8495, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            # plot extractions 
            'plt_extractions':
                [8500, None, None, None,
                 'Plot',
                 None],
            
            'plt_extractions_frame':
                [8505, 'labelframe', None, 
                 None,
                 'Extractions',
                 None],
            
            #'distance_sub_window':
            #    [8520, 'sub_labelframe', None,
            #     None,
            #     'distance',
            #     None],
            
            #'extract_select_points,extract_clear_points':
            #    [8525, 
            #     'sub_button_static_c2',
            #     None, 
            #     ["print('Not implemented')", "print('Not implemented')"],
            #     ['select points', 'Clear points'],
            #     None],
            
            #'point_a_coords':
            #    [8531, 'sub', '0,0', None, 
            #     'point a',
            #     'Point a coords in x,y.'],
            
           # 'point_b_coords':
           #     [8532, 'sub', '0,0', None, 
           #      'point b',
           #      'Point b coords in x,y.'],
            
           # 'point_ab_distance':
           #     [8534, 'sub', '0', None, 
           #      'point a-b distance',
           #      'Point a-b distance.'],
            
           # 'extract_area_frame':
           #     [8535, 'sub_labelframe', None, 
           #      None,
           #      'area',
           #      None],
            
           # 'extract_select_area,extract_clear_area':
           #     [8540, 
           #      'sub_button_static_c2',
           #      None, 
           #      ["print('Not implemented')", "print('Not implemented')"],
           #      ['Select area', 'Clear area'],
           #      None],
            
            #'extract_select_method':
            #    [8541, 'sub', 'polygon', ('rectangle', 'polygon', 'lasso'), 
            #     'selection method', 
            #     'Method to select area to extract data.'],
            
            'extract_polyline_frame':
                [8545, 'sub_labelframe', None, 
                 None,
                 'profile',
                 None],
            
            'extract_select_line,extract_clear_line':
                [8550, 
                 'sub_button_static_c2',
                 None, 
                 ["print('Not implemented')", "print('Not implemented')"],
                 ['Add line', 'Clear line(s)'],
                 None],
            
            'extract_save_line,extract_load_line':
                [8555, 
                 'sub_button_static_c2',
                 None, 
                 ["print('Not implemented')", "print('Not implemented')"],
                 ['Save line(s)', 'Load line(s)'],
                 None],
            
            'extract_generate_Plot':
                [8560, 'sub_button_static_c', None, 
                 "print('Not implemented')",
                 'Generate plot',
                 None],
            
            'plot_title':
                [8580, 'dummy', 'Title', None, 
                 'diagram title', 
                 'diagram title.'],
            
            'lazy_label':
                [8582, 'label', None, None, 
                 'Extractions of area and profiles would be here.\n' +
                 'Algorithms are created for profile extractions,\nbut too lazy :P',
                 None],
            
            'Statistics_frame':
                [8583, 'sub_labelframe', None, 
                 None,
                 'Statistics',
                 None],
            
            'u_data':
                [8585, 'dummy', 'vx', None, 
                 'column name for u-component',
                 'column name for the u-velocity component.' +
                 ' If unknown watch labbook entry.'],
            
            'v_data':
                [8590, 'dummy', 'vy', None, 
                 'column name for v-component',
                 'column name for v-velocity component.' +
                 ' If unknown watch labbook entry.' +
                 ' For histogram only the v-velocity component is needed.'],
            
            
            
            
            
            
            # plot preferences
            'modify_plot_appearance':
                [8600, None, None, None,
                 'Plot',
                 None],
            
            'modify_plot_frame':
                [8605, 'labelframe', None, 
                 None,
                 'Preferences',
                 None],
            
            'exclusion_plotting_sub_frame':
                [8610, 'sub_labelframe', None,
                 None,
                 'Exclusions',
                 None],
            
            'roi_border_width':
                [8615, 'sub_int', 1, (1, 2, 3, 4),
                 'ROI border width',
                 'Define the border width of the ROI exclusion zone.'],
            
            'roi_line_style':
                [8620, 'sub', '--', ('-', '--', '-.', ':'),
                 'ROI border line style',
                 'Define the border line style of the ROI exclusion zone.'],
            
            'roi_border':
                [8630, 'dummy2', 'yellow', None,
                 'ROI border color',
                 'Define the border color of the ROI exclusion zone.'],
            
            'mask_fill':
                [8640, 'dummy2', '#960000', None,
                 'mask fill color',
                 'Define the fill color of the object mask polygons.'],
            
            'mask_alpha':
                [8660, 'sub_float', 0.75, None,
                 'mask transparency [0-1]',
                 'Define how transparent mask objects are where »0« is ' + 
                 'completley clear and »1« is completely opaque.'],
            
            'derived_subframe':
                [8700, 'sub_labelframe', None, 
                 None,
                 'Contours/color map',
                 None],
            
            'color_map':
                [8710, 'sub', 'viridis', ('viridis','jet','short rainbow',
                                          'long rainbow','seismic','autumn','binary'),
                 'Color map', 'Color map for streamline and contour plot.'],
            
            'velocity_color': 
                [8720, 'sub', 'magnitude', ('u-component', 'v-component',
                                            'magnitude', 
                                            'vorticity', 'enstrophy',
                                            'shear strain', 'normal strain',
                                            'divergence', 'acceleration', 'kinetic energy',
                                            'gradient du/dx', 'gradient du/dy',
                                            'gradient dv/dx', 'gradient dv/dy'),
                 'component: ',
                 'Set contours to selected velocity component.'],
            
            'color_levels':
                [8740, 'sub', '10', None, 
                 'number of color levels',
                 'Select the number of color levels for contour plot.'],
            
            'vmin':
                [8750, 'sub', '', None, 
                 'min velocity for colormap',
                 'minimum velocity for colormap (contour plot).'],
            
            'vmax':
                [8760, 'sub', '', None, 
                 'max velocity for colormap',
                 'maximum velocity for colormap (contour plot).'],

            'show_colorbar':
                [8760, 'sub_bool', False, None, 
                 'show colorbar (experimental)',
                 'Display colorbar ontop of current vector plot.\n'+
                 'Note: colorbar hides data that is underneath it.'],
            
            'function_plotting_sub_frame':
                [8770, 'sub_labelframe', None,
                 None,
                 'function plotting',
                 None],
            
            'func_cmap':
                [8771, 'sub', 'binary', ('viridis','autumn','binary'),
                 'color map', 
                 'Color map for correlation or window function plot'],
            
            'reverse_func_cmap':
                [8772, 'sub_bool', True, None,
                'reverse colormap',
                'Reverse correlation or window function colormap.'],
            
            'plot_func_3d':
                [8773, 'sub_bool', None, None,
                 'plot in 3D',
                 'Plot correlation or window function in 3D.'],
            
            'image_plotting_sub_frame':
                [8780, 'sub_labelframe', None,
                 None,
                 'image plotting',
                 None],

            'matplot_intensity_max':
                [8781, 'sub_int', 255, None,
                 'max reference intensity',
                 'Define a max reference intensity for the plotting of images.'],
            
            'matplot_intensity_min':
                [8782, 'sub_int', 0, None,
                 'min reference intensity',
                 'Define a min reference intensity for the plotting of images.'],
            
            'preview_preferences':
                [8785, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
            
            # lab-book
            'lab_book':
                [9000, None, None, None,
                 'Lab-Book',
                 None],

            'lab_book_content':
                [9010, 'text',
                 '',
                 None,
                 None,
                 None],
            
            'data_information':
                [9020, 'dummy', False, None, 'log column information',
                 'shows column names, if you choose a file at the ' +
                 'right side.'],

            # user-function
            'user_func':
                [10000, None, None, None,
                 'User-Function',
                 None],

            'user_func_def':
                [10010, 'text',
                 example_user_function,
                 None,
                 None,
                 None],
        
        # exporting current figure
            'export_1':
                [11000, None, None, None,
                 'Export_1',
                 None],
            
            'export_1_frame':
                [11005, 'labelframe', None, 
                 None,
                 'Export current plot figure',
                 None],
                
            'export1_fname':
                [11015, 'str', 'frame_{}', None,
                 'file name',
                 'What to call the exported plot(s).'],
            
            'export1_ext':
                [11018, 'str', 'jpeg', (
                    'eps', 
                    'jpeg', 
                    'jpg', 
                    'pdf', 
                    'pgf', 
                    'png', 
                    'svg', 
                    'tif', 
                    'tiff'
                 ),
                 'file type',
                 'File format/type.'],
            
            
            
            'export1_figsize':
                [11020, 'str', '11,8', None,
                 'figure size',
                 'Matplotlib figure size (same syntax as figsize).'],
            
            'export1_dpi':
                [11030, 'int', 300, None,
                 'dpi',
                 'Matplotlib dpi.'],
            
            #'export1_show_frame':
            #    [11035, 'bool', True, None,
            #     'show frame',
            #     'If disabled, removes matplotlib edge frame/border.'],
            
            #'export1_tight_layout': 
            #    [11038, 'bool', False, None,
            #     'tight layout',
            #     'Matplotlib tight layout.'],
            
            'export1_modified_img':
                [11040, 'bool', False, None,
                 'pre-process image',
                 'Pre-process image.'],
            
             'export1_current_button':
                [11050, 'button_static_c', None, 
                 "self.export_current_plot(index = self.index)",
                 'Export current frame',
                 None],
            
            'export1_all_button':
                [11070, 'button_static_c', None, 
                 "self.export_current_plot(index = None)",
                 'Export all frame(s)',
                 None],
            
            # exporting results as ASCI-II
            'export_2':
                [11100, None, None, None,
                 'Export_1',
                 None],
            
            'export_2_frame':
                [11105, 'labelframe', None, 
                 None,
                 'Export results as ASCI-II',
                 None],
                
            'export2_fname':
                [11115, 'str', 'frame_{}', None,
                 'file name',
                 'What to call the exported result(s).'],
            
            'export2_components':
                [11118, 'str', 'x,y,u,v,flag', ('x,y,u,v',
                                                'x,y,u,v,flag', 
                                                'x,y,u,v,flag,s2n',
                                                'x,y,u,v,s2n,flag'),
                 'components',
                 'Which components to save.\n' +
                 'flag values: 0 = valid; 1 = invalid'],
            
            'asci2_delimiter':
                [11119, 'str', 'tab', ('tab', 'space', ',', ';'),
                 'delimiter',
                 'Delimiter to differentiate the columns of the vector components.'],
            
            'asci2_extension':
                [11120, 'str', '.vec', ('.txt', '.vec', '.dat'),
                 'extension',
                 'File extension.'],
            
            'export2_set_masked_values':
                [11121, 'str', 'zero', ('NaN', 'zero'),
                 'set masked values to:',
                 'Set masked u and v components to selected option.\n' +
                 'Warning: setting values to Nan is experimental.'],
            
             'export2_current_button':
                [11130, 'button_static_c', None, 
                 "self.export_asci2(index = self.index)",
                 'Export current frame',
                 None],
            
            'export2_all_button':
                [11140, 'button_static_c', None, 
                 "self.export_asci2()",
                 'Export all frame(s)',
                 None],
            
            'export_3':
                [11200, None, None, None,
                 'Export_3',
                 None],
            
            'export_3_frame':
                [11205, 'labelframe', None, None,
                 'Export processed image pair(s)',
                 None],
            
            'export_3_label':
                [11210, 'label', None, None,
                'All images are converted to uint8',
                None],
            
            'export_3_h-spacer':
                [11211, 'h-spacer', None, None,
                 None, 
                 None,
                ],
            
            'export3_fname':
                [11215, 'str', 'frame_{}', None,
                 'file name',
                 'What to call the exported images(s).'],
            
            'export3_ext':
                [11218, 'str', 'bmp', (
                    'bmp',
                    #'jpeg', 
                    'jpg', 
                    'png', 
                    #'svg', 
                    'tif', 
                 ),
                 'file type',
                 'File format/type.'],
            
            'export3_convert_uint8':
                [11220, 'dummy', True, None,
                'convert to uin8',
                'Convert image pair to uint8 [0-256] dtype.'],
            
             'export3_current_button':
                [11250, 'button_static_c', None, 
                 "self.export_current_image(index = self.index)",
                 'Export current frame',
                 None],
            
            'export3_all_button':
                [11270, 'button_static_c', None, 
                 "self.export_current_image(index = None)",
                 'Export all frame(s)',
                 None],
        }

        # splitting the dictionary for more convenient access
        self.index = {}
        self.type = {}
        self.param = {}
        self.hint = {}
        self.label = {}
        self.help = {}
        self.add_parameters(self.default)

    def __getitem__(self, key):
        return self.param[key]

    def __setitem__(self, key, value):
        self.param[key] = value

    def load_settings(self, fname):
        """
            Read parameters from a JSON file.

            Args:
                fname (str): Path of the settings file in JSON format.

            Reads only parameter values. Content of the fields index,
            type, hint, label and help are always read from the default
            dictionary. The default dictionary may contain more entries
            than the JSON file (ensuring backwards compatibility).
        """
        try:
            f = open(fname, 'r')
            p = json.load(f)
            f.close()
        except BaseException:
            print('File not found: ' + fname)
        else:
            for key in self.param:
                if key in p:
                    self.param[key] = p[key]

    def dump_settings(self, fname):
        """
            Dump parameter values to a JSON file.

            Args:
                fname (str): A filename.

            Only the parameter values are saved. Other data like
            index, hint, label and help should only be defined in the
            default dictionary in this source code.
        """
        try:
            f = open(fname, 'w')
            json.dump(self.param, f)
            f.close()
        except BaseException:
            print('Unable to save settings: ' + fname)

    def generate_parameter_documentation(self, group=None):
        """
            Return parameter labels and help as reStructuredText def list.

            Parameters
            ----------
            group : int
                Parameter group.
                (e.g. OpenPivParams.PIVPROC)

            Returns
            -------
            str : A reStructuredText definition list for documentation.
        """
        s = ''
        for key in self.default:
            if (group < self.index[key] < group + 1000
                    and self.type[key] not in [
                        'labelframe',
                        'sub_labelframe',
                        'h-spacer',
                        'sub_h-spacer',
                        'dummy'
            ]):
                s = s + str(self.label[key]) + '\n' + "    " \
                    + str.replace(str(self.help[key]), '\n', '\n    ') + '\n\n'
        return s

    def add_parameters(self, param: dict) -> None:
        """
            splitting the dictionary for more convenient access
            outsourcing out of the init method was necessary to use it
            within the Add_In_Handler
            :param param: dictionary containing a list for each parameter
            :type param: dict
            :return: None
        """
        self.index.update(dict(zip(param.keys(),
                                   [val[0] for val in param.values()])))
        self.type.update(dict(zip(param.keys(),
                                  [val[1] for val in param.values()])))
        self.param.update(dict(zip(param.keys(),
                                   [val[2] for val in param.values()])))
        self.hint.update(dict(zip(param.keys(),
                                  [val[3] for val in param.values()])))
        self.label.update(dict(zip(param.keys(),
                                   [val[4] for val in param.values()])))
        self.help.update(dict(zip(param.keys(),
                                  [val[5] for val in param.values()])))
