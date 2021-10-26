from openpivgui.AddIns.AddIn import AddIn
import openpiv.filters as piv_flt
import openpiv.validation as piv_vld
import openpiv.tools as piv_tls
import numpy as np


class results_validate(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "validate_results_addin"

    # variables
    #########################################################
    # Place additional variables in the following sections. #
    # Widgets are created automatically. Don't care about   #
    # saving and restoring - new variables are included     #
    # automatically.                                        #
    #                                                       #
    # e.g.                                                  #
    #   **abbreviation**_**variable_name** =                #
    #       [**id over super group**, **variable_type**,    #
    #        **standard_value**,**hint**, **label**         #
    #        **tool tip**                                   #
    #########################################################
    variables = {
         # individual pass validations
        'validation_first_pass':
            [3400, None, None, None,
             'Validation',
             None],

        'validation1_frame':
            [3406, 'labelframe', None,
             None,
             'First pass validation',
             None],

        'fp_vld_global_threshold':
            [3410, 'bool', False, 'bind',
             'global threshold validation',
             'Validate first pass based on set global ' +
             'thresholds.'],

        'fp_MinU':
            [3411, 'float', -10.0, None,
             'min u',
             'Minimum U allowable component.'],

        'fp_MaxU':
            [3412, 'float', 10.0, None,
             'max u',
             'Maximum U allowable component.'],

        'fp_MinV':
            [3413, 'float', -10.0, None,
             'min v',
             'Minimum V allowable component.'],

        'fp_MaxV':
            [3414, 'float', 10.0, None,
             'max v',
             'Maximum V allowable component.'],
            
        'set_glob_val_second':
            [3415, 'button_static_c', None, 
             "self.other_set_first()",
             'Set to other passes',
             'Set current settings to other passes.'],
            
        'fp_reset_vel_limits':
            [3416, 'button_static_c', None, 
             "self.reset_vel_val(type_ = 'first pass')",
             'Reset velocity limits',
             None],
            
        'fp_std_spacer':
            [3417, 'h-spacer', None,
             None,
             None,
             None],

        'fp_vld_std_threshold':
             [3420, 'bool', False, 'bind',
             'standard deviation validation',
             'Remove vectors, if the the sum of the squared ' +
             'vector components is larger than the threshold ' +
             'times the standard deviation of the flow field.'],
            
        'fp_std_threshold':
            [3421, 'float', 8.0, None,
             'std threshold',
             'Standard deviation threshold.'],
            
        'fp_z_score_spacer':
            [3425, 'h-spacer', None,
             None,
             None,
             None],
            
        'fp_zscore':
            [3430, 'bool', False, 'bind',
             'z-score validation',
             'Discard vector, if the z-score of a vector is greater than '+
             'the the user defined threshold.'],
            
        'fp_zscore_threshold':
            [3431, 'float', 4, None,
             'threshold',
             'Vectors with a z-score higher than this value is discarded.'],
            
        'fp_median_spacer':
            [3435, 'h-spacer', None,
             None,
             None,
             None],
            
        'fp_local_med_threshold':
            [3440, 'bool', True, 'bind',
             'local median validation',
             'Discard vector, if the absolute difference with ' +
             'the local median is greater than the threshold. '],
            
        'fp_local_med':
            [3441, 'float', 3, None,
             'local median threshold',
             'Local median absolute difference threshold.'],

        'fp_local_med_size':
            [3442, 'int', 1, None,
             'local median kernel',
             'Local median filter kernel size.'],

         'fp_peak2peak_spacer':
            [3445, 'h-spacer', None,
             None,
             None,
             None],
            
         'fp_peak2peak_validation':
            [3450, 'bool', False, 'bind',
             'peak to peak validation',
             'Discard vector, if the peak-to-peak value is lower'+
             'than threshold'],
            
        'fp_peak2peak_threshold':
            [3451, 'float', 1, None,
             'threshold',
             'The signal to noise ratio threshold value..'],

        'fp_peak2peak_mask_width':
            [3452, 'int', 2, None,
             'mask half width [px]',
             'The half size of the region around the first'+
             'correlation peak to ignore for finding the second'+
             'peak.'],
            
        'fp_peak2mean_spacer':
            [3455, 'h-spacer', None,
             None,
             None,
             None],
            
         'fp_peak2mean_validation':
            [3460, 'bool', False, 'bind',
             'peak to mean validation',
             'Discard vector, if the peak-to-peak value is lower'+
             'than threshold.'],
            
        'fp_peak2mean_threshold':
            [3461, 'float', 1, None,
             'threshold',
             'The signal to noise ratio threshold value.'],
            
        'validation_other_pass':
            [3464, None, None, None,
             'Validation',
             None],

         'validation_frame':
            [3465, 'labelframe', None,
             None,
             'Other pass validations',
             None],
            
         'sp_vld_global_threshold':
            [3470, 'bool', False, 'bind',
             'global threshold validation',
             'Validate first pass based on set global ' +
             'thresholds.'],
            
        'sp_MinU':
            [3471, 'float', -10.0, None,
             'min u',
             'Minimum U allowable component.'],

        'sp_MaxU':
            [3472, 'float', 10.0, None,
             'max u',
             'Maximum U allowable component.'],

        'sp_MinV':
            [3473, 'float', -10.0, None,
             'min v',
             'Minimum V allowable component.'],

        'sp_MaxV':
            [3474, 'float', 10.0, None,
             'max v',
             'Maximum V allowable component.'],

        'sp_reset_vel_limits':
            [3475, 'button_static_c', None, 
             "self.reset_vel_val(type_ = 'other pass')",
             'Reset velocity limits',
             None],
            
        'sp_std_spacer':
            [3476, 'h-spacer', None,
             None,
             None,
             None],

        'sp_vld_std_threshold':
            [3480, 'bool', False, 'bind',
             'standard deviation validation',
             'Remove vectors, if the the sum of the squared ' +
             'vector components is larger than the threshold ' +
             'times the standard deviation of the flow field.'],
            
        'sp_std_threshold':
            [3481, 'float', 8.0, None,
             'std threshold',
             'Standard deviation threshold.'],

        'sp_zscore_spacer':
            [3485, 'h-spacer', None,
             None,
             None,
             None],
            
        'sp_zscore':
            [3490, 'bool', False, 'bind',
             'z-score validation',
             'Discard vector, if the z-score of a vector is greater than '+
             'the the user defined threshold.'],
            
        'sp_zscore_threshold':
            [3491, 'float', 4, None,
             'threshold',
             'Vectors with a z-score higher than this value is discarded.'],
            
        'sp_median_spacer':
            [3495, 'h-spacer', None,
             None,
             None,
             None],
            
        'sp_local_med_validation':
            [3500, 'bool', True, 'bind',
             'local median validation',
             'Discard vector, if the absolute difference with ' +
             'the local median is greater than the threshold.'],
            
        'sp_local_med':
            [3501, 'float', 3, None,
             'local median threshold',
             'Local median absolute difference threshold.'],

        'sp_local_med_size':
            [3502, 'int', 1, None,
             'local median kernel',
             'Local median filter kernel size.'],
            
        'sp_peak2peak_spacer':
            [3545, 'h-spacer', None,
             None,
             None,
             None],
            
        'sp_peak2peak_validation':
            [3550, 'bool', False, 'bind',
            'peak to peak validation',
            'Discard vector, if the peak-to-peak value is lower'+
            'than threshold'],
            
        'sp_peak2peak_threshold':
            [3551, 'float', 1, None,
            'threshold',
            'The signal to noise ratio threshold value..'],

        'sp_peak2peak_mask_width':
            [3552, 'int', 2, None,
            'mask half width [px]',
            'The half size of the region around the first'+
            'correlation peak to ignore for finding the second'+
            'peak.'],
            
        'sp_peak2mean_spacer':
            [3555, 'h-spacer', None,
            None,
            None,
            None],
            
        'sp_peak2mean_validation':
            [3560, 'bool', False, 'bind',
            'peak to mean validation',
            'Discard vector, if the peak-to-peak value is lower'+
            'than threshold.'],
            
       'sp_peak2mean_threshold':
           [3561, 'float', 1, None,
            'threshold',
            'The signal to noise ratio threshold value.'],
           
        
        
       'individual_pass_postprocessing':
           [3603, None, None, None,
           'PostProcessing',
            None],
            
       'piv_pass_postprocessing_frame':
           [3604, 'labelframe', None,
            None,
            'Post-processing',
            None],
            
       'piv_sub_frame3':
           [3605, 'sub_labelframe', None,
            None,
            'interpolation',
            None],
            
       'pass_repl':
           [3700, 'sub_bool', True, 'bind',
            'replace vectors',
            'Replace vectors between each pass.'],
            
       'pass_repl_method':
            [3701, 'sub', 'localmean',
            ('localmean', 'disk', 'distance'),
            'replacement method',
            'Each NaN element is replaced by a weighed average' +
            'of neighbours. Localmean uses a square kernel, ' +
            'disk a uniform circular kernel, and distance a ' +
            'kernel with a weight that is proportional to the ' +
            'distance.'],

        'pass_repl_iter':
            [3702, 'sub_int', 10, None,
             'number of iterations',
             'If there are adjacent NaN elements, iterative ' +
             'replacement is needed.'],

        'pass_repl_kernel':
            [3703, 'sub_int', 2, None,
            'kernel size [vec]',
            'Diameter of the NaN interpolation kernel in vectors.'],
        
        
        
            'vld': #oops
                [6000, None, None, None,
                 'PostProcessing1',
                 None],

            'vld_frame':
                [6005, 'labelframe', None,
                 None,
                 'Validate compoments',
                 None],

            'vld_global_thr':
                [6010, 'bool', False, 'bind',
                 'global threshold validation',
                 'Validate the data based on set global ' +
                 'thresholds.'],
            
            'MinU':
                [6012, 'float', -10.0, None,
                 'min u',
                 'Minimum U allowable component.'],

            'MaxU':
                [6013, 'float', 10.0, None,
                 'max u',
                 'Maximum U allowable component.'],

            'MinV':
                [6014, 'float', -10.0, None,
                 'min v',
                 'Minimum V allowable component.'],

            'MaxV':
                [6015, 'float', 10.0, None,
                 'max v',
                 'Maximum V allowable component.'],
            
            'set_vel_limits':
                [6016, 'button_static_c', None, 
                 "self.initialize_vel_interact()",
                 'Set velocity limits',
                 None],
            
            'reset_vel_limits':
                [6017, 'button_static_c', None, 
                 "self.reset_vel_val(type_ = 'post val')",
                 'Reset velocity limits',
                 None],
            
            'apply_glov_val_first_pass':
                [6018, 'button_static_c', None, 
                 "self.set_first_pass()",
                 'Apply to first pass',
                 None],
            
            'horizontal_spacer12':
                [6025, 'h-spacer', None,
                 None,
                 None,
                 None],
            
            'vld_global_std':
                [6030, 'bool', False, 'bind',
                 'standard deviation validation',
                 'Validate the data based on a multiple of the ' +
                 'standard deviation.'],

            'global_std_threshold':
                [6031, 'float', 5.0, None,
                 'std threshold',
                 'Remove vectors, if the the sum of the squared ' +
                 'vector components is larger than the threshold ' +
                 'times the standard deviation of the flow field.'],
            
            'horizontal_spacer13':
                [6035, 'h-spacer', None,
                 None,
                 None,
                 None],
            
            'vld_local_med':
                [6040, 'bool', True, 'bind',
                 'local median validation',
                 'Validate the data based on a local median ' +
                 'threshold.'],

            'local_median_threshold':
                [6041, 'float', 2, None,
                 'local median threshold',
                 'Discard vector, if the absolute difference with ' +
                 'the local median is greater than the threshold. '],
            
            'local_median_size':
                [6042, 'int', 1, None,
                 'local median kernel [vec]',
                 'Local median filter kernel distance from (0,0) in vectors.'],
            
            'horizontal_spacer14':
                [6095, 'h-spacer', None,
                 None,
                 None,
                 None],

            'repl':
                [6100, 'bool', True, 'bind',
                 'replace outliers',
                 'Replace outliers.'],

            'repl_method':
                [6101, 'str', 'localmean',
                 ('localmean', 'disk', 'distance'),
                 'replacement method',
                 'Each NaN element is replaced by a weighed average' +
                 'of neighbours. Localmean uses a square kernel, ' +
                 'disk a uniform circular kernel, and distance a ' +
                 'kernel with a weight that is proportional to the ' +
                 'distance.'],

            'repl_iter':
                [6102, 'int', 10, None,
                 'number of iterations',
                 'If there are adjacent NaN elements, iterative ' +
                 'replacement is needed.'],

            'repl_kernel':
                [6103, 'int', 2, None,
                 'kernel size [vec]',
                 'Diameter of the weighting kernel in vectors.'],

            'val_exclude_mask_spacer':
                [6105, 'h-spacer', None,
                 None,
                 None,
                 None],
            
             'validation_exlude_mask':
                [6110, 'bool', False, None,
                 'exclude masked regions',
                 'Exclude masked regions from validations.'],
            
            'validation_spacer':
                [6115, 'h-spacer', None,
                 None,
                 None,
                 None],
            
            'validate_current':
                [6120, 'button_static_c', None, 
                 "self.start_validations(index = self.index)",
                 'Apply to current frame',
                 None],
            
            'validate_all':
                [6130, 'button_static_c', None, 
                 "self.start_validations()",
                 'Apply to all frames',
                 None],
            
            'view_validate':
                [6150, 'button_static_c', None, 
                 "self.show(self.p['files_' + self.toggle][self.index])",
                 'Update current frame',
                 None],
    }

    
    def standard_z_score_validation(
        self,
        u,
        v, 
        threshold
    ):
        # if a mask is present, there could be some issues
        if isinstance(u, np.ma.MaskedArray):
            velocity_magnitude = u.filled(np.nan) ** 2 + v.filled(np.nan) ** 2        
        else:
            velocity_magnitude = u ** 2 + v ** 2
        ind = threshold < np.abs(((velocity_magnitude - np.nanmean(velocity_magnitude)) / np.nanstd(velocity_magnitude)))

        if np.all(ind):
            ind = ~ind  
            print('Oops, something went wrong with the standardized z score validation.')

        u[ind] = np.nan
        v[ind] = np.nan

        flag = np.zeros_like(u, dtype=bool)
        flag[ind] = True

        return u, v, flag
    
    
    def process_results(
        self,
        u, v, 
        mask = np.ma.nomask,
        flag = None,
        s2n = None, 
        s2n_val = False,
        s2n_thresh = 1.2,
        global_thresh = False,
        global_minU = -10,
        global_maxU = 10,
        global_minV = -10,
        global_maxV = 10,
        global_std = False,
        global_std_thresh = 5,
        z_score = False,
        z_score_thresh = 5,
        local_median = False,
        local_median_thresh = 2,
        local_median_kernel = 1,
        replace = True,
        replace_method = 'localmean',
        replace_inter = 10,
        replace_kernel = 2,
    ):
        try:
            flag[0]
        except:
            flag = np.full_like(u, 0)

        u = np.ma.masked_array(u, mask)
        v = np.ma.masked_array(v, mask)
        isSame = 0
        if s2n_val == True and s2n is not None:
            u, v, Flag = piv_vld.sig2noise_val(
                u, v, s2n,
                threshold=s2n_thresh)
            flag += Flag # consolidate effects of flag
            isSame += 1

        if global_thresh:
            u, v, Flag = piv_vld.global_val(
                u, v,
                u_thresholds=(global_minU, global_maxU),
                v_thresholds=(global_minV, global_maxV)
            )
            flag += Flag 
            isSame += 1

        if global_std:
            u, v, Flag = piv_vld.global_std(
                u, v, 
                std_threshold=global_std_thresh)
            flag += Flag
            isSame += 1

        if z_score:
            u, v, Flag = self.standard_z_score_validation(
                u, v,
                threshold = z_score_thresh
            )
            flag += Flag
            isSame += 1

        if local_median:
            u, v, Flag = piv_vld.local_median_val(
                u, v,
                u_threshold = local_median_thresh,
                v_threshold = local_median_thresh,
                size        = local_median_kernel)  
            flag += Flag
            isSame += 1

        if replace:
            u, v = piv_flt.replace_outliers(
                u, v,
                method      = replace_method,
                max_iter    = replace_inter,
                kernel_size = replace_kernel)
            isSame += 1

        if isSame != 0:
            isSame = False
        else:
            isSame = True

        return u, v, flag, isSame
        

    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.postprocessing_methods.update(
            {"validate_results": self.process_results}
        )
        
        gui.toggled_widgets.update({
            'fp_vld_global_threshold':[
                'fp_MinU',
                'fp_MaxU',
                'fp_MinV',
                'fp_MaxV',
                'set_glob_val_second',
                'fp_reset_vel_limits',
            ],
            'fp_vld_std_threshold':[
                'fp_std_threshold'
            ],
            'fp_zscore':[
                'fp_zscore_threshold',
            ],
            'fp_local_med_threshold':[
                'fp_local_med',
                'fp_local_med_size',
            ],
            'fp_peak2peak_validation':[
                'fp_peak2peak_threshold',
                'fp_peak2peak_mask_width',
            ],
            'fp_peak2mean_validation':[
                'fp_peak2mean_threshold',
            ],
            'sp_local_med_validation':[
                'sp_local_med',
                'sp_local_med_size',
            ],
            'sp_vld_std_threshold':[
                'sp_std_threshold'
            ],
            'sp_vld_global_threshold':[
                'sp_MinU',
                'sp_MaxU',
                'sp_MinV',
                'sp_MaxV',
                'sp_reset_vel_limits',
            ],
            'sp_zscore':[
                'sp_zscore_threshold',
            ],
            'sp_peak2peak_validation':[
                'sp_peak2peak_threshold',
                'sp_peak2peak_mask_width',
            ],
            'sp_peak2mean_validation':[
                'sp_peak2mean_threshold',
            ],
            'pass_repl':[
                'pass_repl_method',
                'pass_repl_iter',
                'pass_repl_kernel'
            ],
            'vld_global_thr':[
                'MinU',
                'MaxU',
                'MinV',
                'MaxV',
                'set_vel_limits',
                'reset_vel_limits',
            ],
            'vld_global_std':[
                'global_std_threshold'
            ],
            'vld_local_med':[
                'local_median_threshold',
                'local_median_size'
            ],
            'repl':[
                'repl_method',
                'repl_iter',
                'repl_kernel',
            ],
        })

        gui.toggled_buttons += [
            'set_glob_val_second',
            'fp_reset_vel_limits',
            'sp_reset_vel_limits',
            'set_vel_limits',
            'reset_vel_limits',
            'apply_glov_val_first_pass',
            'validate_current',
            'validate_all',
            'view_validate',
        ]
