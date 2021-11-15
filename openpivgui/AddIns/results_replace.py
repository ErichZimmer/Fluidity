from openpivgui.AddIns.AddIn import AddIn
import openpiv.filters as piv_flt
import openpiv.validation as piv_vld
import openpiv.tools as piv_tls
import numpy as np


class results_replace(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "replace_results_addin"

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
        
        'horizontal_spacer14':
                [6095, 'h-spacer', None,
                 None,
                 None,
                 None],
        
        'replace_stuff':
            [6100, None, None, None,
             'Validation',
             None],

         'replace_frame':
            [6105, 'labelframe', None,
             None,
             'Replace components',
             None],
        
        'repl':
            [6111, 'bool', True, 'bind',
             'replace outliers',
             'Replace outliers.'],

        'repl_method':
            [6112, 'str', 'localmean',
             ('localmean', 'disk', 'distance'),
             'replacement method',
             'Each NaN element is replaced by a weighed average' +
             'of neighbours. Localmean uses a square kernel, ' +
             'disk a uniform circular kernel, and distance a ' +
             'kernel with a weight that is proportional to the ' +
             'distance.'],

        'repl_iter':
            [6113, 'int', 10, None,
             'number of iterations',
             'If there are adjacent NaN elements, iterative ' +
             'replacement is needed.'],

        'repl_kernel':
            [6114, 'int', 2, None,
             'kernel size [vec]',
             'Diameter of the weighting kernel in vectors.'],
        
        'val_exclude_mask_spacer':
            [6155, 'h-spacer', None,
             None,
             None,
             None],

         'validation_exlude_mask':
            [6160, 'bool', True, None,
             'exclude masked regions',
             'Exclude masked regions from validations.'],

        'validation_spacer':
            [6165, 'h-spacer', None,
             None,
             None,
             None],

        'validate_current':
            [6170, 'button_static_c', None, 
             "self.start_validations(index = self.index)",
             'Apply to current frame',
             None],

        'validate_all':
            [6175, 'button_static_c', None, 
             "self.start_validations()",
             'Apply to all frames',
             None],

        'view_validate':
            [6180, 'button_static_c', None, 
             "self.show(self.p['files_' + self.toggle][self.index])",
             'Update current frame',
             None],
    }
    
    
    def process_results(
        self,
        u, v, 
        mask = np.ma.nomask,
        flag = None,
        replace_method = 'localmean',
        replace_inter = 10,
        replace_kernel = 2,
    ):
        u[flag.astype(int).reshape(u.shape)==1] = np.nan
        v[flag.astype(int).reshape(v.shape)==1] = np.nan
        u = np.ma.masked_array(u, mask)
        v = np.ma.masked_array(v, mask)
        
        u, v = piv_flt.replace_outliers(
            u, v,
            method      = replace_method,
            max_iter    = replace_inter,
            kernel_size = replace_kernel
        )
        flag[flag == 1] = 2
        return u, v, flag
        

    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.postprocessing_methods.update(
            {"replace_results": self.process_results}
        )
        
        gui.toggled_widgets.update({
            'pass_repl':[
                'pass_repl_method',
                'pass_repl_iter',
                'pass_repl_kernel'
            ],

            'repl':[
                'repl_method',
                'repl_iter',
                'repl_kernel',
            ],
        })

        gui.toggled_buttons += [
            'validate_current',
            'validate_all',
            'view_validate',
        ]
