from fluidity.AddIns.AddIn import AddIn
import openpiv.smoothn as piv_smt
import numpy as np


class results_modify(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "modify_results_addin"

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
        'piv_sub_frame4':
            [3705, 'sub_labelframe', None,
             None,
             'Smoothing',
             None],

        'smoothn_each_pass':
            [3800, 'sub_bool', True, 'bind',
            'smooth vector field',
            'Smooth each pass.'],

        'smoothn_first_more':
            [3801, 'sub_bool', True, None,
            'strengthen first pass',
            'Strengthen the smoothing on the first pass by x1.5.'],

        'robust1':
            [3802, 'sub_bool', False, None,
            'smoothen robust',
            'Theoretically minimizes influence of outlying data.'],

        'smoothn_val1':
            [3803, 'sub_float', 0.8, None,
            'smoothing strength',
            'Strength of smoothen script. Higher scalar number produces ' +
            'more smoothed data.'],
            

        
        
        'mdfy':
            [7000, None, None, None,
            'PostProcessing2',
            None],
            
        'mdfy_frame':
            [7005, 'labelframe', None,
             None,
            'Modify compoments',
             None],
        
        'mdfy_label_warning':
            [7007, 'label', None, None,
            '     Must be performed after validation',
             None],
            
        'mdfy_grid_spacer':
            [7009, 'h-spacer', None,
             None,
             None,
             None],
            
        'offset_grid':
            [7010, 'bool', False, 'bind',
            'offset grid',
            'Offset the grid by + or - units.'],
            
        'offset_x':
            [7011, 'float', 0.0, None,
             'offset x',
             'Offset the grid by + or - units on the x-axis'],
            
        'offset_y':
            [7012, 'float', 0.0, None,
             'offset y',
             'Offset the grid by + or - units on the y-axis'],
            
        'mdfy_vec_spacer':
            [7015, 'h-spacer', None,
             None,
             None,
             None],
            
        'modify_velocity':
            [7020, 'bool', False, 'bind',
             'add/subtract velocity components',
             'Add or subtract velocities for each velocity component'],
            
        'modify_u':
            [7021, 'float', 0.0, None,
             'u component',
             'Add (+) or subtract (-) entered value from the u component ' +
             'of the vector field.'],
            
        'modify_v':
            [7022, 'float', 0.0, None,
             'v component',
             'Add (+) or subtract (-) entered value from the v component ' +
             'of the vector field.'],
            
        'smoothn_spacer':
            [7075, 'h-spacer', None,
             None,
             None,
             None],
            
        'smoothn':
            [7080, 'bool', False, 'bind',
             'smooth data',
             'Smooth data using openpiv.smoothn.'],

        'robust':
            [7081, 'bool', False, None,
             'robust smoothing',
             'Theoretically minimizes influence of outlying data.'],

        'smoothn_val':
            [7082, 'float', 0.8, None,
             'smoothing strength',
             'Strength of the smooth script. Higher scalar number produces ' +
             'more smoothed results.'],
        
        'flip_coords_spacer':
            [7100, 'dummy', None,
             None,
             None,
             None],

        'mflip_x':
            [7105, 'dummy', False, None,
             'flip x-component',
             'flip x-component array when loading results.'],

        'mflip_y':
            [7108, 'dummy', False, None,
            'flip y-component',
            'flip y-component array when loading results.'],
            
        'mflip_spacer':
            [7120, 'h-spacer', None,
             None,
             None,
             None],

        'mflip_u':
            [7121, 'bool', False, None,
             'flip u-component',
             'flip u-component array when loading results.'],

        'mflip_v':
            [7122, 'bool', False, None,
             'flip v-component',
             'flip v-component array when loading results.'],

        'minvert_spacer':
            [7130, 'h-spacer', None,
             None,
             None,
             None],

        'minvert_u':
            [7131, 'bool', False, None,
             'invert u-component',
             'Invert (negative) u-component array when loading results.'],

        'minvert_v':
            [7132, 'bool', False, None,
             'invert v-component',
             'Invert (negative) v-component array when loading results.'],
        
        'mod_exclude_mask_spacer':
            [7285, 'h-spacer', None,
             None,
             None,
             None],
            
        'modification_exlude_mask':
            [7290, 'bool', True, None,
             'exclude masked regions',
             'Exclude masked regions from modifications.'],
            
        'apply_mdfy_spacer':
            [7305, 'h-spacer', None,
             None,
             None,
             None],
            
        'modify_current':
            [7315, 'button_static_c', None, 
             "self.start_modifications(index = self.index)",
             'Apply to current frame',
             None],
            
        'modify_all':
            [7320, 'button_static_c', None, 
             "self.start_modifications()",
             'Apply to all frames',
             None],
            
        'view_modify':
            [7330, 'button_static_c', None, 
             "self.show(self.p['files_' + self.toggle][self.index])",
             'Update current frame',
              None],
    }

    def process_results(
        self,
        x, y, u, v,
        mask = np.ma.nomask,
        offset_grid = False,
        offset_x = 0,
        offset_y = 0,
        modify_velocity = False,
        u_component = 0,
        v_component = 0,
        smooth = False,
        strength = 0.8,
        robust = False,
        flip_y = False,
        flip_x = False,
        flip_u = False,
        flip_v = False,
        invert_u = False,
        invert_v = False,
    ):
        u = np.ma.masked_array(u, mask)
        v = np.ma.masked_array(v, mask)
        isSame = 0

        if offset_grid != True:
            offset_x = 0
            offset_y = 0
        else:
            x += offset_x,
            y += offset_y,
            isSame += 1

        if modify_velocity:
            u += u_component
            v += v_component
            isSame += 1

        if smooth:
            u, _, _, _ = piv_smt.smoothn(
                u,
                s = strength,
                isrobust = robust
            )

            v, _, _, _ = piv_smt.smoothn(
                v,
                s = strength,
                isrobust = robust
            )
            isSame += 1

        if flip_y:
            y = np.flipud(y)
            isSame += 1
        if flip_x:
            x = np.fliplr(x)
            isSame += 1
        if flip_u:
            u = np.flip(u)
            isSame += 1
        if flip_v:
            v = np.flip(v)
            isSame += 1
        if invert_u:
            u *= -1
            isSame += 1
        if invert_v:
            v *= -1
            isSame += 1

        if isSame != 0:
            isSame = False
        else:
            isSame = True

        return(
            x,
            y,
            u,
            v,
            isSame
        ) 
        

    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.postprocessing_methods.update(
            {"modify_results": self.process_results}
        )
        
        gui.toggled_widgets.update({
            'smoothn_each_pass':[
                'smoothn_first_more',
                'robust1',
                'smoothn_val1',
            ],
            'smoothn':[
                'robust',
                'smoothn_val',
            ],
            #'offset_grid':[
            #    'offset_x',
            #    'offset_y',
            #],
            'modify_velocity':[
                'modify_u',
                'modify_v',
            ],
        })

        gui.toggled_buttons += [
            'modify_current',
            'modify_all',
            'view_modify',
        ]
