from openpivgui.AddIns.AddIn import AddIn
from openpiv.preprocess import offset_image
from skimage import transform as tf
from matplotlib.transforms import Affine2D

class image_transformations(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "transformations_addin"

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
        'preproc1':
            [2000, None, None, None,
            'Transformations',
            None],

        'transform_frame':
            [2005, 'labelframe', None,
            None,
            'Transformations',
            None],

        'do_offset':
            [2011, 'bool', False, 'bind',
            'offset image',
            'Offset image in the x or y direction.'],

        'offset_x_axis':
            [2012, 'int', 0, None,
            'x axis',
            'Negative values offset the image to the left.'],

        'offset_y_axis':
            [2013, 'int', 0, None,
            'y axis',
            'Negative values offset the image to the up.'],

        'stretch_spacerd':
            [2020, 'h-spacer', None,
            None,
            None,
            None],

        'do_stretch':
            [2021, 'bool', False, 'bind',
            'stretch image',
            'Stretch image in the x or y direction.'],

        'stretch_x_axis':
            [2022, 'float', 0, None,
            'x axis',
            'Stretch image x axis.'],

        'stretch_y_axis':
            [2023, 'float', 0, None,
            'y axis',
            'Stretch image y axis.'],

        'rotation_spacer':
            [2030, 'h-spacer', None,
            None,
            None,
            None],
        
        'do_rotation':
            [2031, 'bool', False, 'bind',
            'rotate image',
            'rotate image from origin.'],

        'rotation_angle':
            [2032, 'float', 0, None,
            'angle',
            'Angle, in degrees, to rotate an image on a given origin.'],
        
        'rot_x_coord':
            [2032, 'int', 0, None,
            'origin (x)',
            'Origin for rotation operation.'],

        'rot_y_coord':
            [2033, 'int', 0, None,
            'origin (y)',
            'Origin for rotation operation.'],
        
        'skew_spacerd':
            [2040, 'h-spacer', None,
            None,
            None,
            None],
        
        'do_skew':
            [2041, 'bool', False, 'bind',
            'skew image',
            'skew image in the x or y direction.'],

        'skew_x_axis':
            [2042, 'float', 0, None,
            'x axis',
            'Skew image x axis.'],

        'skew_y_axis':
            [2043, 'float', 0, None,
            'y axis',
            'Stretch image y axis.'],
        
        'apply_second_only_shapcer':
            [2060, 'h-spacer', None,
            None,
            None,
            None],

        'apply_second_only':
            [2061, 'bool', False, None,
            'apply to second image only',
            'Apply to second image only.'],

        'transformation_order_spacer':
            [2070, 'h-spacer', None,
            None,
            None,
            None],

        'transformations_order':
            [2071, 'int', 1, (1,2,3,4,5),
            'interpolation order',
            'Interpolation order of the spline interpolator.'],
        
        'preview_transforms_spacer':
            [2080, 'h-spacer', None,
            None,
            None,
            None],

        'preview_phase_separation_current':
            [2081, 'button_static_c', None, 
            "self.show(self.p['files_' + self.toggle][self.index], preview = True, show_results = False)",
            'Preview current image',
            None],
    }


    def process_image(self, img, parameter):
        if parameter['do_offset'] == True:
            img = offset_image(img, parameter['offset_x_axis'], parameter['offset_y_axis'])
        if parameter['do_stretch'] == True:
            y_axis = parameter['stretch_y_axis'] + 1 # set so zero = no stretch
            x_axis = parameter['stretch_x_axis'] + 1
            if x_axis < 1: x_axis = 1
            if y_axis < 1: y_axis = 1
            img = tf.rescale(img, (y_axis, x_axis), order = parameter['transformations_order'])
            
        if parameter['do_rotation'] == True:
            matrix = Affine2D().rotate_deg_around(
                parameter['rot_x_coord'],
                parameter['rot_y_coord'],
                parameter['rotation_angle']
            ).get_matrix()
            img = tf.warp(img, inverse_map = matrix, order = parameter['transformations_order']) 
            
        if parameter['do_skew'] == True:
            matrix = Affine2D().skew_deg(xShear = parameter['skew_y_axis'], yShear = parameter['skew_x_axis']).get_matrix()
            img = tf.warp(img, inverse_map = matrix, order = parameter['transformations_order'])
        return img

    
    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.preprocessing_methods.update(
            {"transformations":
             self.process_image})

        gui.toggled_widgets.update({
            'do_offset':[
                    'offset_x_axis',
                    'offset_y_axis',
                ],
            'do_stretch':[
                    'stretch_x_axis',
                    'stretch_y_axis',
                ],
            'do_rotation':[
                'rotation_angle',
                'rot_x_coord',
                'rot_y_coord',
            ],
            'do_skew':[
                'skew_x_axis',
                'skew_y_axis',
            ],
        })

        gui.toggled_buttons += [
            'preview_phase_separation_current',
        ]