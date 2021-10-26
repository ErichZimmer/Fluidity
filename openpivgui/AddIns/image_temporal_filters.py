from openpivgui.AddIns.AddIn import AddIn
from openpiv.tools import imread
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class image_temporal_filters(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "temporal_filters_addin"

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
        'preproc3':
            [2400, None, None, None,
            'Temporal filters',
            None],

        'background_frame':
            [2405, 'labelframe', None,
            None,
            'Temporal filters',
            None],

        'temp_label':
            [2406, 'label', None,
            None,
            ' Background images are not saved upon\n closing the GUI.',
            None],

        'background_filt_frame':
            [2407, 'sub_labelframe', None,
            None,
            'background removal',
            None],

        'background_status':
            [2408, 'dummy', None,
            None,
            None,
            None],

        'background_load,background_clear':
            [2410, 'sub_button_static_c2', None, 
            ["self.load_background_img()", 
             "self.background_frame_a = []; "+
             "self.backgroung_frame_b = []; "+
            "self.background_status_frame.config(bg = self.b_color); "+
            "self.ttk_widgets['background_status'].config("+
                "text = 'Background inactive',"+
                "bg = self.b_color); "
            "print('Cleared background images')"
            ],
            ['Load image', 'Clear image'],
            None],

        'background_type':
            [2420, 'sub', 'temporal minimum', (
                #'lowpass',
                'temporal maximum',
                'temporal mean', 
                'temporal minimum',
            ),
            'operation',
            'The algorithm used to generate the background which '+
            'is typically subtracted from the piv images. '
            ],

        'starting_frame':
            [2431, 'sub_int', 0, None,
            'starting frame',
            'Defining the starting frame of the background subtraction.'],

        'ending_frame':
            [2432, 'sub_int', -1, None,
            'ending frame',
            'Defining the ending frame of the background subtraction.'],

        'background_apply_type':
            [2433, 'sub', 'subtract', (
                'subtract',
                'multiply', 
                'divide',
            ),
            'application',
            'How the background image is applied to the image pairs.'
            ],

        'background_clip':
            [2434, 'sub_bool', True, None,
             'clip negative values',
             'Clip pixel intensity levels less than zero.'
            ],
        'background_generate':
            [2440, 'sub_button_static_c', None, 
            "self.generate_background_images()",
            'Generate background',
            None],

        'background_preview':
            [2450, 'sub_button_static_c', None, 
            "self.preview_background()",
            'Preview background',
            None],

        'background_save':
            [2460, 'sub_button_static_c', None, 
            "self.save_background()",
            'Save background',
            None],

        'preview_bg_current':
            [2470, 'button_static_c', None, 
            "self.show(self.p['files_' + self.toggle][self.index], preview = True, show_results = False)",
            'Preview current image',
            None],
    }

    def convert_(img):
        return img
#        img = img.astype(np.float64)
#        return img / img.max()
    
        
    def gen_background(
        self,
        image_list,
        method,
        sigma = 3,
        convert = convert_
    ):
        if method == 'temporal minimum':
            for img in image_list:
                # the original image is already included, so skip it in the for loop
                if img == image_list[0]:
                    background = imread(img)
                else:
                    image = imread(img)
                    background = np.min(np.array([background, image]), axis=0)
            return convert(background)

        elif method == 'temporal maximum':
            for img in image_list:
                if img == image_list[0]:
                    background = convert(imread(img))
                else:
                    image = convert(imread(img))
                    background = np.max(np.array([background, image]), axis=0)
            return background

        elif method == 'temporal mean':
            for img in image_list:
                if img == image_list[0]:
                    background = convert(imread(img))
                    dtype = background.dtype
                else:
                    image = convert(imread(img))
                    background = np.sum(np.array([background, image]), axis=0)
            return (background / len(image_list)).astype(dtype) 

        elif method == 'lowpass':
            for img in image_list:
                if img == image_list[0]:
                    img = convert(imread(img))
                    dtype = img.dtype
                    background = gaussian_filter(img, sigma = sigma)
                else:
                    img = convert(imread(img))
                    background += gaussian_filter(img, sigma = sigma)
            return (background / len(image_list)).astype(dtype) 

        else:
            print('Background generation algorithm not implemented')
        
        
    def process_image(self, img, background, parameter):
        if parameter['background_apply_type'] == 'subtract':
            img -= background
        elif parameter['background_apply_type'] == 'multiply':
            img *= background
        elif parameter['background_apply_type'] == 'divide':
            img /= background
        else:
            print('Background method not supported')
        if parameter['background_clip'] == True:
            img[img < 0] = 0
        return img

    
    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.preprocessing_methods.update(
            {"temporal_filters":
             self.process_image})
        
        gui.preprocessing_methods.update(
            {"generate_background":
             self.gen_background})
        
        #gui.toggled_widgets.update({
        #})


        gui.toggled_buttons += [
            'background_load',
            'background_clear',
            'background_generate',
            'background_preview',
            'background_save',
            'preview_bg_current',
        ]