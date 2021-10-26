from openpivgui.AddIns.AddIn import AddIn
from openpivgui.open_piv_gui_tools import coords_to_xymask
import numpy as np
import openpiv.preprocess as piv_pre
from skimage import exposure, filters, util
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.signal.signaltools import wiener as wiener_filter
from skimage.measure import find_contours, approximate_polygon, points_in_poly

class image_spatial_filters(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "spatial_filters_addin"

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
        'preproc4':
            [2500, None, None, None,
            'Filters',
            None],

        'filters_frame':
            [2505, 'labelframe', None,
            None,
            'Spatial filters',
            None],

        'invert':
            [2520, 'bool', False, None,
            'invert image',
            'Invert image (see skimage invert()).'],

        'contrast_stretch_spacer':
            [2525, 'h-spacer', None,
            None,
            None,
            None],

        'contrast_stretch_filter':
            [2530, 'bool', False, 'bind',
            'contrast stretch',
            'Simple percentile contrast stretching.'],

        'contast_stretch_low':
            [2531, 'float', 2.0, None,
            'lower limit [%]',
            'Lower percentile limit.'],

        'contast_stretch_high':
            [2532, 'float', 98.0, None,
            'upper limit [%]',
            'Upper percentile limit.'],

        'median_spacer':
            [2535, 'h-spacer', None,
            None,
            None,
            None],

        'median_filter':
            [2540, 'bool', False, 'bind',
            'median filter',
            'Median filter'],

        'median_filter_size':
            [2545, 'int', 5, (3,4,5,6,7,9),
            'filter size',
            'Define the size of the median filter.'],

        'high_pass_filter_spacer':
            [2550, 'h-spacer', None, 
            None,
            None,
            None],

        'high_pass_filter':
            [2551, 'bool', False, 'bind',
            'Gaussian high pass filter',
            'A simple subtracted Gaussian high pass filter.'],

        'hp_sigma':
            [2552, 'int', 7, None,
            'sigma',
            'Defining the sigma size of the subtracted gaussian filter in the ' + 
            'high pass filter (positive ints only).'],
        
        'hp_clip':
            [2553, 'bool', True, None,
            'clip negative values',
            'Clip pixel intensity levels less than zero.'],
        
        'CLAHE_spacer':
            [2560, 'h-spacer', None,
            None,
            None,
            None],

        'CLAHE':
            [2565, 'bool', False, 'bind',
            'CLAHE filter',
            'Warning: processes in 64 bit\nContrast Limited Adaptive Histogram Equalization filter ' +
            '(see skimage adapthist()).'],

        'CLAHE_auto_kernel':
            [2566, 'dummy', False, None,
            'automatic kernel sizing',
            'Have the kernel automatically sized to 1/8 width and height of the image.'],

        'CLAHE_kernel':
            [2567, 'int', 20, None,
            'kernel size',
            'Defining the size of the kernel for CLAHE.'],

        'CLAHE_contrast':
            [2568, 'dummy', 1, None,
            'contrast [1-100]',
            'Values 1-100 with higher number producing higher contrast.'],

        'local_norm_filter_spacer':
            [2569, 'h-spacer', None, 
            None,
            None,
            None],

        'local_norm_filter':
            [2570, 'bool', False, 'bind',
            'local normalization filter',
            'Simple local normalization filter.'],

        'local_norm_sigma1':
            [2571, 'float', 2, None,
            'sigma 1',
            'Local normalization sigma 1.'],

        'local_norm_sigma2':
            [2572, 'float', 2, None,
            'sigma 2',
            'Local normalization sigma 2.'],

        'local_norm_sub_z':
            [2573, 'str', 'zero', ('positive', 'zero', 'negative'),
            'set values <0 to..',
            'Set values less than zero to zero or positive.'],

        'intensity_threshold_spacer':
            [2585, 'h-spacer', None,
            None,
            None,
            None],

        'intensity_cap_filter':
            [2590, 'bool', False, 'bind',
            'intensity capping',
            'Simple global intesity cap filter. ' +
            'Masked pixels are set to the mean pixel intensity.'],

        'ic_mult':
            [2591, 'float', 2, None,
            'std multiplication',
            'Multiply the standard deviation of the pixel intensities ' +
            'to get a higher cap value.'],

        # 'intensity_clip_spacer':
            #     [2295, 'h-spacer', None,
            #      None,
            #      None,
            #      None],

        'intensity_clip': # no need for this currently
            [2600, 'dummy2', True, 'bind',
            'intensity clip',
            'Any intensity less than the threshold is set to zero.'],

        'intensity_clip_min':
            [2601, 'dummy2', 0, None,
            'min intensity',
            'Any intensity less than the threshold is set to zero with respect to ' +
            'the resized image intensities.'],

        'wiener_spacer':
            [2605, 'h-spacer', None,
            None,
            None,
            None],

        'wiener_filter':
            [2610, 'bool', False, 'bind',
            'wiener filter',
            'Warning: processes in 64 bit\nWiener denoise filter'],

        'wiener_filter_size':
            [2620, 'int', 7, (5,7,9,11,15),
            'filter kernel',
            'Define the kernel size of the wiener denoise filter.'],

        'Gaussian_lp_spacer':
            [2625, 'h-spacer', None,
            None,
            None,
            None],

        'gaussian_filter':
            [2630, 'bool', 'False', 'bind',
            'Gaussian low pass filter',
            'Standard Gaussian blurring filter (see scipy gaussian_filter()).'],

            'gf_sigma':
            [2641, 'int', 1, None,
            'sigma',
            'Defining the sigma size for gaussian blur filter.'],

        'apply_buttons_spacer':
            [2655, 'h-spacer', None,
            None,
            None,
            None],

        'preview_preproc_current':
            [2660, 'button_static_c', None, 
            "self.show(self.p['files_' + self.toggle][self.index], preview = True, show_results = False)",
            'Preview current image',
            None],

        'preproc5':
            [2700, None, None, None,
            'Exclusions',
            None],

        'exclusions_frame':
            [2705, 'labelframe', None,
            None,
            'Exclusions',
            None],

        'roi-xmin':
            [2710, 'dummy2', '', None,
            'x min',
            "Define left side of region of interest."],

        'roi-xmax':
            [2711, 'dummy2', '', None,
            'x max',
            "Define right side of region of interest."],

        'roi-ymin':
            [2712, 'dummy2', '', None,
            'y min',
            "Define top of region of interest."],

        'roi-ymax':
            [2713, 'dummy2', '', None,
            'y max',
            "Define bottom of region of interest."],

        'mask_sub_frame':
            [2721, 'sub_labelframe', None,
            None,
            'Manual masking',
            None],

        'mask_status':
            [2723, 'dummy', None,
            None,
            None,
            None],

        'mask_select,mask_clear':
            [2725, 
            'sub_button_static_c2',
            None, 
            ["self.mask_select()", "self.mask_clear()"],
            ['Add mask', 'Clear mask'],
            None],

        'mask_save,mask_load':
            [2730, 
            'sub_button_static_c2',
            None, 
            ["self.mask_save()", "self.mask_load()"],
            ['Save mask', 'Load mask'],
            None],

        'mask_apply_all':
            [2740, 'sub_button_static_c', None, 
            "self.apply_mask_all()",
            'Apply to select frames',
            None],

        'mask_load_external':
            [2745, 'sub_button_static_c', None, 
            "self.mask_load_external()",
            'Load image mask',
            None],

        'mask_selection':
            [2750, 'sub', '0:-1', None,
            'selected frames',
            'Select specific frames. Acceptable inputs: 0, 0:-1, 0:-1:2. '+
            '-1 targets ending frane.'],

        'mask_type':
            [2755, 'sub', 'polygon', ('polygon',
                                'rectangle',
                                'lasso'),
            'object mask type',
            'Define the type of the object mask.'],
        
        'use_dynamic_mask':
            [2760, 'sub_bool', False, None,
            'use dynamic masking',
            'Use dynamic masking on noisy or poor quality mask image(s).'],

        #'external_mask_kernel':
            #    [2723, 'sub_int', '4', None,
            #     'external mask divisor',
            #     'Divide the external mask by N x N to theoretically '+
            #     'obtain more accurate masks.'
            #],
        
        'dynamic_mask_sub_frame':
            [2800, 'sub_labelframe', None,
             None,
             'Dynamic masking',
             None],
        
        'do_dynamic_mask':
            [2805, 'sub_bool', False, 'bind',
            'enable dynamic masking',
            'Enables a basic form of dynamic masking.'],
        
        'dynamic_mask_type':
            [2808, 'sub', 'intensity', ('intensity', 'edges'),
            'method',
            'Method used to calculate mask.'],
        
        'dynamic_mask_kernel':
            [2810, 'sub_int', 7, None,
             'kernel size [px]',
             'Define size of the Gaussian blur and median filter in sigma and pixels.'],
        
        'dynamic_mask_threshold':
            [2815, 'sub_float', 0.005, None,
             'threshold',
             'A value of the threshold to segment the background from the object.'+
             'When a string is entered, threshold is replaced by sckimage.filter.threshold_otsu value.'],
        
        'dynamic_mask_preview':
            [2820, 'sub_button_static_c', None, 
            "self.preview_dynamic_mask()",
            'Preview mask',
            None],
        
        'mask_settings_sub_frame':
            [2900, 'sub_labelframe', None,
             None,
             'Mask settings',
             None],
        
        'invert_mask':
            [2901, 'dummy', False, None,
            'invert all masks',
            'Invert all masks for all frames.'
            ], 
        
        'masked_set_pixels':
            [2905, 'sub', 'mean intensity', (
                'original',
                'zero', 
                'mean intensity', 
                'max intensity',
                #'max value (256)',
                #'random'
            ),
            'set masked pixels to',
            'Define masked pixels.'],
        
        'find_mask_min':
            [2910, 'sub_int', 10, None,
            'min points in mask',
            'Threshold on minimum amount of points in mask when searching for valid masks.'],
        
        'find_mask_tolerance':
            [2915, 'sub_float', 1.5, None,
            'approximation tolerance',
            'Tolerance for down-sampling of mask coords when searching for valid masks.'],
    }

    def process_image(
        self,
        img, 
        parameter,
        preproc = True,
        preview = True,
        roi_xmin = '',
        roi_xmax = '',
        roi_ymin = '',
        roi_ymax = '',
    ):
    
        '''Starting the pre-processing chain'''    
        if preproc == True:
            if roi_xmin and roi_xmax and roi_ymin and roi_ymax != ('', ' '):
                try:
                    xmin=int(roi_xmin)
                    xmax=int(roi_xmax)
                    ymin=int(roi_ymin)
                    ymax=int(roi_ymax)
                    img = img[ymin:ymax,xmin:xmax]  
                except:
                    print('Invalid value in roi, ignoring filter')

            if parameter['invert'] == True:
                img = util.invert(img)

            if preview == True:
                if parameter['contrast_stretch_filter'] == True:
                    img = piv_pre.contrast_stretch(
                        img, 
                        parameter['contast_stretch_low'], parameter['contast_stretch_high']
                    )

                if parameter['median_filter'] == True:
                    img = median_filter(img, size = parameter['median_filter_size'])

                if parameter['high_pass_filter'] == True:
                    img = piv_pre.high_pass(img, parameter['hp_sigma'], parameter['hp_clip'])

                if parameter['CLAHE'] == True:
                    if parameter['CLAHE_auto_kernel']:
                        kernel = None
                    else:
                        kernel = parameter['CLAHE_kernel']
                    CLAHE_clip = parameter['CLAHE_contrast']
                    if CLAHE_clip < 1:
                        clip_limit = 0.01
                    elif CLAHE_clip > 100:
                        clip_limit = 1
                    else:
                        clip_limit = CLAHE_clip/100
                    img = exposure.equalize_adapthist(img, 
                                                      kernel_size = kernel, 
                                                      clip_limit  = clip_limit,
                                                      nbins       = 256).astype('float32')

                if parameter['local_norm_filter'] == True: # effective local normalization inspired from a stack overflow forum
                    flag = parameter['local_norm_sub_z']
                    sigma_1 = parameter['local_norm_sigma1']
                    sigma_2 = parameter['local_norm_sigma2']
                    img_blur = gaussian_filter(img, sigma_1)
                    high_pass = img - img_blur
                    img_blur = gaussian_filter(high_pass * high_pass, sigma_2)
                    den = np.power(img_blur, 0.5)
                    img = np.divide( # stops image from being all black
                        high_pass, den,
                        out = np.zeros_like(img),
                        where = (den != 0.0)
                    )
                    img[img == np.nan] = 0
                    if flag == 'zero':
                        img[img < 0] = 0 
                    elif flag == 'positive':
                        img = np.abs(img)
                    img = (img - img.min()) / (img.max() - img.min())

                # simple intensity capping
                if parameter['intensity_cap_filter'] == True:
                    img = piv_pre.instensity_cap(
                        img,
                        parameter['ic_mult'],
                    )

                # simple intensity clipping
                if parameter['intensity_clip'] == True:
                    img *= 255
                    img = piv_pre.intensity_clip(
                        img,
                        parameter['intensity_clip_min'],
                        None,
                        'clip'
                    )
                    img /= 255

                # wiener low pass filter
                if parameter['wiener_filter'] == True:
                    img = wiener_filter(img, (parameter['wiener_filter_size'], parameter['wiener_filter_size'])).astype('float32')

                # gausian low pass with gausian kernel
                if parameter['gaussian_filter'] == True:
                    img = gaussian_filter(img, sigma = parameter['gf_sigma'])

                #if binarize_intensity == True:
                #    img *= 255
                #    img[img < binarize_intensity_threshold] = 0
                #    img[img > binarize_intensity_threshold] = 255
                #    img /= 255

        return img
    
    
    def generate_mask(self, mask_img, parameter, proc = True):
        if proc == True:
            _, mask_img = piv_pre.dynamic_masking(
                mask_img, 
                parameter['dynamic_mask_type'],
                parameter['dynamic_mask_kernel'],
                parameter['dynamic_mask_threshold']
            )
        mask_coords = []
        for contour in find_contours(mask_img, 0):
            coords = approximate_polygon(contour, tolerance=parameter['find_mask_tolerance'])
            if len(coords) > parameter['find_mask_min']:
                mask_coords.append(coords)
        new_mask = []
        for mask in mask_coords:
            flipped_mask = []
            for coords in mask:
                flipped_mask.append((coords[1], coords[0])) # fixes a naughty bug, can't process arrays
            new_mask.append(flipped_mask)
        return new_mask
        
    
    def apply_mask_to_image(self, img, mask_coords, parameter):
        if parameter['masked_set_pixels'] != 'original':
            img_y, img_x = np.meshgrid(
                np.arange(0,img.shape[1]),
                np.arange(0,img.shape[0])
            )
            xymask = coords_to_xymask(
                img_y,
                img_x,
                mask_coords
            ).reshape(img.shape)
            if parameter['masked_set_pixels'] == 'zero':
                mask_val = 0
            elif parameter['masked_set_pixels'] == 'mean intensity':
                mask_val = int(np.mean(img))
            elif parameter['masked_set_pixels'] == 'max intensity':
                mask_val = np.max(img)
            elif parameter['masked_set_pixels'] == 'max value (256)':
                mask_val = 2**8
            elif parameter['masked_set_pixels'] == 'random':
                mask_val = np.random.rand(img.shape[0], img.shape[1])
                max_val = img.max()
                if max_val > 2**8:
                    max_val = 2**8
                mask_val = np.uint8(mask_val * max_val)[xymask==1]
            img[xymask==1] = mask_val
        return img
        
        

    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.preprocessing_methods.update(
            {"spatial_filters": self.process_image}
        )
        gui.preprocessing_methods.update(
            {"generate_mask": self.generate_mask}
        )
        gui.preprocessing_methods.update(
            {"apply_mask": self.apply_mask_to_image}
        )
        
        gui.toggled_widgets.update({
            'contrast_stretch_filter':[
                'contast_stretch_low',
                'contast_stretch_high'
            ],
             'median_filter':[
                'median_filter_size'
            ],
            'CLAHE':[
            #    'CLAHE_auto_kernel', 
                'CLAHE_kernel',
            #    'CLAHE_contrast'
            ],
            'high_pass_filter':[
                'hp_sigma',
                'hp_clip',
            ],
            'local_norm_filter':[
                'local_norm_sigma1',
                'local_norm_sigma2',
                'local_norm_sub_z',
            ],
            'intensity_cap_filter':[
                'ic_mult'
            ],
            #'intensity_clip':[
            #    'intensity_clip_min'
            #],
            'wiener_filter':[
                'wiener_filter_size'
            ],
            'gaussian_filter':[
                'gf_sigma'
            ],
            'do_dynamic_mask':[
                'dynamic_mask_type',
                'dynamic_mask_kernel',
                'dynamic_mask_threshold',
                'dynamic_mask_preview',
            ],
        })

        gui.toggled_buttons += [
            'preview_preproc_current',
            'mask_select',
            'mask_clear',
            'mask_save',
            'mask_load',
            'mask_apply_all',
            'mask_load_external',
        ]
