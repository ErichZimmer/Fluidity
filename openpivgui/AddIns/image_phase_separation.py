from openpivgui.AddIns.AddIn import AddIn
from openpiv.phase_separation import (opening_method, median_filter_method,
    khalitov_longmire, get_particles_size_array, get_size_brightness_map)

class image_phase_separation(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "phase_separation_addin"

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
        'preproc2':
            [2100, None, None, None,
            'Phase separation',
            None],

        'phase_separation_frame':
            [2105, 'labelframe', None,
            None,
            'Phase separation',
            None],

        'do_phase_separation':
            [2106, 'bool', False, 'bind',
            'enable phase separation (very slow)',
            'Enable two-phase separation to extract small particles.'],

        'phase_method':
            [2108, 'str', 'opening', ('opening', 'median', 'Khalitov-Longmire'),
            'separation method',
            'Method to separate big particles from small particles.'],

        'opening_sub_frame':
            [2110, 'sub_labelframe', None,
            None,
            'Opening method',
            None],

        'opening_kernel_size':
            [2111, 'sub_int', 11, None,
            'kernel size [px]',
            'Kernel size of the erosion/dilation filters.'],

        'opening_iterations':
            [2112, 'sub_int', 1, None,
            'iterations',
            'Number of iterations of the erosion filter.'],

        'opening_threshold':
            [2113, 'sub_float', 1.05, None,
            'threshold',
            'Used to mask big particles, default = 1.1\n'+
            'Mask condition is defined as:\n'+
            'pixel value > thresh_factor * local average intensity'],

        'median_sub_frame':
            [2120, 'sub_labelframe', None,
            None,
            'Median method',
            None],

        'median_kernel_size':
            [2121, 'sub_int', 11, None,
            'kernel size [px]',
            'Kernel size of the median filter.'],

        'kl_sub_frame':
            [2130, 'sub_labelframe', None,
            None,
            'Khalitov-Longmire method',
            None],

        'kl_scatter_plot':
            [2131, 'sub_button_static_c', None, 
            (
            "frame = tk.Toplevel(self, width = 600, height = 600)\n"
            "frame.attributes('-topmost', 'true')\n"
            "fig = Fig()\n"
            "fig_frame = ttk.Frame(frame)\n"
            "fig_frame.pack(side = 'left', fill = 'both', expand =True)\n"
            "fig_canvas = FigureCanvasTkAgg(fig, master = fig_frame)\n"
            "fig_canvas.draw()\n"
            "fig_canvas.get_tk_widget().pack(side='left', fill='x', expand='True')\n"
            "ax = fig.add_subplot(111)\n"
            "ax.hist(self.plotting_methods['get_size_distribution']("
            "piv_tls.imread(self.p['files_' + self.toggle][self.index])), 60)\n"
            "ax.set_yscale('log')\n"
            "ax.set_title('Particle size distribution')\n"
            "fig.canvas.draw()\n"
            "print('Plotted image particle size distribution')"),
            'Generate particle size plot',
            None],

        'kl_inten_plot':
            [2132, 'sub_button_static_c', None, 
            ("frame = tk.Toplevel(self, width = 600, height = 600)\n"
            "frame.attributes('-topmost', 'true')\n"
            "fig = Fig()\n"
            "fig_frame = ttk.Frame(frame)\n"
            "fig_frame.pack(side = 'left', fill = 'both', expand =True)\n"
            "fig_canvas = FigureCanvasTkAgg(fig, master = fig_frame)\n"
            "fig_canvas.draw()\n"
            "fig_canvas.get_tk_widget().pack(side='left', fill='x', expand='True')\n"
            "ax = fig.add_subplot(111)\n"
            "ax.imshow(self.plotting_methods['get_brightness_map']("
            "piv_tls.imread(self.p['files_' + self.toggle][self.index])), "
                 "interpolation='nearest', aspect='auto', origin='lower', cmap='jet')\n"
            "ax.set_title('Signal density')\n"
            "ax.set_ylabel('Size [px]')\n"
            "ax.set_xlabel('Brightness')\n" 
            "fig.canvas.draw()\n"
            "print('Plotted image signal density')"),
            'Generate signal density plot',
            None],

        'big_par_spacer':
            [2133, 'sub_h-spacer', None,
            None,
            None,
            None],

        'big_par_char':
            [2140, 'sub_label', None, None,
            ' Big particle characteristics:',
            None],

        'big_par_min_dist':
            [2141, 'sub_int', 100, None,
            'min particle size',
            'Minimum big particles size distribution.'],

        'big_par_max_dist':
            [2142, 'sub_int', 350, None,
            'max particle size',
            'Maximum big particles size distribution.'],

        'big_par_min_int':
            [2143, 'sub_int', 100, None,
            'min intensity',
            'Minimum big particles intensity.'],

        'big_par_max_int':
            [2144, 'sub_int', 180, None,
            'max intensity',
            'Maximum big particles intensity.'],

        'small_par_spacer':
            [2150, 'sub_h-spacer', None,
            None,
            None,
            None],

        'small_par_char':
            [2151, 'sub_label', None, None,
            ' Small particle characteristics:',
            None],

        'small_par_min_dist':
            [2152, 'sub_int', 25, None,
            'min particle size',
            'Minimum small particles size distribution.'],

        'small_par_max_dist':
            [2153, 'sub_int', 100, None,
            'max particle size',
            'Maximum small particles size distribution.'],

        'small_par_min_int':
            [2154, 'sub_int', 30, None,
            'min intensity',
            'Minimum small particles intensity.'],

        'small_par_max_int':
            [2155, 'sub_int', 100, None,
            'max intensity',
            'Maximum small particles intensity.'],

        'other_par_spacer':
            [2160, 'sub_h-spacer', None,
            None,
            None,
            None,],

        'kl_blur_kernel_size':
            [2161, 'sub_int', 1, None,
            'kernel size [px]',
            'Kernel size for blurring filter.'],

        'kl_sat':
            [2162, 'sub_int', 230, None,
            'saturation',
            'Saturation intensity for object pixels detection process.'],

        'kl_openeing_size':
            [2162, 'sub_int', 3, None,
            'opening kernel size [px]',
            'Stencil width for opening operation used to remove tiny regions from object pixels.\n'+
            'Use -1 to skip.'],

        'preview_phasesep_spacer':
            [2170, 'h-spacer', None,
            None,
            None,
            None],

        'preview_transforms_current':
            [2181, 'button_static_c', None, 
            "self.show(self.p['files_' + self.toggle][self.index], preview = True, show_results = False)",
            'Preview current image',
            None],
    }
    
    def process_image(self, img, parameter):
        if parameter['phase_method'] == 'opening':
            return opening_method(
                img, 
                kernel_size = parameter['opening_kernel_size'], 
                iterations = parameter['opening_iterations'], 
                thresh_factor = parameter['opening_threshold'],
            )[1]
        elif parameter['phase_method'] == 'median': 
            return median_filter_method(
                img,
                kernel_size = parameter['median_kernel_size']
            )[1]
        elif parameter['phase_method'] == 'Khalitov-Longmire':
            return khalitov_longmire(
                img,
                {
                    'min_size' : parameter['big_par_min_dist'],
                    'max_size' : parameter['big_par_max_dist'],
                    'min_brightness' : parameter['big_par_min_int'],
                    'max_brightness' : parameter['big_par_max_int'] 
                },
                {
                    'min_size' : parameter['small_par_min_dist'],
                    'max_size' : parameter['small_par_max_dist'],
                    'min_brightness' : parameter['small_par_min_int'],
                    'max_brightness' : parameter['small_par_max_int'] 
                },
                blur_kernel_size = parameter['kl_blur_kernel_size'],
                I_sat = parameter['kl_sat'],
                opening_ksize = parameter['kl_openeing_size'],
            )[1]

    
    def get_size_distribution(self, img):
        return get_particles_size_array(img)        
        
    def get_brightness_map(self, img):
        return get_size_brightness_map(img)
        
    def __init__(self, gui):
        super().__init__()
        # has to be the method which is implemented above
        gui.preprocessing_methods.update(
            {"phase_separation":
             self.process_image})
        
        gui.plotting_methods.update(
            {"get_size_distribution":
             self.get_size_distribution})
        
        gui.plotting_methods.update(
            {"get_brightness_map":
             self.get_brightness_map})
        
        gui.toggled_widgets.update({
            'do_phase_separation':[
                'phase_method',
                'opening_kernel_size',
                'opening_iterations',
                'opening_threshold',
                'median_kernel_size',
                'kl_scatter_plot',
                'kl_inten_plot',
                'big_par_char',
                'big_par_min_dist',
                'big_par_max_dist',
                'big_par_min_int',
                'big_par_max_int',
                'small_par_char',
                'small_par_min_dist',
                'small_par_max_dist',
                'small_par_min_int',
                'small_par_max_int',
                'kl_blur_kernel_size',
                'kl_sat',
                'kl_openeing_size',
                'preview_transforms_current',
            ]
        })

        gui.toggled_buttons += [
            'kl_scatter_plot',
            'kl_inten_plot',
            'preview_transforms_current',
        ]