def button_list():
    buttons = [
        'remove_current_image',
        'apply_frequence_button',
        'toggle_frames_button',
        'preview_grid',
        'analyze_current',
        'analyze_all',
        'clear_results',
        'load_calib_img_button',
        'sel_ref_distance_button',
        'clear_calib_img_button',
        'get_calib_button',
        'apply_calib_button',
        'preview_cali',
        'average_results',
        'preview_vectors',
        'preview_contours',
        'preview_streamlines',
        'view_scatter_plot',
        'view_histogram_plot',
        'extract_select_corr',
        #'extract_select_points',
        #'extract_clear_points',
        #'extract_select_area',
        #'extract_clear_area',
        'extract_select_line',
        'extract_clear_line',
        'preview_statistics',
        'preview_preferences',
        'export1_current_button',
        'export1_all_button',
        'export2_current_button',
        'export2_all_button',
    ]
    return buttons

def widget_list():
    widgets = {
        'disable_autocorrelation':[
            'disable_autocorrelation_distance'
        ],
        'limit_peak_search_each':[
            'limit_peak_search_auto_each',
            'limit_peak_search_distance_each'
        ],
        'limit_peak_search_last':[
            'limit_peak_search_distance_last'
        ],
        'do_s2n':[
            's2n_method',
            's2n_mask',
        ],
        'manual_select_cores':[
            'cores',
        ],
        'contours_custom_density':[
            'contours_density',
        ],
        'contours_uniform_color':[
            'contour_color',
        ]
    }
    return widgets

def disabled_widgets():
    widgets = [
        'starting_ref_point',
        'ending_ref_point',
        'reference_dist',
        #'average_results',
        'statistics_vec_amount',
        'statistics_vec_time',
        'statistics_vec_time2',
        'statistics_vec_valid',
        'statistics_vec_invalid',
        'statistics_vec_masked',
        'statistics_s2n_mean',
        #'point_a_coords',
        #'point_b_coords',
        #'point_ab_distance',
    ]
    return widgets