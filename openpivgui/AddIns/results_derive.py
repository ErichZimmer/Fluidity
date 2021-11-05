from openpivgui.AddIns.AddIn import AddIn
import openpiv.smoothn as piv_smt
import numpy as np


class results_derive(AddIn):
    """
        Blueprint for developing own methods and inserting own variables
        into the already existing PIV GUI via the AddIn system
    """

    # description for the Add_In_Handler textarea
    addin_tip = "This is the description of the advanced filter addin which " \
                "is still missing now"

    # has to be the add_in_name and its abbreviation
    add_in_name = "derive_results_addin"

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
        'derive':
            [7400, None, None, None,
            'PostProcessing3',
            None],
            
        'derive_frame':
            [7405, 'labelframe', None,
             None,
            'Derive compoments',
             None],
        
        'average_results':
             [7410, 'button_static_c', None, 
             "self.average_results()",
             'Average results',
             None],
    }
        

    def __init__(self, gui):
        super().__init__()

        gui.toggled_buttons += [
            'average_results',
        ]
