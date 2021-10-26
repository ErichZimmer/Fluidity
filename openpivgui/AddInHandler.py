import tkinter as tk
from tkinter import ttk
import os
from functools import partial
import importlib

add_ins = {}
imported_add_ins = {}


def init_add_ins(gui):
    """
        In this method, the parameters that are already part of the GUI
        (loaded from OpenPIVParams) are extended by the parameters of
        the selected AddIns.
    :param gui: active instance of the OpenPivGui class
    :type gui: obj(OpenPivGui)
    :return:
    """
    # get the parameters which are already part of the gui
    parameters = gui.get_parameters()
    # get the add_ins loaded in the last session
    add_ins_to_be_included = [
        'image_transformations',  'image_phase_separation', 
        'image_temporal_filters', 'image_spatial_filters',
        'results_validate',       'results_modify'
    ]
    # Iterate through the selected add-ins, creating an instance of the
    # add-ins and reading out the variables of the class. These are then
    # appended to the parameter object.
    for add_in in add_ins_to_be_included:
        add_in_file = importlib.import_module("openpivgui.AddIns." + add_in)
        add_in_instance = getattr(add_in_file, add_in)(gui)
        imported_add_ins.update({add_in: add_in_instance.get_variables()})
        parameters.add_parameters(add_in_instance.get_variables())