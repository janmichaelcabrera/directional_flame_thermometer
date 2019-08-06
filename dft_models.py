#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
from .dft_properties import *
from .heat_transfer_coefficients import *

class physical_models:
    def __init__(self, *args, **kwargs):
        super(physical_models, self).__init__(*args, **kwargs)

    def plot_results(self):
        return 0

    def plot_uncertainties(self):
        return 0

    def save_output(self, out_directory=None):
        return 0
    #     if os.path.isdir(out_directory) == True: pass
    #     else: os.mkdir(out_directory)
    #     np.save(out_directory+self.name+)

class one_dim_conduction(physical_models):
    def __init__(self, T_f, T_r, time, Kelvin=False, L=0.0016, plate_material='stainless_steel', h='natural_convection', *args, **kwargs):
        self.Kelvin = Kelvin
        if not Kelvin:
            self.T_f = T_f + 273
            self.T_r = T_r + 273
        else:
            self.T_f = T_f
            self.T_r = T_r

        self.time = time
        self.plate = plate_material
        self.h = h

    def run_model(self):
        return 0
