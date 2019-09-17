#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import linalg
from .dft_properties import *
from .heat_transfer_coefficients import *

# Stefan-Boltzman constant (W/m2K4)
sigma = 5.6704E-8

class physical_models:
    def __init__(self, *args, **kwargs):
        super(physical_models, self).__init__(*args, **kwargs)

    def plot_results(self):
        plt.figure()
        plt.plot(self.time, self.q_inc)
        plt.show()
        return 0

    def plot_uncertainties(self):
        return 0

    def save_output(self, out_directory=None):
        return 0
    #     if os.path.isdir(out_directory) == True: pass
    #     else: os.mkdir(out_directory)
    #     np.save(out_directory+self.name+)

class one_dim_conduction(physical_models):
    def __init__(self, T_f, T_b, time, h_f, h_b, Kelvin=False, L_i=0.019, plate_thickness=0.0016, plate_material=stainless_steel, insulation=ceramic_fiber, T_inf=None, T_sur=None, nodes=10, run_on_init=True, *args, **kwargs):
        self.Kelvin = Kelvin
        if not Kelvin:
            self.T_f = T_f + 273
            self.T_b = T_b + 273
        else:
            self.T_f = T_f
            self.T_b = T_b

        self.time = time
        self.plate_thickness = plate_thickness
        self.plate_material = plate_material
        self.L_i = L_i
        self.h_f = h_f
        self.h_b = h_b
        self.insulation_material = insulation
        self.nodes = nodes

        if not T_inf:
            self.T_inf = self.T_f[0]
        else:
            self.T_inf = T_inf

        if not T_sur:
            self.T_sur = self.T_f[0]
        else:
            self.T_sur = T_sur

        ### Steel
        self.front_plate = self.plate_material(self.T_f)
        self.back_plate = self.plate_material(self.T_b)

        ### Insulation
        self.insul = self.insulation_material(self.T_inf)


        if run_on_init:
            self.q_inc = one_dim_conduction.incident_heat_flux(self)

    def incident_heat_flux(self):
        one_dim_conduction.one_d_conduction(self)

        self.q_conv = (self.h_f * (self.T_f - self.T_inf) + self.h_b * (self.T_b - self.T_inf))/1000

        self.q_rad = (sigma * self.front_plate.epsilon * (self.T_f**4 - self.T_sur**4) + sigma * self.back_plate.epsilon * (self.T_b**4 - self.T_sur**4))/1000

        self.q_inc = 1/self.front_plate.epsilon*(self.q_net + self.q_rad + self.q_conv)
        return self.q_inc

    def one_d_conduction(self):
        
        def construct_banded_matrix(Fo=5, nodes=10):
            """
            Inputs
            ----------
                Fo: float
                    Fourrier number for populating banded matrix for implicit finite difference method

                nodes: int
                    Number of nodes for finite difference method

                inv: bool
                    Returns inverse of banded matrix if True, returns just the banded matrix otherwise

            Returns
            ----------
                A^{-1}: array
                    Inverse of constructed banded matrix of size nodes by nodes if inv==True

                A: array
                    Banded matrix of size nodes by nodes if inv!=True
            """
            k = np.array([np.ones(nodes-1)*(-Fo), np.ones(nodes)*(1+2*Fo), np.ones(nodes-1)*(-Fo)])

            offset = [-1, 0, 1]

            A = diags(k, offset).toarray()

            return linalg.inv(A)

        # # Instantiate implicit model parameters
        delta_x = self.L_i/(self.nodes+1)
        time_steps = len(self.time)
        T_nodes_i = np.ones((time_steps, self.nodes))*self.T_inf

        delta_t = self.time[1] - self.time[0]

        # Insulation Fourrier number
        Fo_i = delta_t*self.insul.alpha/delta_x**2
        # Insulation implicit matrix construction and inversion
        A_i_inv = construct_banded_matrix(Fo=Fo_i, nodes=self.nodes)

        # Loop over time
        for t in range(time_steps-1):
            # Create temporary vector for current time step
            Temp_i = T_nodes_i[t].copy()
            # Update boundary condition at x=0
            Temp_i[0] = Temp_i[0] + Fo_i*self.T_f[t+1]
            # Update boundary condition at x=L
            Temp_i[-1] = Temp_i[-1] + Fo_i*self.T_b[t+1]
            # Solve for temperatures at next time step
            T_nodes_i[t+1] = A_i_inv @ Temp_i

        # q_{cond} = -k_i A_{i} \frac{dT_i}{dx}|_{x=0} (Watts)
        q_cond_if = -self.insul.k*(T_nodes_i[:,0] - self.T_f)/delta_x
        q_cond_ib = -self.insul.k*(T_nodes_i[:,-1] - self.T_b)/delta_x

        self.q_st_ins = q_cond_if - q_cond_ib

        self.q_st_f = self.front_plate.rCp * self.plate_thickness * np.gradient(self.T_f, self.time)

        self.q_st_b = self.back_plate.rCp * self.plate_thickness * np.gradient(self.T_b, self.time)

        self.q_net = (self.q_st_f + self.q_st_ins + self.q_st_b)/1000

        return self.q_net

# see if can use getattr to run proper method