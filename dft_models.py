#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import linalg
import matplotlib.style as style
from .dft_properties import *
from .heat_transfer_coefficients import *
style.use(['seaborn'])

# Stefan-Boltzman constant (W/m2K4)
sigma = 5.6704E-8

class physical_models:
    def __init__(self, *args, **kwargs):
        super(physical_models, self).__init__(*args, **kwargs)

    def plot_net(self, out_directory=None, *args):
        ax = plt.subplot(111)
        plt.plot(self.time, self.q_net, label='Net Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (kW/m$^2$)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not out_directory:
            plt.show()
            plt.close()
        else:
            if os.path.isdir(out_directory) == True:
                pass
            else:
                os.mkdir(out_directory)
            plt.savefig(out_directory+'net_heat_flux.pdf')
            plt.close()

    def plot_components(self, out_directory=None, *args):
        ax = plt.subplot(111)
        plt.plot(self.time, self.q_inc, label='Incident Flux')
        plt.plot(self.time, self.q_net, label='Net Flux')
        if hasattr(self, 'q_st_f'):
            plt.plot(self.time, self.q_st_f, label='Front Stored')
        if hasattr(self, 'q_st_b'):
            plt.plot(self.time, self.q_st_b, label='Back Stored')
        if hasattr(self, 'q_st_ins'):
            plt.plot(self.time, self.q_st_ins, label='Insul Stored')
        plt.plot(self.time, self.q_conv, label='Convective Flux')
        plt.plot(self.time, self.q_rad, label='Radiative Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (kW/m$^2$)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not out_directory:
            plt.show()
            plt.close()
        else:
            if os.path.isdir(out_directory) == True:
                pass
            else:
                os.mkdir(out_directory)
            plt.savefig(out_directory+'component_heat_flux.pdf')
            plt.close()

    def plot_incident(self, out_directory=None, *args):
        ax = plt.subplot(111)
        plt.plot(self.time, self.q_inc, label='Incident Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (kW/m$^2$)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not out_directory:
            plt.show()
            plt.close()
        else:
            if os.path.isdir(out_directory) == True:
                pass
            else:
                os.mkdir(out_directory)
            plt.savefig(out_directory+'incident_heat_flux.pdf')
            plt.close()

    def plot_uncertainties(self):
        return 0

    def save_output(self, out_directory=None):
        if out_directory:
            if os.path.isdir(out_directory) == True:
                pass
            else:
                os.mkdir(out_directory)
            X = np.array([self.time, self.q_inc, self.q_net, self.q_conv, self.q_rad]).T
            header = 'Time (s), q_inc (kW/m2), q_net (kW/m2), q_conv (kW/m2), q_rad (kW/m2)'
            np.savetxt(out_directory+'heat_fluxes.csv', X, delimiter=',', header=header)
        else:
            raise NameError('Output directory not specified. Data not saved.')    

class one_dim_conduction(physical_models):
    def __init__(self, T_f, T_b, time, h_f, h_b, model='one_d_conduction', Kelvin=False, L_i=0.019, plate_thickness=0.0016, plate_material=stainless_steel, insulation=ceramic_fiber, T_inf=None, T_sur=None, nodes=10, run_on_init=True, *args, **kwargs):
        self.Kelvin = Kelvin
        if not Kelvin:
            self.T_f = T_f + 273
            self.T_b = T_b + 273
        else:
            self.T_f = T_f
            self.T_b = T_b

        self.time = time
        self.h_f = h_f
        self.h_b = h_b
        self.model = model
        self.plate_thickness = plate_thickness
        self.plate_material = plate_material
        self.L_i = L_i
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
        getattr(one_dim_conduction, self.model)(self)

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

        self.q_st_ins = (q_cond_if - q_cond_ib)/1000

        self.q_st_f = (self.front_plate.rCp * self.plate_thickness * np.gradient(self.T_f, self.time))/1000

        self.q_st_b = (self.back_plate.rCp * self.plate_thickness * np.gradient(self.T_b, self.time))/1000

        self.q_net = self.q_st_f + self.q_st_ins + self.q_st_b

        return self.q_net

    def energy_storage_method(self):
        T_ins = (self.T_f + self.T_b)/2
        self.q_net = (self.front_plate.rCp*self.plate_thickness*np.gradient(self.T_f, self.time) + self.insul.k*(self.T_f - self.T_b)/self.L_i + self.front_plate.rCp*self.plate_thickness*np.gradient(T_ins, self.time))/1000
        return self.q_net

    def semi_infinite(self):
        self.q_st_f = (self.front_plate.rCp * self.plate_thickness * np.gradient(self.T_f, self.time))/1000
        self.q_st_ins = np.zeros(len(self.time))
        dT = np.gradient(self.T_f[1:], self.time[1:])
        dt = self.time[1] - self.time[0]
        F1 = 1/np.sqrt(self.time[1:])
        self.q_st_ins[1:] = (np.sqrt((self.insul.k*self.insul.rCp)/np.pi)*np.convolve(dT, F1)[:len(dT)]*dt)/1000
        self.q_net = self.q_st_f + self.q_st_ins
        return self.q_net