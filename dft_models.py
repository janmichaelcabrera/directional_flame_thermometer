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
    """
    Common methods among physical models
    """
    def __init__(self, *args, **kwargs):
        super(physical_models, self).__init__(*args, **kwargs)

    def sensitivity_coefficients(self, hf, hb):
        multiplier = 0.1
        sigma_eps = self.front_plate.epsilon*multiplier
        sigma_C = hf.C*multiplier
        sigma_n = hf.n*multiplier
        sigma_k = self.insul.k*multiplier
        sigma_rcpf = self.front_plate.rCp*multiplier
        sigma_rcpb = self.back_plate.rCp*multiplier
        sigma_l = self.plate_thickness*multiplier


        self.dq_deps = - self.front_plate.epsilon**(-2) * ((self.q_net + self.q_conv)*1000)
        self.dq_dC = 1/(hf.C * self.front_plate.epsilon) * (self.q_conv*1000)
        self.dq_dn = hf.n/(self.front_plate.epsilon*hf.Ra) * (self.q_conv_f*1000) + hb.n/(self.back_plate.epsilon*hb.Ra) * (self.q_conv_b*1000)
        self.dq_dn[np.isnan(self.dq_dn)] = 0
        self.dq_dk = 1/(self.insul.k*self.front_plate.epsilon) * (self.q_st_ins*1000)
        self.dq_drcpf = 1/(self.front_plate.epsilon * self.front_plate.rCp) * (self.q_st_f*1000)
        self.dq_drcpb = 1/(self.back_plate.epsilon * self.back_plate.rCp) * (self.q_st_b*1000)
        self.dq_dl = 1/(self.back_plate.epsilon*self.plate_thickness) * ((self.q_st_f + self.q_st_b)*1000)

        W_eps = (self.dq_deps*sigma_eps)**2
        W_C = (self.dq_dC*sigma_C)**2
        W_n = (self.dq_dn*sigma_n)**2
        W_k = (self.dq_dk*sigma_k)**2
        W_rcpf = (self.dq_drcpf*sigma_rcpf)**2
        W_rcpb = (self.dq_drcpb*sigma_rcpb)**2
        W_l = (self.dq_dl*sigma_l)**2

        W = W_eps + W_C + W_n + W_k + W_rcpf + W_rcpb + W_l

        S_eps = (self.dq_deps*sigma_eps)**2/W
        S_C = (self.dq_dC*sigma_C)**2/W
        S_n = (self.dq_dn*sigma_n)**2/W
        S_k = (self.dq_dk*sigma_k)**2/W
        S_rcpf = (self.dq_drcpf*sigma_rcpf)**2/W
        S_rcpb = (self.dq_drcpb*sigma_rcpb)**2/W
        S_l = (self.dq_dl*sigma_l)**2/W

        
        plt.figure()
        plt.plot(S_eps, label='dq_deps')
        plt.plot(S_C, label='dq_dC')
        plt.plot(S_n, label='dq_dn')
        plt.plot(S_k, label='dq_dk')
        plt.plot(S_rcpf , label='dq_drcpf')
        plt.plot(S_rcpb, label='dq_drcpb')
        plt.plot(S_l, '--', label='dq_dl')
        plt.legend(loc=0)
        plt.show()

    def plot_temps(self, out_directory=None, *args):
        """
        Notes
        ----------
            This method plots the DFT temperatures
        """
        ax = plt.subplot(111)
        plt.plot(self.time, self.T_f, label='Front Plate Temperature')
        plt.plot(self.time, self.T_b, label='Back Plate Temperature')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
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
            plt.savefig(out_directory+'temperatures.pdf')
            plt.close()

    def plot_net(self, out_directory=None, *args):
        """
        Notes
        ----------
            This method plots the net heat flux to the DFT
        """
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
        """
        Notes
        ----------
            This method plots the relevant heat flux components for determining the incident heat flux for a given model
        """
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
        """
        Notes
        ----------
            This method plots the incident heat flux to a DFT for a given model
        """
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
        """
        Parameters
        ----------
            out_directory: str
                Directory to where model results will be saved

        Notes
        ----------
            This method saves the output from the model to a csv file
        """
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
        """
        Parameters
        ----------
            T_f: array like
                Front plate temperature history of a DFT

            T_b: array like
                Back plate temperature history of a DFT

            time: array like
                Time vector pertaining to the temperature history of a DFT

            h_f: float, array like
                Scalar or vector of heat transfer coefficients in W/m^2-K for the front plate

            h_b: float, array like
                Scalar or vector of heat transfer coefficients in W/m^2-K for the back plate

            model: str
                Type of model to use for determining the net heat flux. Available: one_d_conduction, energy_storage_method, semi_infinite

            Kelvin: bool (optional)
                Whether or not the temperatures are in Celsius or Kelvin. If in Celsius, the data is converted to Kelvin

            L_i: float (optional)
                Insulation thickness in meters

            plate_thickness: float (optional)
                Front and back plate thickness in meters

            plate_material: obj (optional)
                Plate material object with necessary parameters for determining net heat flux

            insulation: obj (optional)
                Insulation material object with necessary parameters for determining net heat flux

            T_inf: float or array like (optional)
                Temperature of the fluid 'far' from the DFT. If not provided, T_inf is assumed to be constant and equal to the temperature at the beginning of the array. 

            T_sur: float or array like (optional)
                Temperature of the surroundings 'far' from the DFT. If not provided, T_sur is assumed to be constant and equal to the temperature at the beginning of the array

            nodes: int (optional)
                Number of nodes for finite difference one_d_conduction method

            run_on_init: bool (optional)
                Whether or not to run the model to obtain q_inc on initialization
        """
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
        """
        Notes
        ----------
            Calls one of the availbe methods for q_net and also determines q_inc based on:
                ..math : q_{inc} = \\frac{1}{\\epsilon}(q_{net} + q_{rad} + q_{conv})

        Returns
        ----------
            q_inc: array like
                Incident heat flux to the DFT
        """
        getattr(one_dim_conduction, self.model)(self)

        self.q_conv_f = (self.h_f * (self.T_f - self.T_inf))/1000
        self.q_conv_b = (self.h_b * (self.T_b - self.T_inf))/1000

        self.q_conv = self.q_conv_f + self.q_conv_b

        self.q_rad = (sigma * self.front_plate.epsilon * (self.T_f**4 - self.T_sur**4) + sigma * self.back_plate.epsilon * (self.T_b**4 - self.T_sur**4))/1000

        self.q_inc = 1/self.front_plate.epsilon*(self.q_net + self.q_rad + self.q_conv)
        return self.q_inc

    def one_d_conduction(self):
        """
        Returns
        ----------
            q_net: array like
                The net heat flux to the DFT based on a finite difference implicit method:
                    ..math : q_{net} = q_{st,f} + q_{st,b} + q_{st, ins}
                    ..math : q_{st,f} = \\rho_p c_p l_p \\frac{dT_f}{dt}
                    ..math : q_{st,b} = \\rho_p c_p l_p \\frac{dT_b}{dt}
                    ..math : q_{st, ins} = q_{in} - q_{out}; q_{in} = -k \\frac{dT_f}{dx}; q_{out} = -k \\frac{dT_b}{dx}
                    ..math : \\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}
                    ..math : T(x, t=0) = T_0, T(x=0, t) = T_f, T(x=L_i, t) = T_b
        """
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
                    Inverse of constructed banded matrix of size nodes by nodes
            """
            k = np.array([np.ones(nodes-1)*(-Fo), np.ones(nodes)*(1+2*Fo), np.ones(nodes-1)*(-Fo)])

            offset = [-1, 0, 1]

            A = diags(k, offset).toarray()

            return linalg.inv(A)

        # Instantiate implicit model parameters
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
        """
        Returns
        -----------
            q_net: array like
                The net heat flux to the DFT based on the energy storage method, ASTM E3057
                .. math : q_{net} = \\rho_p \\c_p \\frac{dT_f}{dt} + \\frac{k_{ins}(T_f - T_b)}{L_i} + \\rho_i c_i \\frac{dT_{ins}}{dt}
        """
        T_ins = (self.T_f + self.T_b)/2

        self.q_net = (self.front_plate.rCp*self.plate_thickness*np.gradient(self.T_f, self.time) + self.insul.k*(self.T_f - self.T_b)/self.L_i + self.insul.rCp*self.L_i*np.gradient(T_ins, self.time))/1000

        return self.q_net

    def semi_infinite(self):
        """
        Returns
        ----------
            q_net: array like
                The net heat flux to the DFT based on a semi infinite approximation to heat transfer within the insulation
                .. math: q_{net} = q_{st,f} + q_{ins}
                .. math: q_{ins} = \\left( \\frac{k_i \\rho_i c_i}{\\pi} \\right)^2 \\int_0^t \\frac{dT_f(\\lambda)}{d \\lambda} \\frac{d \\lambda}{\\sqrt{t - \\lambda}}
        """
        self.q_st_f = (self.front_plate.rCp * self.plate_thickness * np.gradient(self.T_f, self.time))/1000

        self.q_st_ins = np.zeros(len(self.time))
        dT = np.gradient(self.T_f[1:], self.time[1:])
        dt = self.time[1] - self.time[0]
        F1 = 1/np.sqrt(self.time[1:])

        self.q_st_ins[1:] = (np.sqrt((self.insul.k*self.insul.rCp)/np.pi)*np.convolve(dT, F1)[:len(dT)]*dt)/1000

        self.q_net = self.q_st_f + self.q_st_ins
        
        return self.q_net
