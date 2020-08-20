#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import linalg
import scipy.stats as stats
import pandas as pd
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

    def one_d_uncertainty(self):
        """
        Attributes
        ----------
            dq_deps: float (vector)
                Partial derivative of model with respect to epsilon
                .. math :: \\frac{\\partial q_{inc,r}}{\\partial \\varepsilon} = - \\varepsilon^{-2} \\left[ q_{net} + q_{conv} \\right]

            dq_dk: float (vector)
                Partial derivative of model with respect to insulation thermal conductivity
                .. math :: \\partial q_{inc,r}}{\\partial k_{ins}}=\\frac{1}{k_{ins} \\varepsilon} \\left[q_{st,ins} \\right]

            dq_drcpf: float (vector)
                Partial derivative of model with respect to volumetric heat capacity
                .. math :: \\frac{\\partial q_{inc,r}}{\\partial \\rho_s c_{p,sf}} = \\frac{1}{\\varepsilon \\rho_s c_{p,sf}} \\left[q_{st,f} \\right]

            dq_dl: float (vector)
                Partial derivative of model with respect to plate thickness
                .. math :: \\partial q_{inc,r}}{\\partial l_s}=\\frac{1}{\\varepsilon l_s} \\left[ q_{st,f} + q_{st,b} \\right]

            L_i: float (scalar)
                Insulation thickness

            dq_dL: float (scalar)
                Partial derivative of model with respect to insulation thickness
                .. math :: \\frac{\\partial q_{inc,r}}{\\partial L} \\approx \\frac{q_{net}(L) - q_{net}(L+\\delta L)}{\\delta L}
        """
        self.dq_deps = - self.front_plate.epsilon**(-2) * ((self.q_net + self.q_conv)*1000)
        self.dq_dk = 1/(self.insul.k*self.front_plate.epsilon) * (self.q_st_ins*1000)
        self.dq_drcpf = 1/(self.front_plate.epsilon * self.front_plate.rCp) * (self.q_st_f*1000)
        self.dq_drcpb = 1/(self.back_plate.epsilon * self.back_plate.rCp) * (self.q_st_b*1000)
        self.dq_dl = 1/(self.back_plate.epsilon*self.plate_thickness) * ((self.q_st_f + self.q_st_b)*1000)
        q_net = self.q_net*1000
        self.L_i = self.L_i + self.sigma_L
        q_net_dL = self.one_d_conduction()*1000

        self.dq_dL = (q_net - q_net_dL)/self.sigma_L
        

    def sensitivity_coefficients(self, hf, hb, out_directory=None):
        """
        Notes
        ----------
            Caclulates first order sensitivity coefficients for 1D conduction model
        """
        if self.model == 'one_d_conduction':
            getattr(self, 'one_d_uncertainty')()
            cond_keys = ['epsilon', 'k', 'rCp_f', 'rCp_b', 'l', 'L_i']
            self.W_eps = (self.dq_deps*self.sigma_eps/1000)**2
            self.W_k = (self.dq_dk*self.sigma_k/1000)**2
            self.W_rcpf = (self.dq_drcpf*self.sigma_rcpf/1000)**2
            self.W_rcpb = (self.dq_drcpb*self.sigma_rcpb/1000)**2
            self.W_l = (self.dq_dl*self.sigma_l/1000)**2
            self.W_L = (self.dq_dL*self.sigma_L/1000)**2

            self.W_cond = np.array([self.W_eps, self.W_k, self.W_rcpf, self.W_rcpb, self.W_l, self.W_L])

        if hasattr(hf, 'C') and hasattr(hf, 'n') and not hasattr(hf, 'm'):
            conv_keys = ['C', 'n']
            self.dq_dC = 1/(hf.C * self.front_plate.epsilon) * (self.q_conv*1000)
            self.dq_dn = hf.n/(self.front_plate.epsilon*hf.Ra) * (self.q_conv_f*1000) + hb.n/(self.back_plate.epsilon*hb.Ra) * (self.q_conv_b*1000)
            self.dq_dn[np.isnan(self.dq_dn)] = 0

            self.W_C = (self.dq_dC*hf.sigma_C/1000)**2
            self.W_n = (self.dq_dn*hf.sigma_n/1000)**2

            self.W_conv = np.array([self.W_C, self.W_n])

        self.keys = cond_keys + conv_keys
        self.W_model = np.row_stack((self.W_cond, self.W_conv))

        self.S = self.W_model/self.W_model.sum(axis=0)
        self.S_mean = pd.DataFrame(data=self.S.mean(axis=1)[None], columns=self.keys)
        self.sigma_model = np.sqrt(self.W_model.sum(axis=0))

        if not out_directory:
            print(self.S_mean)
        else:
            if os.path.isdir(out_directory) == True:
                pass
            else:
                os.mkdir(out_directory)
            self.S_mean.to_csv(out_directory+'sensitivity_coefficients.csv')

    def plot_sensitivity_coefficients(self, out_directory=None):
        """
        Notes
        ----------
            This method plots the sensitivity coefficients as a function of time
        """
        if not hasattr(self, 'S'):
            raise NameError('Run sensitivity_coefficients method prior to plotting')

        ax = plt.subplot(111)
        for s, s_coeff in enumerate(self.S):
            plt.plot(self.time, s_coeff, label=self.keys[s])
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
            plt.savefig(out_directory+'sensitivity_coefficients.pdf')
            plt.close()

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

    def plot_uncertainties(self, out_directory=None):
        """
        Notes
        ----------
            This method plots the incident heat flux and its uncertainty for a DFT (Uncertainty is estimated using a single term Taylor Series expansion)
        """
        if not hasattr(self, 'sigma_model'):
            raise NameError('Run sensitivity_coefficients method prior to plotting')

        y1 = self.q_inc - 1.96*self.sigma_model
        y2 = self.q_inc + 1.96*self.sigma_model
        
        ax = plt.subplot(111)
        plt.plot(self.time, self.q_inc, label='Incident Flux')
        plt.fill_between(x=self.time, y1=y1, y2=y2, color='grey')
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
            plt.savefig(out_directory+'flux_uncertainty.pdf')
            plt.close()

    def plot_uncertainties_mcmc(self, out_directory=None, samples=200):
        """
        Notes
        ----------
            This method draws samples of parameters and evaluates the model from these draws and plots the resultant curves
        """
        q_inc = self.q_inc
        mcmc_eps = self.front_plate.epsilon*stats.norm.rvs(1, 0.1**2, size=samples)
        mcmc_k = self.insul.k*stats.norm.rvs(1, 0.25**2, size=samples)
        mcmc_rcpf = self.front_plate.rCp.mean()*stats.norm.rvs(1, 0.05**2, size=samples)
        mcmc_rcpb = self.back_plate.rCp.mean()*stats.norm.rvs(1, 0.05**2, size=samples)
        mcmc_l = self.plate_thickness*stats.norm.rvs(1, 0.05, size=samples)
        mcmc_L = self.L_i*stats.norm.rvs(1, 0.05, size=samples)

        # plt.figure()
        ax = plt.subplot(111)
        for i in range(samples):
            self.front_plate.epsilon = mcmc_eps[i]
            self.insul.k = mcmc_k[i]
            self.front_plate.rCp = mcmc_rcpf[i]
            self.back_plate.rCp = mcmc_rcpb[i]
            self.plate_thickness = mcmc_l[i]
            self.L_i = mcmc_L[i]
            
            self.incident_heat_flux()
            plt.plot(self.time, self.q_inc, color='grey', linewidth=1)

        plt.plot(self.time, q_inc, label='Incident Flux')
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
            plt.savefig(out_directory+'flux_uncertainty.pdf')
            plt.close()

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

        self.T_0 = (T_f[0] + T_b[0])/2

        self.time = time
        self.h_f = h_f
        self.h_b = h_b
        self.model = model
        self.plate_thickness = plate_thickness
        self.plate_material = plate_material
        self.L_i = L_i
        self.insulation_material = insulation
        self.nodes = nodes

        if not np.any(T_inf):
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
        self.insul = self.insulation_material(self.T_0)

        if run_on_init:
            self.q_inc = one_dim_conduction.incident_heat_flux(self)

        ### Parameter Uncertainties
        self.sigma_eps = self.front_plate.epsilon*0.1 # Sandia Report
        self.sigma_k = self.insul.k*0.25 # Sandia Report
        self.sigma_rcpf = self.front_plate.rCp*0.05 # Sandia Report
        self.sigma_rcpb = self.back_plate.rCp*0.05 # Sandia Report
        self.sigma_l = self.plate_thickness*0.05 # Measurements
        self.sigma_L = self.L_i*0.05 # Measurements

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
        T_nodes_i = np.ones((time_steps, self.nodes))*self.T_0

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
