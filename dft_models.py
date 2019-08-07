#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
from .dft_properties import *
from .heat_transfer_coefficients import *

# Stefan-Boltzman constant (W/m2K4)
sigma = 5.6704E-8

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
    def __init__(self, T_f, T_r, time, Kelvin=False, L_i=0.0019, plate_material=stainless_steel, h=natural_convection, run_on_init=True, *args, **kwargs):
        self.Kelvin = Kelvin
        if not Kelvin:
            self.T_f = T_f + 273
            self.T_r = T_r + 273
        else:
            self.T_f = T_f
            self.T_r = T_r

        self.time = time
        self.plate = plate_material
        self.L_i = L_i
        self.h = h

        if run_on_init:
            self.q_inc = one_dim_conduction.run_model(self)

    def run_model(self):
        one_dim_conduction.net_heat_flux(self)
        return 0

    def net_heat_flux(self):
        
        def construct_banded_matrix(Fo=5, nodes=10, inv=True):
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

            if inv==True:
                return linalg.inv(A)
            else:
                return A

        T_inf = self.T_f[0]

        ### Steel
        # Volumetric heat capacity (J/m^3-K)
        
        front_plate = self.plate(self.T_f)
        back_plate = self.plate(self.T_f)

        # rcp_sf = getattr(self.plate, 'volumetric_heat_capacity')(self.T_f)

        ### Insulation

        insul = ceramic_fiber(T_inf)

        # # Thermal conductivity (W/m-k)
        # # k_i = k_c(T_inf, params)
        # if params == [None]:
        #     k_i = k_c(T_inf)
        # else:
        #     k_i = np.exp(params)
        #     # k_i = params

        # Volumetric heat capacity (J/m^3-K)
        

        # # Instantiate implicit model parameters
        # delta_x = L_i/(nodes+1)
        # time_steps = len(my_times)
        # T_nodes_i = np.ones((time_steps, nodes))*T_inf

        # # Insulation diffusivity (m^2/s)   
        # alpha_i = k_i/rcp_i
        # # Insulation Fourrier number
        # Fo_i = delta_t*alpha_i/delta_x**2
        # # Insulation implicit matrix construction and inversion
        # A_i_inv = construct_banded_matrix(Fo=Fo_i, nodes=nodes, inv=True)

        # # Loop over time
        # for t in range(time_steps-1):
        #     # Create temporary vector for current time step
        #     Temp_i = T_nodes_i[t].copy()
        #     # Update boundary condition at x=0
        #     Temp_i[0] = Temp_i[0] + Fo_i*Tf[t+1]
        #     # Update boundary condition at x=L
        #     Temp_i[-1] = Temp_i[-1] + Fo_i*Tr[t+1]
        #     # Solve for temperatures at next time step
        #     T_nodes_i[t+1] = A_i_inv @ Temp_i

        # # q_{cond} = -k_i A_{i} \frac{dT_i}{dx}|_{x=0} (Watts)
        # q_cond_if = -k_i*(T_nodes_i[:,0] - Tf)/delta_x
        # q_cond_ir = -k_i*(T_nodes_i[:,-1] - Tr)/delta_x

        # # q_{st,k} = V_p \rho c_p \frac{dT_{k}}{dt} \frac{1}{\epsilon} (Watts)
        # q_st_f = rcp_s*np.gradient(Tf, my_times)
        # q_st_r = rcp_s*np.gradient(Tr, my_times)
        # q_st_ins = (q_cond_if - q_cond_ir)

        # self.q_net = q_st_f + q_st_ins + q_st_r 

        # return self.q_net
