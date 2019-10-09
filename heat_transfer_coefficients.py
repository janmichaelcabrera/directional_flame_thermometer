#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np

class air_props:
    def __init__(self, T, Kelvin=False):
        """
        Parameters
        ----------
            T: float, array like
                Temperature at which air properties will be evaluated at

            Kelvin: bool
                Whether or not the temperatures are in Celsius or Kelvin. If in Celsius, the data is converted to Kelvin

        """
        if not Kelvin:
            self.T = T + 273
        else:
            self.T = T
        self.k = air_props.thermal_conductivity(self, self.T)
        self.nu = air_props.kinematic_viscosity(self, self.T)
        self.alpha = air_props.thermal_diffusivity(self, self.T)
        self.beta = air_props.expansion_coefficient(self, self.T)
        self.Pr = self.nu/self.alpha

    def thermal_conductivity(self, Temp):
        """
        Inputs
        ----------
            T_f: scalar or vector
                Film temperature in Kelvin

        Returns
        ----------
            k_air: scalar or vector
                Thermal conductivity of air in W/m-K evaluated at the film temperature
        """
        #Interpolation temperatures in K
        interp_temp = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500] 
        #Thermal Conductivity of air at various temperatures [*10^-3 W/m-K]
        air_k = [18.1, 22.3, 26.3, 30.0, 33.8, 37.3, 40.7, 43.9, 46.9, 49.7, 52.4, 54.9, 57.3, 59.6, 62.0, 64.3, 66.7, 71.5, 76.3, 82, 91, 100]
        self.k = np.interp(Temp, interp_temp, air_k)*10.**-3.
        return self.k

    def kinematic_viscosity(self, Temp):
        """
        Inputs
        ----------
            T_f: scalar or vector
                Film temperature in Kelvin

        Returns
        ----------
            nu: scalar or vector
                Kinematic viscosity in m^2/s evaluated at the film temperature
        """
        #Interpolation temperatures in K
        interp_temp = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500] 
        #Viscosity of air at various temperatures [*10^-6 m^2/s]
        air_nu = [7.590, 11.44, 15.89, 20.92, 26.41, 32.39, 38.79, 45.57, 52.69, 60.21, 68.10, 76.37, 84.93, 93.80, 102.9, 112.2, 121.9, 141.8, 162.9, 185.1, 213, 240]
        
        self.nu = np.interp(Temp, interp_temp, air_nu)*10.**-6.
        return self.nu

    def thermal_diffusivity(self, Temp):
        """
        Inputs
        ----------
            T_f: scalar or vector
                Film temperature in Kelvin

        Returns
        ----------
            alpha_air: scalar or vector
                Thermal diffusivity of air in m^2/s evaluated at the film temperature
        """
        #Interpolation temperatures in K
        interp_temp = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500] 
        #Thermal Diffusivity of air at various temperatures [*10^-6 m2/s]
        air_alpha = [10.3, 15.9, 22.5, 29.9, 38.3, 47.2, 56.7, 66.7, 76.9, 87.3, 98.0, 109, 120, 131, 143, 155, 168, 195, 224, 238, 303, 350]
        self.alpha = np.interp(Temp, interp_temp, air_alpha)*10.**-6.
        return self.alpha

    def expansion_coefficient(self, Temp):
        self.beta = 1/self.T
        return self.beta

class natural_convection:
    def __init__(self, T, T_infty=None, L_ch=.0762, Kelvin=False):
        """
        Parameters
        ----------
            T: float, array like
                Temperature at which air properties will be evaluated at

            Kelvin: bool (optional)
                Whether or not the temperatures are in Celsius or Kelvin. If in Celsius, the data is converted to Kelvin

            T_infty: float or array like (optional)
                Temperature of the fluid 'far' from the DFT. If not provided, T_inf is assumed to be constant and equal to the temperature at the beginning of the array. 

            L_ch: float (optional)
                Characteristic length for determining heat transfer coefficients
        """
        if not Kelvin:
            self.T = T + 273
        else:
            self.T = T

        if not T_infty:
            self.T_infty = self.T[0]
        else:
            self.T_infty = T_infty

        self.L_ch = L_ch

    def horizontal(self):
        """
        Returns
        ----------
            h: float, array like
                Heat transfer coefficient in W/m^2-K
                .. math : Nu = 0.54 Ra^0.25
                .. math : h = \\frac{k_{air} Nu}{L_{ch}}
        """
        self.C = 0.54
        self.n = 0.25
        air = air_props(self.T)
        self.Ra = 9.81*air.beta*(self.T - self.T_infty)*self.L_ch**3/(air.nu*air.alpha)
        Nu = C*self.Ra**n
        self.h = Nu*air.k/self.L_ch
        self.h[np.isnan(self.h)] = 0
        return self.h

    def vertical(self):
        """
        Returns
        ----------
            h: float, array like
                Heat transfer coefficient in W/m^2-K
                .. math : Nu = 0.68 + (0.67 Ra^0.25)/(1 + (0.492/air.Pr)^{9/16})^{4/9}
                .. math : h = \\frac{k_{air} Nu}{L_{ch}}
        """
        air = air_props(self.T)
        Ra = 9.81*air.beta*(self.T - self.T_infty)*self.L_ch**3/(air.nu*air.alpha)
        Nu = 0.68 + (0.67*Ra**0.25)/(1 + (0.492/air.Pr)**(9/16))**(4/9)
        self.h = Nu*air.k/self.L_ch
        self.h[np.isnan(self.h)] = 0
        return self.h

    def custom(self, C, n):
        """
        Parameters
        ----------
            C: float
                Correlation constant

            n: float
                Correlation exponent
        Returns
        ----------
            h: float, array like
                Heat transfer coefficient in W/m^2-K
                .. math : Nu = C Ra^n
                .. math : h = \\frac{k_{air} Nu}{L_{ch}}
        """
        self.C, self.n = C, n
        air = air_props(self.T)
        self.Ra = 9.81*air.beta*(self.T - self.T_infty)*self.L_ch**3/(air.nu*air.alpha)
        # self.Ra[np.isnan(self.Ra)] = 0
        Nu = C*self.Ra**n
        self.h = Nu*air.k/self.L_ch
        self.h[np.isnan(self.h)] = 0
        return self.h

class forced_convection:
    def __init__(self, T, T_infty=None, L_ch=.0762, Kelvin=False):
        """
        Parameters
        ----------
            T: float, array like
                Temperature at which air properties will be evaluated at

            Kelvin: bool (optional)
                Whether or not the temperatures are in Celsius or Kelvin. If in Celsius, the data is converted to Kelvin

            T_infty: float or array like (optional)
                Temperature of the fluid 'far' from the DFT. If not provided, T_inf is assumed to be constant and equal to the temperature at the beginning of the array. 

            L_ch: float (optional)
                Characteristic length for determining heat transfer coefficients
        """
        if not Kelvin:
            self.T = T + 273
        else:
            self.T = T

        if not T_infty:
            self.T_infty = self.T[0]
        else:
            self.T_infty = T_infty

        self.L_ch = L_ch

    def custom(self, C, m, n, Re):
        """
        Parameters
        ----------
            C: float
                Correlation constant

            n: float
                Correlation exponent

            m: float
                Correlation exponent
        Returns
        ----------
            h: float, array like
                Heat transfer coefficient in W/m^2-K
                .. math : Nu = C Re^m Pr^n
                .. math : h = \\frac{k_{air} Nu}{L_{ch}}
        """
        air = air_props(self.T)
        Nu = C*Re**m*air.Pr**n
        self.h = Nu*air.k/self.L_ch
        self.h[np.isnan(self.h)] = 0
        return self.h
