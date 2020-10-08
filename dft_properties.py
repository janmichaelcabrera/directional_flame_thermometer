from __future__ import division
import numpy as np

class stainless_steel:
    def __init__(self, T, epsilon=0.94, *args, **kwargs):
        self.T = T
        self.rCp = stainless_steel.volumetric_heat_capacity(self, self.T)
        self.epsilon = epsilon

    def volumetric_heat_capacity(self, Temp):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Rho times heat of steel (kg J/m^3-K)

        """
        self.rCp = 5.535E-09*Temp**5 - 2.671E-05*Temp**4 + 4.978E-02*Temp**3 - 4.469E+01*Temp**2 + 2.041E+04*Temp + 5.125E+05
        return self.rCp

    def emissivity(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon

class ceramic_fiber:
    def __init__(self, T, *args, **kwargs):
        self.T = T
        self.rCp = ceramic_fiber.volumetric_heat_capacity(self, self.T)
        self.k = ceramic_fiber.thermal_conductivity(self, self.T)
        self.alpha = self.k/self.rCp

    def volumetric_heat_capacity(self, Temp):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Rho times heat of steel (kg J/m^3-K)

        """
        self.rCp = -9.517E-08*Temp**4 + 3.309E-04*Temp**3 - 4.394E-01*Temp**2 + 3.211E+02*Temp + 8.151E+04
        return self.rCp

    def thermal_conductivity(self, Temp):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Thermal conductivity in W/m-K

        """
        self.k = 7.36E-17*Temp**5 - 3.02E-13*Temp**4 + 4.87E-10*Temp**3 - 2.35E-07*Temp**2 + 1.43E-04*Temp + 3.11E-03
        return self.k