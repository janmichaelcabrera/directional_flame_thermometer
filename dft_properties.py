#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np

class stainless_steel:
    def __init__(self, T, epsilon=0.9, *args, **kwargs):
        self.T = T
        self.rCp = 5.535E-09*self.T**5 - 2.671E-05*self.T**4 + 4.978E-02*self.T**3 - 4.469E+01*self.T**2 + 2.041E+04*self.T + 5.125E+05
        self.epsilon = epsilon

    def update_rCp(self, x):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Rho times heat of steel (kg J/m^3-K)

        """
        self.rCp = 5.535E-09*x**5 - 2.671E-05*x**4 + 4.978E-02*x**3 - 4.469E+01*x**2 + 2.041E+04*x + 5.125E+05
        return self.rCp

    def update_epsilon(self, x):
        self.epsilon = x
        return self.epsilon


class ceramic_fiber:
    def __init__(self, T, *args, **kwargs):
        self.T = T
        self.rCp = -9.517E-08*self.T**4 + 3.309E-04*self.T**3 - 4.394E-01*self.T**2 + 3.211E+02*self.T + 8.151E+04
        self.k = 7.36E-17*self.T**5 - 3.02E-13*self.T**4 + 4.87E-10*self.T**3 - 2.35E-07*self.T**2 + 1.43E-04*self.T + 3.11E-03

    def update_rCp(self, x):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Rho times heat of steel (kg J/m^3-K)

        """
        self.rCp = -9.517E-08*x**4 + 3.309E-04*x**3 - 4.394E-01*x**2 + 3.211E+02*x + 8.151E+04
        return self.rCp

    def update_k(self, x):
        """
        Inputs
        ----------
            x: scalar, or vector
                temperature in K

        Returns
        ----------
            Rho times heat of steel (kg J/m^3-K)

        """
        self.k = 7.36E-17*x**5 - 3.02E-13*x**4 + 4.87E-10*x**3 - 2.35E-07*x**2 + 1.43E-04*x + 3.11E-03
        return self.k