#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
sys.path.append('../../')
import differential_flame_thermometer as dft
import pandas as pd

df = pd.read_csv('data/1905-01_10.csv')
Tf = df.tc_1.values
Tb = df.tc_2.values
time = df.time.values

h_f = dft.natural_convection(Tf)
# h_f.custom(0.54, 0.25)
h_f.horizontal()

h_b = dft.natural_convection(Tb)
# h_b.custom(0.54, 0.25)
h_b.horizontal()

sensor_1 = dft.one_dim_conduction(Tf, Tb, time, h_f.h, h_b.h)

# Pass the ceramic fiber and steel similarly to how it was done for the heat transfer coefficient

# This should allow you to change the attributes within a method for later uncertainty quantification and sensitivity analysis