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

C = 0.54
n = 0.25

h_f = dft.natural_convection(Tf)
h_f.custom(C, n)

h_b = dft.natural_convection(Tb)
h_b.custom(C, n)

sensor_1 = dft.one_dim_conduction(Tf, Tb, time, h_f.h, h_b.h)

# sensor_1.plot_components()
# sensor_1.save_output(out_directory='data/')

# sensor_1.plot_components()