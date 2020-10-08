from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append('../../')
import directional_flame_thermometer as dft
import pandas as pd

df = pd.read_csv('data/1905-01_10.csv')
Tf = df.tc_1.values
Tb = df.tc_2.values
time = df.time.values

C = 0.65
n = 0.25

h_f = dft.natural_convection(Tf)
h_f.custom(C, n)

h_b = dft.natural_convection(Tb)
h_b.custom(C, n)

sensor_1 = dft.one_dim_conduction(Tf, Tb, time, h_f.h, h_b.h, model='one_d_conduction')

plt.figure()
plt.plot(time, sensor_1.q_inc)
plt.show()