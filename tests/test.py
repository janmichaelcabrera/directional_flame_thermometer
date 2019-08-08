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

sensor_1 = dft.one_dim_conduction(Tf, Tb, time)

sensor_1.plot_results()