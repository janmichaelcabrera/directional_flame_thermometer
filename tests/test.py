import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
import directional_flame_thermometer as dft
import pandas as pd

# Read in test data
df = pd.read_csv('data/1905-01_10.csv')

# Pass data to variables
Tf = df.tc_1.values
Tb = df.tc_2.values
time = df.time.values

# Instantiate heat transfer coefficient objects for front and back plates
h_f = dft.natural_convection(Tf)
h_b = dft.natural_convection(Tb)

# Set heat transfer coefficient model variables
C = 0.65
n = 0.25

# Pass model variables to heat transfer coefficient objects
h_f.custom(C, n)
h_b.custom(C, n)

# Pass variables and run model
sensor_1 = dft.one_dim_conduction(Tf, Tb, time, h_f.h, h_b.h, model='one_d_conduction')

# Plot results
plt.figure()
plt.plot(time, sensor_1.q_inc, label='$q_{inc}$')
plt.plot(time, sensor_1.q_net, label='$q_{net}$')
plt.plot(time, sensor_1.q_conv, label='$q_{conv}$')
plt.plot(time, sensor_1.q_rad, label='$q_{rad}$')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Heat Flux (kW/m$^2$)', fontsize=14)
plt.legend(loc=0)
plt.show()