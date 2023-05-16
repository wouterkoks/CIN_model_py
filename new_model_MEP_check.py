# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:54:58 2023
Recreate my thesis's case with the new model to confirm that results have not changed
@author: Wouter Koks
"""

#
# Example of how to run the Python code, and access the output
# This case is identical to the default setup of CLASS (the version with interface) 
#

import matplotlib.pyplot as plt
import numpy as np
import copy
import os 
import analyse_Barts_LES
os.chdir('C:/Users/Wouter Koks/Desktop/MEP/CLASS_height_dep_arm/')
from model_zdep import *


thesiscase = model_input()

thesiscase.dt         = 5.        # time step [s]
thesiscase.runtime    = 8*3600    # total run time [s]

# mixed-layer input
thesiscase.sw_ml      = True      # mixed-layer model switch
thesiscase.sw_shearwe = False     # shear growth mixed-layer switch
thesiscase.sw_fixft   = False     # Fix the free-troposphere switch
thesiscase.h          = 790      # initial ABL height [m]
thesiscase.Ps         = 1e5     # surface pressure [Pa]
thesiscase.divU       = 0.        # horizontal large-scale divergence of wind [s-1]
thesiscase.fc         = 1e-4     # Coriolis parameter [m s-1]

thesiscase.theta      = 297.21612166726635     # initial mixed-layer potential temperature [K]
thesiscase.dtheta     = 0.5496       # initial temperature jump at h [K]
thesiscase.gammatheta = 3.5e-3      # free atmosphere potential temperature lapse rate [K m-1]
thesiscase.advtheta   = 0.        # advection of heat [K s-1]
thesiscase.beta       = 0.2      # entrainment ratio for virtual heat [-]  #0.2????
thesiscase.wtheta     = None      # surface kinematic heat flux [K m s-1]
thesiscase.theta_ft0  = thesiscase.theta + thesiscase.dtheta - thesiscase.gammatheta * thesiscase.h #  [K]

thesiscase.q          = 0.011601958563757722   # initial mixed-layer specific humidity [kg kg-1]
thesiscase.dq         = -0.0013676690740116286   # initial specific humidity jump at h [kg kg-1]
thesiscase.gammaq     = -3.5e-6      # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
thesiscase.advq       = 0        # advection of moisture [kg kg-1 s-1]
thesiscase.wq         = 0      # surface kinematic moisture flux [kg kg-1 m s-1]
thesiscase.q_ft0      = thesiscase.q + thesiscase.dq - thesiscase.gammaq * thesiscase.h  # free tropospheric profile extrapolated to surface[kg kg-1]

thesiscase.CO2        = 422.      # initial mixed-layer CO2 [ppm]
thesiscase.dCO2       = -44.      # initial CO2 jump at h [ppm]
thesiscase.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
thesiscase.advCO2     = 0.        # advection of CO2 [ppm s-1]
thesiscase.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

thesiscase.sw_wind    = False     # prognostic wind switch
thesiscase.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
thesiscase.du         = 4.        # initial u-wind jump at h [m s-1]
thesiscase.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
thesiscase.advu       = 0.        # advection of u-wind [m s-2]

thesiscase.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
thesiscase.dv         = 4.0       # initial u-wind jump at h [m s-1]
thesiscase.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
thesiscase.advv       = 0.        # advection of v-wind [m s-2]

thesiscase.sw_sl      = False     # surface layer switch
thesiscase.ustar      = 0.3       # surface friction velocity [m s-1]
thesiscase.z0m        = 0.02      # roughness length for momentum [m]
thesiscase.z0h        = 0.002     # roughness length for scalars [m]

thesiscase.sw_rad     = False     # radiation switch
thesiscase.lat        = 51.97     # latitude [deg]
thesiscase.lon        = -4.93     # longitude [deg]
thesiscase.doy        = 268.      # day of the year [-]
thesiscase.tstart     = 2         # time of the day [h UTC]
thesiscase.cc         = 0.0       # cloud cover fraction [-]
thesiscase.Q          = 400.      # net radiation [W m-2] 
thesiscase.dFz        = 0.        # cloud top radiative divergence [W m-2] 

thesiscase.sw_ls      = False     # land surface switch
thesiscase.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
thesiscase.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
thesiscase.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
thesiscase.cveg       = 0.85      # vegetation fraction [-]
thesiscase.Tsoil      = 285.      # temperature top soil layer [K]
thesiscase.T2         = 286.      # temperature deeper soil layer [K]
thesiscase.a          = 0.219     # Clapp and Hornberger retention curve parameter a
thesiscase.b          = 4.90      # Clapp and Hornberger retention curve parameter b
thesiscase.p          = 4.        # Clapp and Hornberger retention curve parameter c
thesiscase.CGsat      = 3.56e-6   # saturated soil conductivity for heat

thesiscase.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
thesiscase.wfc        = 0.323     # volumetric water content field capacity [-]
thesiscase.wwilt      = 0.171     # volumetric water content wilting point [-]

thesiscase.C1sat      = 0.132     
thesiscase.C2ref      = 1.8

thesiscase.LAI        = 2.        # leaf area index [-]
thesiscase.gD         = 0.0       # correction factor transpiration for VPD [-]
thesiscase.rsmin      = 110.      # minimum resistance transpiration [s m-1]
thesiscase.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
thesiscase.alpha      = 0.25      # surface albedo [-]

thesiscase.Ts         = 290.      # initial surface temperature [K]

thesiscase.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
thesiscase.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

thesiscase.Lambda     = 5.9       # thermal diffusivity skin layer [-]

thesiscase.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

thesiscase.sw_cu      = False      # Cumulus parameterization switch
thesiscase.dz_h       = 150.      # Transition layer thickness [m]

# Time dependent surface variables; linearly interpolated by the model
# Note the time offset, as the mixed-layer model starts one hour later than LES!
# time   = thesiscase.tstart + np.array([0., 4, 6.5,  7.5,  10, 12.5, 14.5]) - 1

# daytime = 13
# time = np.linspace(0, 13)

# H      = np.array([-30.,  90., 140., 140., 100., -10.,  -10])
# LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
# rho    = thesiscase.Ps / (287. * thesiscase.theta * (1. + 0.61 * thesiscase.q))
# wtheta = H  / (rho*1005.)
# wq     = LE / (rho*2.5e6)
# wtheta = 0.12 * np.sin(np.pi * time / daytime)
# wq = .167e-3 * np.sin(np.pi * time / daytime)

time   = np.linspace(0, 10) 
H      = 424.26 * np.sin(np.pi * time / 12)
# LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
rho    = thesiscase.Ps / (287. * thesiscase.theta * (1. + 0.61 * thesiscase.q))
wtheta = H  / (rho*1005.)
# wq     = LE / (rho*2.5e6)


time  *= 3600.

thesiscase.timedep    = {'wtheta': (time, wtheta)}

# Binned height dependent lapse rates
# z1         = np.array([0, 700, 5000])
# gammatheta = np.array([3.4e-3, 5.7e-3])

# z2         = np.array([0, 650, 1300, 5000])
# gammaq     = np.array([-0.6e-6, -2e-6, -8.75e-6])

# thesiscase.heightdep  = {'gammatheta': (z1, gammatheta),
#                   'gammaq':     (z2, gammaq)}

thesiscase.heightdep = {}

thesiscase.sw_rhtend = True
thesiscase.sw_plume = False
thesiscase.save_ent_plume = True
thesiscase.zmax = 4990  
thesiscase.n_pts  = 499
thesiscase.ent_corr_factor = 0.7
thesiscase.sw_cin   = False
thesiscase.phi_cu = 0.51  #!!!
thesiscase.wcld_prefact = 0.84  #!!! 
thesiscase.sw_store = False
thesiscase.hstore = 3.5e3

"""
Init and run the model
"""
# With cloud parameterisation
r1 = model(thesiscase)
r1.run()

# Without cloud parameterisation
thesiscase.sw_cu = True

r2 = model(thesiscase)
r2.run()

thesiscase.sw_plume = True
thesiscase.sw_cin = True
thesiscase.sw_store = True
r3 = model(thesiscase)
r3.run()


#%% Import LES data 
PLOT_LES = False
if PLOT_LES:
    les_data_loc = 'C:/Users/Wouter Koks/Desktop/MEP/LES_Bart' 
    
    data, z = analyse_Barts_LES.main(les_data_loc)
    
    # determine mixed-layer quantities based on h_ml=height at which wthv is minimized. 
    thml, qtml, h_ml = analyse_Barts_LES.make_fits(data)
    
    ind = np.where(data.wthv[:, 0] > 0)
    
    # determine cloud core massflux as in Van Stratum 2014. 
    wstar = np.zeros(data.time.size)
    wstar[ind] = (9.81 * h_ml[ind] * data.wthv[ind, 0] / data.thv[ind, 0]) ** (1. / 3.)       
    i_acc_max = np.argmax(data.acc, axis=1)
    i_arr = np.arange(data.time.size)
    acc_max = np.max(data.acc, axis=1)
    Mcc = data.acc[i_arr, i_acc_max] * data.wcc[i_arr, i_acc_max]

else:
    data = None
#%%
"""
Plot output
"""
plt.close('all')

fsize = 14
plt.figure()
plt.plot(r1.out.t, r1.out.h, label='thesiscase Simple')
plt.plot(r2.out.t, r2.out.h, label='thesiscase Cu')
plt.plot(r3.out.t, r3.out.h, label='thesiscase Cu+CIN')
if PLOT_LES:
    t_les = data.time/3600 
    plt.plot(t_les, h_ml, label='LES')

plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('h [m]', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.theta, label='thesiscase Simple')
plt.plot(r2.out.t, r2.out.theta, label='thesiscase Cu')
plt.plot(r3.out.t, r3.out.theta, label='thesiscase Cu+CIN')
if PLOT_LES:
    plt.plot(t_les, thml, label='LES')
plt.xlabel('time [h]')
plt.ylabel(r'$\theta$ [K]', fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.q*1000., label='thesiscase Simple')
plt.plot(r2.out.t, r2.out.q*1000., label='thesiscase Cu')
plt.plot(r3.out.t, r3.out.q*1000., label='thesiscase Cu+CIN')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('q [g kg-1]', fontsize=fsize)
if PLOT_LES:
    plt.plot(t_les, qtml*1000, label='LES')
plt.legend()
plt.show() 

plt.figure()
# plt.plot(r1.out.t, thesiscase.dt * np.cumsum(r1.out.M), label=r'$$')
plt.plot(r2.out.t, thesiscase.dt * np.cumsum(r2.out.M), label=r'$Cu$')
plt.plot(r3.out.t, thesiscase.dt * np.cumsum(r3.out.M), label=r'$Cu+Sft+CIN$')
if PLOT_LES:
    dt_les = data.time[1] - data.time[0]
    plt.plot(t_les, dt_les  * np.cumsum(Mcc), label=r'$LES$')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$ \sum M \Delta t$ [m]', fontsize=fsize)
plt.legend()
plt.show()

plt.figure()
# plt.plot(r1.out.t, thesiscase.dt * np.cumsum(r1.out.M), label=r'$$')
plt.plot(r2.out.t, r2.out.M, label=r'$Cu$')
plt.plot(r3.out.t, r3.out.M, label=r'$Cu+Sft+CIN$')
if PLOT_LES:
    plt.plot(t_les, Mcc, label=r'$LES$')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$M$ [m s-1]', fontsize=fsize)
plt.legend()
plt.show()

plt.figure()
# plt.plot(r1.out.t, r1.out.wthetav, label='surface')
plt.plot(r1.out.t, r1.out.wthetav , label='entrainment')
if PLOT_LES:
    plt.plot(t_les, data.wthv[:, 0])
plt.xlabel('time [h]')
plt.ylabel('wthv [K m s-1]')
plt.show()
#%%
# print(((9.8 * r3.out.h * r3.out.wthetav) / r3.out.thetav))

plt.figure()

# plt.plot(r1.out.t, r3.out.ac*((9.8 * r3.out.h * r3.out.wthetav) / r3.out.thetav)**(1./3.))
plt.plot(r1.out.t, thesiscase.dt * np.cumsum(r3.out.M))
plt.plot(r1.out.t, thesiscase.dt * np.cumsum(r2.out.M))
plt.show()

plt.figure()
plt.plot(r1.out.t, r3.out.cin)
plt.show()

#%%
plt.figure()
plt.plot(r3.out.t, r3.out.Stheta/thesiscase.hstore)
plt.show()