#
# Example of how to run the Python code, and access the output
# This case is identical to the default setup of CLASS (the version with interface) 
#

import matplotlib.pyplot as plt
import numpy as np
import copy
import os 
os.chdir('C:/Users/Wouter Koks/Desktop/MEP/CLASS_height_dep_arm/')
import analyse_Barts_LES
from model_zdep import *


""" 
Create empty model_input and set up ARM-SGP case
"""
arm = model_input()

arm.dt         = 10.        # time step [s]
arm.runtime    = 12*3600    # total run time [s]

# mixed-layer input
arm.sw_ml      = True      # mixed-layer model switch
arm.sw_shearwe = False     # shear growth mixed-layer switch
arm.sw_fixft   = False     # Fix the free-troposphere switch
arm.h          = 140.      # initial ABL height [m]
arm.Ps         = 97000     # surface pressure [Pa]
arm.divU       = 0.        # horizontal large-scale divergence of wind [s-1]
arm.fc         = 1e-4     # Coriolis parameter [m s-1]

arm.theta      = 301.4     # initial mixed-layer potential temperature [K]
arm.dtheta     = 0.4       # initial temperature jump at h [K]
arm.gammatheta = None      # free atmosphere potential temperature lapse rate [K m-1]
arm.advtheta   = 0.        # advection of heat [K s-1]
arm.beta       = 0.15      # entrainment ratio for virtual heat [-]  #0.2????
arm.wtheta     = None      # surface kinematic heat flux [K m s-1]
arm.theta_ft0  = arm.theta + arm.dtheta - 3.4e-3 * 140. #  [K]

arm.q          = 15.3e-3   # initial mixed-layer specific humidity [kg kg-1]
arm.dq         = -0.2e-3   # initial specific humidity jump at h [kg kg-1]
arm.gammaq     = None      # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
arm.advq       = 0        # advection of moisture [kg kg-1 s-1]
arm.wq         = None      # surface kinematic moisture flux [kg kg-1 m s-1]
arm.q_ft0      = arm.q + arm.dq + 0.6e-6 * 140  # free tropospheric profile extrapolated to surface[kg kg-1]

arm.CO2        = 422.      # initial mixed-layer CO2 [ppm]
arm.dCO2       = -44.      # initial CO2 jump at h [ppm]
arm.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
arm.advCO2     = 0.        # advection of CO2 [ppm s-1]
arm.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

arm.sw_wind    = False     # prognostic wind switch
arm.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
arm.du         = 4.        # initial u-wind jump at h [m s-1]
arm.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
arm.advu       = 0.        # advection of u-wind [m s-2]

arm.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
arm.dv         = 4.0       # initial u-wind jump at h [m s-1]
arm.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
arm.advv       = 0.        # advection of v-wind [m s-2]

arm.sw_sl      = False     # surface layer switch
arm.ustar      = 0.3       # surface friction velocity [m s-1]
arm.z0m        = 0.02      # roughness length for momentum [m]
arm.z0h        = 0.002     # roughness length for scalars [m]

arm.sw_rad     = False     # radiation switch
arm.lat        = 51.97     # latitude [deg]
arm.lon        = -4.93     # longitude [deg]
arm.doy        = 268.      # day of the year [-]
arm.tstart     = 12.5         # time of the day [h UTC]
arm.cc         = 0.0       # cloud cover fraction [-]
arm.Q          = 400.      # net radiation [W m-2] 
arm.dFz        = 0.        # cloud top radiative divergence [W m-2] 

arm.sw_ls      = False     # land surface switch
arm.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
arm.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
arm.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
arm.cveg       = 0.85      # vegetation fraction [-]
arm.Tsoil      = 285.      # temperature top soil layer [K]
arm.T2         = 286.      # temperature deeper soil layer [K]
arm.a          = 0.219     # Clapp and Hornberger retention curve parameter a
arm.b          = 4.90      # Clapp and Hornberger retention curve parameter b
arm.p          = 4.        # Clapp and Hornberger retention curve parameter c
arm.CGsat      = 3.56e-6   # saturated soil conductivity for heat

arm.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
arm.wfc        = 0.323     # volumetric water content field capacity [-]
arm.wwilt      = 0.171     # volumetric water content wilting point [-]

arm.C1sat      = 0.132     
arm.C2ref      = 1.8

arm.LAI        = 2.        # leaf area index [-]
arm.gD         = 0.0       # correction factor transpiration for VPD [-]
arm.rsmin      = 110.      # minimum resistance transpiration [s m-1]
arm.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
arm.alpha      = 0.25      # surface albedo [-]

arm.Ts         = 290.      # initial surface temperature [K]

arm.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
arm.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

arm.Lambda     = 5.9       # thermal diffusivity skin layer [-]

arm.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

arm.sw_cu      = False      # Cumulus parameterization switch
arm.dz_h       = 150.      # Transition layer thickness [m]

# Time dependent surface variables; linearly interpolated by the model
# Note the time offset, as the mixed-layer model starts one hour later than LES!
# time   = arm.tstart + np.array([0., 4, 6.5,  7.5,  10, 12.5, 14.5]) - 1

# daytime = 13
# time = np.linspace(0, 13)

# H      = np.array([-30.,  90., 140., 140., 100., -10.,  -10])
# LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
# rho    = arm.Ps / (287. * arm.theta * (1. + 0.61 * arm.q))
# wtheta = H  / (rho*1005.)
# wq     = LE / (rho*2.5e6)
# wtheta = 0.12 * np.sin(np.pi * time / daytime)
# wq = .167e-3 * np.sin(np.pi * time / daytime)

time   = np.array([0., 4, 6.5,  7.5,  10, 12.5, 14.5]) - 1
H      = np.array([-30.,  90., 140., 140., 100., -10.,  -10])
LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
rho    = arm.Ps / (287. * arm.theta * (1. + 0.61 * arm.q))
wtheta = H  / (rho*1005.)
wq     = LE / (rho*2.5e6)
print(wq)
print(wtheta)

time  *= 3600.

arm.timedep    = {'wtheta': (time, wtheta),
                  'wq':     (time, wq)}

# Binned height dependent lapse rates
z1         = np.array([0, 700, 5000])
gammatheta = np.array([3.4e-3, 5.7e-3])

z2         = np.array([0, 650, 1300, 5000])
gammaq     = np.array([-0.6e-6, -2e-6, -8.75e-6])

arm.heightdep  = {'gammatheta': (z1, gammatheta),
                  'gammaq':     (z2, gammaq)}

arm.sw_rhtend = True
arm.sw_plume = False
arm.save_ent_plume = True
arm.zmax = 4990  
arm.n_pts  = 499
arm.ent_corr_factor = 0.7
arm.sw_cin   = False
arm.phi_cu = 0.51 #!!!
arm.wcld_prefact = 0.84  #!!! 
arm.sw_store = False
arm.hstore = 3.5e3

"""
Init and run the model
"""

# With cloud parameterisation
r1 = model(arm)
r1.run()

# Without cloud parameterisation
arm.sw_cu = True
arm.sw_plume = True
r2 = model(arm)
r2.run()

#%%
arm.sw_plume = True
arm.sw_cin = True
arm.sw_store = True
r3 = model(arm)
r3.run()


#%% Import LES data 
PLOT_LES = True
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
    data.wqM[data.wqM > 100] = np.nan
    wqM_accmax = data.wqM[i_arr, i_acc_max]



else:
    data = None
#%%
t_les = data.time/3600 + 11.5
"""
Plot output
"""
plt.close('all')

fsize = 14
plt.figure()
plt.plot(r1.out.t, r1.out.h, label='ARM Simple')
plt.plot(r2.out.t, r2.out.h, label='ARM Cu')
plt.plot(r3.out.t, r3.out.h, label='ARM Cu+CIN')
if PLOT_LES:
    plt.plot(t_les, h_ml, label='LES')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('h [m]', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.theta, label='ARM Simple')
plt.plot(r2.out.t, r2.out.theta, label='ARM Cu')
plt.plot(r3.out.t, r3.out.theta, label='ARM Cu+CIN')
if PLOT_LES:
    plt.plot(t_les, thml, label='LES')
plt.xlabel('time [h]')
plt.ylabel(r'$\theta$ [K]', fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.q*1000., label='ARM Simple')
plt.plot(r2.out.t, r2.out.q*1000., label='ARM Cu')
plt.plot(r3.out.t, r3.out.q*1000., label='ARM Cu+CIN')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('q [g kg-1]', fontsize=fsize)
if PLOT_LES:
    plt.plot(t_les, qtml*1000, label='LES')
plt.legend()
plt.show() 

plt.figure()
# plt.plot(r1.out.t, arm.dt * np.cumsum(r1.out.M), label=r'$$')
plt.plot(r2.out.t, arm.dt * np.cumsum(r2.out.M), label=r'$Cu$')
plt.plot(r3.out.t, arm.dt * np.cumsum(r3.out.M), label=r'$Cu+Sft+CIN$')
if PLOT_LES:
    dt_les = data.time[1] - data.time[0]
    plt.plot(t_les, dt_les  * np.cumsum(Mcc), label=r'$LES$')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$ \sum M \Delta t$ [m]', fontsize=fsize)
plt.legend()
plt.show()

plt.figure()
# plt.plot(r1.out.t, arm.dt * np.cumsum(r1.out.M), label=r'$$')
plt.plot(r2.out.t, r2.out.M, label=r'Cu')
# plt.plot(r3.out.t, r3.out.M, label=r'$Cu+Sft+CIN$')
if PLOT_LES:
    plt.plot(t_les, Mcc, label=r'LES')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$M$ [m s-1]', fontsize=fsize)
plt.legend()
plt.show()

plt.figure()
# plt.plot(r1.out.t, r1.out.wthetav, label='surface')
plt.plot(r1.out.t, r1.out.wthetav)
if PLOT_LES:
    plt.plot(t_les, data.wthv[:, 0])
plt.xlabel('time [h]')
plt.ylabel('wthv [K m s-1]')
plt.show()
#%%
q2_h = np.zeros_like(data.time)

indh = np.zeros(len(data.time), dtype=int)
# qtcc_accmax = np.zeros_like(data.time)
# qtcc_mean = np.zeros_like(data.time)
for it in range(len(data.time)):
    if not np.isnan(h_ml[it]):
        indh[it] = np.where(data.z > h_ml[it])[0][0]
        q2_h[it] = np.interp(h_ml[it], data.z, data.qt_2[it])  # find q2_h at h using interpolation
q2_h[q2_h > 1] = np.nan
plt.figure()
plt.plot(r2.out.t, np.sqrt(r2.out.q2_h))
if PLOT_LES:
    plt.plot(t_les, np.sqrt(q2_h))  # it was in units gkg-1ms-1 (wrongly noted in NetCDF file)
plt.show()
#%%
plt.figure()

plt.plot(r1.out.t, r3.out.M)
plt.plot(r1.out.t, r2.out.M)
plt.show()

plt.figure()
plt.plot(r1.out.t, r3.out.w_lfc)
plt.plot(r1.out.t, ((9.81 * r3.out.h * r3.out.wthetav) / r3.out.thetav)**(1./3.))
plt.show()

