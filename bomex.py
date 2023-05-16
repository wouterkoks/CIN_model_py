import matplotlib.pyplot as plt
import numpy as np
import copy
import os 
import analyse_Barts_LES
os.chdir('C:/Users/Wouter Koks/Desktop/MEP/CLASS_height_dep_arm/')
from model_zdep import *


""" 
Create empty model_input and set up BOMEX case
"""
bomex = model_input()

bomex.dt         = 60.        # time step [s]
bomex.runtime    = 12 * 3600.    # total run time [s]

# mixed-layer input
bomex.sw_ml      = True      # mixed-layer model switch
bomex.sw_shearwe = False     # shear growth mixed-layer switch
bomex.sw_fixft   = False     # Fix the free-troposphere switch
bomex.h          = 520.      # initial ABL height [m]
bomex.Ps         = 97000     # surface pressure [Pa]
bomex.divU       = 0.65 / (100 * 1500)   # horizontal large-scale divergence of wind [s-1] 

bomex.fc         = 1e-4      # Coriolis parameter [m s-1]

bomex.theta      = 298.7     # initial mixed-layer potential temperature [K]
bomex.dtheta     = 0.4       # initial temperature jump at h [K]
bomex.gammatheta = None      # free atmosphere potential temperature lapse rate [K m-1]
bomex.advtheta   = -2 / (24 * 3600)         # advection of heat [K s-1]
bomex.beta       = 0.15      # entrainment ratio for virtual heat [-]  #0.2????
bomex.wtheta     = None      # surface kinematic heat flux [K m s-1]
bomex.theta_ft0  = bomex.theta + bomex.dtheta - 3.85e-3 * bomex.h #  [K]

bomex.q          = 16.3e-3   # initial mixed-layer specific humidity [kg kg-1]  16.3
bomex.dq         = -0.3e-3   # initial specific humidity jump at h [kg kg-1]
bomex.gammaq     = None      # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
bomex.advq       = -1e-3 / (3600 * 24)        # advection of moisture [kg kg-1 s-1]
bomex.wq         = None      # surface kinematic moisture flux [kg kg-1 m s-1]
bomex.q_ft0      = bomex.q + bomex.dq + 5.833e-6 * bomex.h  # free tropospheric profile extrapolated to surface[kg kg-1]

bomex.CO2        = 422.      # initial mixed-layer CO2 [ppm]
bomex.dCO2       = -44.      # initial CO2 jump at h [ppm]
bomex.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
bomex.advCO2     = 0.        # advection of CO2 [ppm s-1]
bomex.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

bomex.sw_wind    = False     # prognostic wind switch
bomex.u          = -8.75        # initial mixed-layer u-wind speed [m s-1]
bomex.du         = 4.        # initial u-wind jump at h [m s-1]
bomex.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
bomex.advu       = 0.        # advection of u-wind [m s-2]

bomex.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
bomex.dv         = 4.0       # initial u-wind jump at h [m s-1]
bomex.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
bomex.advv       = 0.        # advection of v-wind [m s-2]

bomex.sw_sl      = False     # surface layer switch
bomex.ustar      = 0.3       # surface friction velocity [m s-1]
bomex.z0m        = 0.02      # roughness length for momentum [m]
bomex.z0h        = 0.002     # roughness length for scalars [m]

bomex.sw_rad     = False     # radiation switch
bomex.lat        = 51.97     # latitude [deg]
bomex.lon        = -4.93     # longitude [deg]
bomex.doy        = 268.      # day of the year [-]
bomex.tstart       = 0      # time of the day [h UTC]
bomex.cc         = 0.0       # cloud cover fraction [-]
bomex.Q          = 400.      # net radiation [W m-2] 
bomex.dFz        = 0.        # cloud top radiative divergence [W m-2] 

bomex.sw_ls      = False     # land surface switch
bomex.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
bomex.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
bomex.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
bomex.cveg       = 0.85      # vegetation fraction [-]
bomex.Tsoil      = 285.      # temperature top soil layer [K]
bomex.T2         = 286.      # temperature deeper soil layer [K]
bomex.a          = 0.219     # Clapp and Hornberger retention curve parameter a
bomex.b          = 4.90      # Clapp and Hornberger retention curve parameter b
bomex.p          = 4.        # Clapp and Hornberger retention curve parameter c
bomex.CGsat      = 3.56e-6   # saturated soil conductivity for heat

bomex.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
bomex.wfc        = 0.323     # volumetric water content field capacity [-]
bomex.wwilt      = 0.171     # volumetric water content wilting point [-]

bomex.C1sat      = 0.132     
bomex.C2ref      = 1.8

bomex.LAI        = 2.        # leaf area index [-]
bomex.gD         = 0.0       # correction factor transpiration for VPD [-]
bomex.rsmin      = 110.      # minimum resistance transpiration [s m-1]
bomex.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
bomex.alpha      = 0.25      # surface albedo [-]

bomex.Ts         = 290.      # initial surface temperature [K]

bomex.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
bomex.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

bomex.Lambda     = 5.9       # thermal diffusivity skin layer [-]

bomex.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

bomex.sw_cu      = False      # Cumulus parameterization switch
bomex.dz_h       = 150.      # Transition layer thickness [m]

# Time dependent surface variables; linearly interpolated by the model
# Note the time offset, as the mixed-layer model starts one hour later than LES!
# time   = bomex.tstart + np.array([0., 4, 6.5,  7.5,  10, 12.5, 14.5]) - 1

# daytime = 13
# time = np.linspace(0, 13)

# H      = np.array([-30.,  90., 140., 140., 100., -10.,  -10])
# LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
# rho    = bomex.Ps / (287. * bomex.theta * (1. + 0.61 * bomex.q))
# wtheta = H  / (rho*1005.)
# wq     = LE / (rho*2.5e6)
# wtheta = 0.12 * np.sin(np.pi * time / daytime)
# wq = .167e-3 * np.sin(np.pi * time / daytime)

time   = np.array([0., 12.]) 
wtheta = np.array([8e-3, 8e-3])
wq     = np.array([5.2e-5, 5.2e-5])
print(2.5e6 * wq / (1005 * wtheta + 2.5e6 * wq))
theta_ft0 = bomex.theta_ft0 - 2 / 24 * time
q_ft0 = bomex.q_ft0 - 1e-3 / 24 * time
time  *= 3600.
# thft0t = bomex.theta + bomex.dtheta - 3.85e-3 * bomex.h - 2/86400 * time 
bomex.timedep    = {'wtheta': (time, wtheta),
                  'wq':     (time, wq),
                  'theta_ft0': (time, theta_ft0),
                  'q_ft0': (time, q_ft0)}

# Binned height dependent lapse rates
z1         = np.array([0, 1480, 2000, 5500])
gammatheta = np.array([3.85e-3, 11.15e-3, 3.65e-3])

z2         = np.array([0, 1480, 2000, 5500])
gammaq     = np.array([-5.833e-6, -12.5e-6, -1.2e-6])

bomex.heightdep  = {'gammatheta': (z1, gammatheta),
                  'gammaq':     (z2, gammaq)}

bomex.sw_rhtend = True
bomex.sw_plume = False
bomex.save_ent_plume = False
bomex.zmax = 5000
bomex.n_pts  = 500
bomex.ent_corr_factor = 0.7
bomex.sw_cin   = False
bomex.phi_cu = 0.4
bomex.wcld_prefact = 0.84 
bomex.sw_store = False
bomex.hstore = 1.5e3


"""
Init and run the model
"""
# With cloud parameterisation
r1 = model(bomex)
r1.run()

# Without cloud parameterisation
bomex.sw_cu = True

r2 = model(bomex)
r2.run()

bomex.sw_plume = True
bomex.sw_cin = True
bomex.sw_store = True
r3 = model(bomex)
r3.run()


#%% Import LES data 
PLOT_LES = True
if PLOT_LES:
    les_data_loc = 'C:/Users/Wouter Koks/Desktop/MEP/LES_Bart' 
    
    data, z = analyse_Barts_LES.main(les_data_loc, bomex=True)
    
    # determine mixed-layer quantities based on h_ml=height at which wthv is minimized. 
    thml, qtml, h_ml = analyse_Barts_LES.make_fits(data, bomex=True)
    
    ind = np.where(data.wthv[:, 0] > 0)
    
    # determine cloud core massflux as in Van Stratum 2014. 
    wstar = np.zeros(data.time.size)
    wstar[ind] = (9.81 * h_ml[ind] * data.wthv[ind, 0] / data.thv[ind, 0]) ** (1. / 3.)       
    i_acc_max = np.argmax(data.acc*data.wcc, axis=1)
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
plt.plot(r1.out.t, r1.out.h, label='ARM Simple')
plt.plot(r2.out.t, r2.out.h, label='ARM Cu')
plt.plot(r3.out.t, r3.out.h, label='ARM Cu+CIN')
if PLOT_LES:
    plt.plot(data.time/3600, h_ml, label='LES')

plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('h [m]', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.theta, label='ARM Simple')
plt.plot(r2.out.t, r2.out.theta, label='ARM Cu')
# plt.plot(r3.out.t, r3.out.theta, label='ARM Cu+CIN+Sft')
if PLOT_LES:
    plt.plot(data.time/3600, thml, label='LES')
plt.xlabel('time [h]')
plt.ylabel(r'$\theta$ [K]', fontsize=fsize)
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.q*1000., label='ARM Simple')
plt.plot(r2.out.t, r2.out.q*1000., label='ARM Cu')

plt.plot(r3.out.t, r3.out.q*1000., label='ARM Cu+CIN+Sft')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('q [g kg-1]', fontsize=fsize)
if PLOT_LES:
    plt.plot(data.time/3600, qtml*1000, label='LES')
plt.legend()
plt.show() 

plt.figure()
# plt.plot(r1.out.t, bomex.dt * np.cumsum(r1.out.M), label=r'$$')
plt.plot(r2.out.t, bomex.dt * np.cumsum(r2.out.M), label=r'$Cu$')
# plt.plot(r3.out.t, bomex.dt * np.cumsum(r3.out.M), label=r'$Cu+CIN+Sft$')
if PLOT_LES:
    dt_les = data.time[1] - data.time[0]
    plt.plot(data.time/3600, dt_les  * np.cumsum(Mcc), label=r'$LES$')
plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$ \sum M \Delta t$ [m]', fontsize=fsize)
plt.legend()
plt.show()


plt.figure()
# plt.plot(r1.out.t, r1.out.wthetav, label='surface')
plt.plot(r1.out.t, r1.out.wthetav , label='entrainment')
if PLOT_LES:
    plt.plot(data.time/3600, data.wthv[:, 0])
plt.xlabel('time [h]')
plt.ylabel('wthv [K m s-1]')
plt.show()

plt.figure()
# plt.plot(r1.out.t, r1.out.wthetav, label='surface')
plt.plot(r1.out.t, r1.out.dthetav , label='entrainment')

plt.xlabel('time [h]')
plt.ylabel('wthv [K m s-1]')
plt.show()

plt.figure()
plt.plot(r1.out.t, r2.out.M)
plt.plot(r1.out.t, r3.out.M)
if PLOT_LES:
    dt_les = data.time[1] - data.time[0]
    plt.plot(data.time/3600, Mcc, label=r'$LES$')
plt.show()
#%%
q2_h = np.zeros_like(data.time)


qtcc_accmax = np.zeros_like(data.time)
qtcc_mean = np.zeros_like(data.time)
for it in range(len(data.time)):
    if not np.isnan(h_ml[it]):
        indh[it] = np.where(data.z > h_ml[it])[0][0]
        q2_h[it] = np.interp(h_ml[it], data.z, data.qt_2[it])  # find q2_h at h using interpolation
        imax_cc  = np.argmax(data.acc[it])   # height-index at which cloud core fraction is maximized
        qtcc_accmax[it] = data.qtcc[it, imax_cc]
        qtcc_mean[it] = np.nanmean(data.qtcc[it])

    q2_h[q2_h == 0] = np.nan
plt.figure()
plt.plot(r1.out.t, np.sqrt(r2.out.q2_h))
if PLOT_LES:
    plt.plot(data.time/3600, np.sqrt(q2_h))
plt.show()