# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:36:03 2021

@author: Wouter Koks
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os



def calc_sat(temp, pref):
    esat = 610.78 * np.exp(17.2694 * (temp - 273.16)/(temp - 35.86))
    qsat = rdorv * esat / (pref + (1 - rdorv) * esat)
    # qsat = 0.622 * esat / pref
    return qsat

class vars_save:
    def __init__(self, shape_prof):
        self.thl = np.zeros(shape_prof)  #t, z
        self.qt = np.zeros(shape_prof)
        self.ql = np.zeros(shape_prof)
        self.thv = np.zeros(shape_prof)
        self.qt2r = np.zeros(shape_prof)
        self.qt = np.zeros(shape_prof)
        self.zi = np.zeros(shape_prof)  
    
    
def tmean(var, t, d2t):
    if len(var) < (t + d2t):
        print('ERROR')
    var_r = var[t - d2t:t + d2t + 1]
    return np.mean(var_r, axis=0)


def initial_profs(data):
    it = 350
    print(data.time[it]/3600)
    dz = data.z[1] - data.z[0]


    plt.figure()
    plt.plot(data.thl[0, :], data.z)
    plt.plot(data.thl[it, :], data.z)

    plt.title('Initial profile')
    plt.ylabel('z (m)')
    plt.xlabel(r'$\theta_l$ (K)')
    plt.show()
    plt.figure()
    plt.plot(data.qt[0, :], data.z)
    plt.plot(data.qt[it, :], data.z)
    plt.show()
    return 

    

def make_fits(data, bomex=False):
    '''Calculate h_ml using fit or DALES' output, calculate thml and qvml using fit. '''
    
    thml = np.zeros_like(data.time)
    qtml = np.zeros_like(data.time)
    h_ml = np.zeros_like(data.time)
    if bomex: 
        hmax = 1e3
    else:
        hmax = 2e3
    imax = np.searchsorted(data.z, hmax)
    
    for it in range(data.time.size):
        # h_ml = data.zi[it]
        wthv = data.wthv[it, :imax]
        ih = np.argmin(wthv)
        h_ml[it] = data.z[ih]

        thml[it] = np.mean(data.thl[it, :ih])
        qtml[it] = np.mean(data.qt[it, :ih])
        # wthv = data.wthv[it]
        # plt.plot(wthv[1:], z)
        # plt.hlines(h_ml[it], np.min(wthv), np.max(wthv))
        # plt.show()
        
    return thml, qtml, h_ml

        

def main(upper_dir, bomex=False, info=True):
    print(bomex)
    if bomex:
        default_dir = upper_dir + '/bomex.default.0000000.nc'
        core_dir = upper_dir + '/bomex.qlcore.0000000.nc'
    else:
        default_dir = upper_dir + '/arm.default.0000000.nc'
        core_dir = upper_dir + '/arm.qlcore.0000000.nc'
    tstart = 0
    prof_test = nc.Dataset(default_dir)
    z = np.array(prof_test['z'])
    t = np.array(prof_test['time'])
    t_range = np.where(t >= tstart)
        
    shape_prof = (len(t[t_range]), len(z))
    data = vars_save(shape_prof)
    prof = nc.Dataset(default_dir)
    core = nc.Dataset(core_dir)
    
    data.z = z
    data.time = t[t_range]
    data.thl = np.array(prof['thermo/thl'])[t_range]
    data.qt = np.array(prof['thermo/qt'])[t_range]
    data.ql = np.array(prof['thermo/ql'])[t_range]
    data.thv = np.array(prof['thermo/thv'])[t_range]
    data.wthv = np.array(prof['thermo/thv_flux'])[t_range]
    data.qt_2 = np.array(prof['thermo/qt_2'])[t_range]
    # data.qt2r = np.array(prof['thermo/qt2r'])[t_range]
    data.zi = np.array(prof['thermo/zi'])[t_range]
    data.acc = np.array(core['default/areah'])[t_range]
    data.wcc = np.array(core['default/w'])[t_range]
    data.qtcc = np.array(core['thermo/qt'])[t_range]
    data.wqM = np.array(core['thermo/qt_flux'])[t_range]
    
    
    return data, z
    

Rd= 287
Rv = 461
rdorv = Rd / Rv
cp = 1.004e+3
Lv = 2.5e+6
tv_const = 0.608

class sv():
    def __init__(self):
        pass

if __name__ == "__main__":
    upper_dir = 'C:/Users/Wouter Koks/Desktop/MEP/LES_Bart' 
    bomex = False
    data, z = main(upper_dir, bomex)


    idt = 10
    t_eval = len(data.time) -idt - 1 
    print('t= '+str(data.time[t_eval]/3600) + ' hr')
    sim = 0

    initial_profs(data)
    
    it = 500

    thml, qtml, h_ml = make_fits(data)
    
    
    
    #%%
    t_les = data.time/3600 + 11.5
    print(data.qtcc)
    fsize = 14
    data.qtcc[data.qtcc > 1] = np.nan
    data.qt_2[data.qt_2 > 1] = np.nan
    data.qt_2[data.qt_2 == 0] = np.nan
    if bomex:
        h_ml[h_ml > 700] = np.nan   # remove bad data


    plt.figure()
    plt.plot(data.time, h_ml)
    plt.show()
    
    i_acc_max = np.argmax(data.acc, axis=1)
    plt.figure()
    plt.plot(data.acc[500, 1:], data.z)
    plt.hlines(data.z[i_acc_max[500]], 0, np.max(data.acc[500]), colors='k', linestyle='--')
    plt.xlabel(r"$a_\mathrm{cc}$", fontsize=fsize)
    plt.ylabel(r"$z$", fontsize=fsize)
    plt.xlim([0, 0.06])
    plt.show()
    
    plt.figure()
    plt.plot(data.qtcc[500], data.z)
    plt.hlines(data.z[i_acc_max[500]], 0.014, 0.0175, colors='k', linestyle='--')
    plt.xlabel(r"$q_{t,cc}$", fontsize=fsize)
    plt.ylabel(r'z', fontsize=fsize)
    plt.show()
    
    indh = np.zeros(len(data.time), dtype=int)
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
    plt.plot(data.time/3600+11.5, qtml, label=r'$q_\mathrm{t,ml}$')
    plt.plot(data.time/3600+11.5, qtcc_accmax, label=r'$q_\mathrm{t,cc}$')
    plt.xlim([12.5, 12.5+12])
    plt.ylim([0.014, 0.017])

    plt.legend(fontsize=12)
    plt.show()
    plt.figure()
    plt.plot(data.time, np.sqrt(q2_h))
    plt.show()
    var = (qtcc_accmax - qtml) / np.sqrt(q2_h)

    # var[var > 10] = np.nan

    plt.figure()
    plt.plot(data.time/3600+11.5, var)
    plt.ylabel('$\phi$')
    plt.title('BOMEX')
    plt.ylim([0,1])
    plt.show()
    #%%
    wstar = (9.81 * h_ml * data.wthv[:, 0] / data.thv[:, 0]) ** (1. / 3.)      
    i_arr = np.arange(len(data.time))

    wcc = data.wcc[i_arr, i_acc_max]
    wcc[wcc > 10] = np.nan
    # wcc[wcc > 10] = np.nan
    plt.figure()
    # plt.plot(t_les, wcc/wstar)
    plt.plot(t_les, wstar, label=r'$w_\star$')
    plt.plot(t_les, wcc, label=r'$w_{cc}$')
    # plt.ylabel(r'$w_\mathrm{cc}/w_\star$ (-)', fontsize=fsize)
    plt.ylabel(r'$w$ (ms-1)', fontsize=fsize)
    plt.xlabel('t (h UTC)', fontsize=fsize)
    plt.legend(fontsize=12)
    plt.ylim(0,2)
    plt.title("BOMEX")
    # plt.title("ARM")
    plt.show()

    
    #%%
    mlm_saveloc ='new_outputs/testcase.txt'   # first create a dir "new_outputs", the script doesnt do this yet
    init_params = sv()
    init_params.tstart = 1
    if init_params.tstart == 0:
        if not bomex: 
            init_params.h = 50
            init_params.theta0 = 299 + 1.25
            init_params.q0 = 15.185e-3
            init_params.theta = 299 + 2 * 1.25
            init_params.q = None
    else:
        it = np.searchsorted(data.time, init_params.tstart*3600)
        # init_params.theta = thml[it]
        # init_params.q = qtml[it]
        # init_params.h = h_ml[it]
        
        init_params.h = 140
        init_params.q = 15.3e-3 + 0.2e-3
        init_params.theta = 301.4 
        if bomex:
            init_params.theta0 = 298.7
            init_params.q0 = 17e-3
        else:
            init_params.theta0 = 301.4 + 0.4 - 140 * 3.4e-3  # 299
            init_params.q0 = 15.3e-3 - 0.2e-3 + 140 * 0.6e-6    #15.2e-3
    #%%
    mlm_out1 = runmodel.run_mlm(mlm_saveloc, init_params, sw_cu=False, sw_Wouter=False)
    mlm_out2 = runmodel.run_mlm(mlm_saveloc, init_params, sw_cu=True, sw_Wouter=False)
    mlm_out3 = runmodel.run_mlm(mlm_saveloc, init_params, sw_cu=True, sw_Wouter=True)
    
    #%%

    plt.figure()
    plt.plot(data.time/3600, qtml * 1e3)
    plt.plot(mlm_out1.t, mlm_out1.q * 1e3, label='Simple MLM')
    plt.plot(mlm_out2.t, mlm_out2.q * 1e3, label='MLM Cu')
    plt.plot(mlm_out3.t, mlm_out3.q * 1e3, label='MLM Cu+CIN')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(data.time/3600, thml)
    plt.plot(mlm_out1.t, mlm_out1.theta, label='Simple MLM')
    plt.plot(mlm_out2.t, mlm_out2.theta, label='MLM Cu')
    plt.plot(mlm_out3.t, mlm_out3.theta, label='MLM Cu+CIN')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data.time/3600, h_ml)
    plt.plot(mlm_out1.t, mlm_out1.h, label='Simple MLM')
    plt.plot(mlm_out2.t, mlm_out2.h, label='MLM Cu')
    plt.plot(mlm_out3.t, mlm_out3.h, label='MLM Cu+CIN')
    plt.legend()
    plt.show()
    
    #%%
    plt.figure()
    plt.plot(mlm_out3.t, 1.113*1005*mlm_out3.wtheta)
    plt.plot(mlm_out3.t, 1.113 * 2.5e6 * mlm_out3.wq)
    plt.show()



