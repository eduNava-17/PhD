"""
Created on Fri Jan  5 18:20:47 2024
@author: Edu

Let us start by importing all the packages that will be used or potentially used.

"""

from sys import path
import os.path
root = os.getcwd()
path.append(root + r'/Nextcloud\Documents\Jose PhD project\python scripts/')
import numpy as np
import pandas as pd 
import csv
from matplotlib import pyplot as plt  # can be written as import matplotlib.pyplot as plt
import func as fn
import openpyxl
from openpyxl.styles import PatternFill
import datetime
import Cooling_dynamics_funcs as cdf
from scipy.optimize import curve_fit
from time import time
import threading
from scipy import sparse
import h5py
from numba import jit
from scipy.sparse import csr_matrix, lil_matrix
import xlwings as xw

atime = time() # keeping track of how much time the simulation takes. This starts the timer.
dateToday = str(datetime.date.today())

wn_to_eV = 1.23984e-4  # multiply by to convert Wavenumbers to electronvolts
Emax = 195000 # The max number in the energy grid, this number will depend on your initial distribution

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(2,2)
axs = gs.subplots()

''' The initial data will be the frequencies of the vibrational modes and intensities. 
We will calculate the density of states by calling the function bs_dens_wn from Cooling_dynamics_funcs.py'''

# exp_path_70 = r'D:\c70\June_19/'
x_c70, y_c70, err_c70 = cdf.xl_clean(r'D:\c70/' + '2024-corrected_sptDecay_linearbins.xlsx','Sheet1')
freq_e = 'freqs_6-311G.txt'
ints_e = 'ints_6-311G.txt'
f_i = np.array(cdf.get_intensities(exp_path_70 + ints_e))
f_n = np.array(cdf.get_frequencies_space(exp_path_70 + freq_e)).astype(int) # in wavenumbers


pathc = r'D:\C60_C70_paper_figures\discussion\all_theory_levels/'
f_i1 = np.array(cdf.get_intensities(pathc + 'B3LYP-6-31+Gd_intens.txt'))
f_n1 = np.array(cdf.get_frequencies_space(pathc + 'B3LYP-6-31+Gd_freqs.txt')).astype(int)
freqs_mine = np.dstack((f_n1,f_i1))
sorted_freqs_mine = np.sort(freqs_mine[0][:,],0)
freqs_array = np.array(sorted_freqs_mine.T[0]).astype(int)
ints_array = np.array(sorted_freqs_mine.T[1])

axs[1,1].scatter(f_n1,f_i1,marker = '^',edgecolors = 'black', label = 'B3LYP/6-31+G(d)')
axs[1,1].scatter(f_n,f_i,marker = 'o',edgecolors = 'green', label = 'B3LYP/6-31+G(d)')

axs[1,1].set_xlabel(r'Energy (cm$^{-1}$)')
axs[1,1].set_ylabel('Intensities')
axs[1,1].legend(loc = 'best')

zero_point_energy = np.sum((1/2)*f_n1*wn_to_eV)

print ('zero point energy:' , round(zero_point_energy,3) , ' eV')

rho_states = cdf.bs_dens_wn(Emax, f_n1)*wn_to_eV

dim = int(round(412/206,0))

axs[0,0].plot(rho_states)
axs[0,0].set_yscale('log')
axs[0,0].set_ylim([1e-2,1e7])
axs[0,0].set_xlim([0,3000])
axs[0,0].set_ylabel('No. states')
axs[0,0].set_xlabel(r'Energy (cm$^{-1}$)')


''' setting IR, RF and VAD matrices'''

rf_states = {'excited energies':np.array([0.13,0.27]),
      'osc. strengths':np.array([6.46e-5,5.9e-2])/25}

ea0 = 2.77
freq_fac = 2e13
# freq_fac = 3.5e13

#3.5e13
photonEmissionIsOn = False

IR_tot, RF_rate_grid, VAD_rate_grid, k_huge  = cdf.get_k_huge(Emax,
                                                              f_n1, # for parent
                                                              f_i1,
                                                              f_n1, # for daughter
                                                              rf_states,
                                                              ea0,
                                                              freq_fac,
                                                              photonEmissionIsOn)


# getting the IR matrix
# As =0.000000125*f_n1**2*f_i1
# IR_rate_grid = np.zeros((len(f_n1),Emax))
# for s in range(len(f_n1)):
#     print ('In freq: ', f_n1[s], 'n will go up to', int(Emax / f_n1[s]))
#     print (len(f_n1) - s, ' freqs left')
#     if f_i1[s] != 0:
#         IR_rate_grid[s] = cdf.get_ratio_rhos(IR_rate_grid[s],rho_states,f_n1[s],As[s], Emax)

total_k_diag = k_huge.diagonal()
total_k_diag_no_zero = np.where(abs(total_k_diag) < 10, 10,abs(total_k_diag))
elem_10 = np.where(abs(total_k_diag) < 10)[0]
h_dynamic3 = total_k_diag_no_zero


energy_grid = np.arange(1,Emax + 1,1)
ini_temp = 1500
g_boltzman = cdf.get_initial_Boltzmann_wn(energy_grid[:-1],ini_temp,rho_states)

pars_g0 = [1,10000,100000]

e_vib = np.arange(0.01,22,1e-2)
pg00,covg00 = curve_fit(cdf.g_dist_a,energy_grid[:-1],g_boltzman,[*pars_g0],maxfev=10000)
g_gaussian = cdf.g_dist_a(energy_grid[:-1],*pg00)
peak_func = VAD_rate_grid*np.exp(-VAD_rate_grid*(1e-4))


axs[0,1].plot(energy_grid,IR_tot, label = r'k_${IR}$')
axs[0,1].plot(energy_grid,VAD_rate_grid, color = 'black', label = r'k_${e}$')
axs[0,1].plot(energy_grid[:-1],g_boltzman, color = 'black')
axs[0,1].axhline(y = 1e-3)

for i in range(len(RF_rate_grid)):
    axs[0,1].plot(energy_grid,RF_rate_grid[i], linestyle = 'dashed',
             label = r'k$_{RF} = $' + str(rf_states['excited energies'][i]) + ' eV, osc.:' + str(rf_states['osc. strengths'][i]))
    
axs[0,1].set_yscale('log')
axs[0,1].set_ylim([1e-4,1e6])
axs[0,1].legend(loc = 'best', fontsize = 10)
axs[0,1].set_xlabel('Energy (cm$^{-1}$)')
axs[0,1].set_ylabel('IR Rate Constant (s$^{-1}$)')
axs[0,1].grid()


num_zeros = 610
zeros = np.where(rho_states[:num_zeros] == 0)[0]
no_zeros = np.where(rho_states[:num_zeros] != 0)[0]
nearest_indices = [fn.find_nearest(no_zeros, elem)[1] for elem in zeros]

# data_cooling = {'x': energy_grid*wn_to_eV,
#                 'y': VAD_rate_grid,
#                 'z': RF_rate_grid[0],
#                 'z2':RF_rate_grid[1],
#                 'w': IR_tot
#                 }

# df1 = pd.DataFrame(data_cooling)
# df1.to_excel(r'D:\c70/' + 'c70_all_ks_for_paper_KH.xlsx', index = False)
 

' setting a faster dynamical time step'

def double_exp(x,a1,tau1,tau2,c):
    return a1*(np.exp(-x/tau1 + 0*-x/tau2)) + c

def dyn_line(x,m,b):
    return m*x + b

# num_cte = 9631
num_cte = 30000
exp_path_70 = r'D:\c70/'
#D:\c70\energy_distributions
pars1 = [0.012,10000,15000,0.001]
y20 = 0.02
h_dynamic4 = 1/(h_dynamic3*3e2)
slope = -(y20 - h_dynamic4[num_cte - 2])/num_cte
pars_line = [slope,y20]
# x_arange_dynamic = np.arange(num_cte,40000,1)
x_guess = np.arange(0,40000,1)
# y_guess = double_exp(x_arange_dynamic,*pars1)
# y_line = dyn_line(x_guess,*pars_line)

h_dynamic4_faster = h_dynamic4.copy()
h_dynamic4_faster[:num_cte] = dyn_line(x_guess[:num_cte],*pars_line)

axs[1,0].plot(h_dynamic4[:50000])
axs[1,0].plot(h_dynamic4_faster[:50000])

# print ('Setting took: ' + str(round(time() - atime,4)) + ' s ')


#%%

rs = xw.Book(exp_path_70 + 'cooling_dynamic_vectors_c70_1300K_finer.xlsx').sheets['Sheet1']
r1 = rs.range('A2:A79').value
r2 = rs.range('B2:B79').value

print (r1)

#%%
from time import time
from colorama import Fore, Back, Style


atime = time()

t0 = 0
tf = 0.06
t_initial_decoupling = 1.7
E_decoupling = [1070]
temps = np.arange(1500,1510,100)
switch_on = [True for i in range(len(E_decoupling))]
neutral_rate = [[] for i in range(len(temps))]
storage_times_rate = [[] for i in range(len(temps))]

# pretty_times = ['0', '100 us', '700 us', '1.3 ms',
#                 '2.4 ms', '4.5 ms', '8.3 ms', '15 ms',
#                 '28 ms', '50 ms', '0.1 s', '0.2 s',
#                 '0.3 s', '0.6 s', '1 s', '2 s']

for tn, edec in enumerate(E_decoupling):
    print (Back.GREEN + 'Temp ' + str(temps[0]) + ' K')
    print (Style.RESET_ALL)

    cplt = np.concatenate((np.array(x_c70[:10]),np.array(x_c70[10:500:10])))
    # cplt = r1 + [94,95]
    # cplt = np.arange(1.8,15.1,0.2)
    # cplt = np.array([0,100e-6,700e-6,1.3e-3,2.4e-3,4.5e-3,8.3e-3,15e-3,
                    # 28e-3,50e-3,0.1,0.2,0.3,0.6,1,2,2.2])

    counter = tn
    energy_grid_cut,VAD_rate_grid_cut,k_huge_cut,ini_distribution,h = cdf.Boltzmann_and_the_cut(VAD_rate_grid,
                                                                                                k_huge,
                                                                                                temps[0],
                                                                                                rho_states,
                                                                                                min(f_n1),
                                                                                                1e-3,
                                                                                                h_dynamic4)
    
    'This one is IVR ON all times'
    storage_times, cooling_dynamics, mean_energies, centers = cdf.RK2_decoupling(t0,tf,h,ini_distribution,k_huge_cut,
                                                                              cplt,zeros,nearest_indices,h_dynamic4_faster,
                                                                            f_n1,f_i1,
                                                                            switch_on[tn],t_initial_decoupling,counter,temps[0])
    
    
    'This one is for decoupling'
    # storage_times, cooling_dynamics, mean_energies, centers = cdf.RK2_decoupling_Gsplit(t0,tf,h,
    #                                                                                     ini_distribution,
    #                                                                                     k_huge_cut,
    #                                                                           cplt,zeros,
    #                                                                           nearest_indices,
    #                                                                           h_dynamic4_faster,
    #                                                                           f_n,f_i,
    #                                                                           switch_on[tn],
    #                                                                           edec,
    #                                                                           counter,
    #                                                                           temps[0],
    #                                                                           IR_rate_grid)
    
    neutral_rate[tn] = cdf.neutral_rate_function(energy_grid_cut,np.array(storage_times),cooling_dynamics,VAD_rate_grid_cut)
    storage_times_rate[tn] = storage_times
    
    # data_cooling = {'Storaged time': storage_times,
    #                 'mean energies (arb. units)': mean_energies
    #                 }

    # df1 = pd.DataFrame(data_cooling)
    # df1.to_excel(r'D:\c70\Gsplit_simulation_6-311G/' + 'G_split_{}eV_cut_1e-3_interp_totalEnergy.xlsx.xlsx'.format(edec), index = False)


#%%


''' least-squares '''

ydata = np.concatenate((np.array(y_c70[4:10]),np.array(y_c70[10:470:10])))
sigma_data = np.concatenate((np.array(err_c70[4:10]),np.array(err_c70[10:470:10])))



# def mod_master(T):
#     neutralRate = cdf.neutral_rate_function(cooling_dyn(T))
#     return neutralRate

# def model_true(a,T):
#     return a*mod_master(T)

def model(a):
    return a*mod[:]


# Define the residuals (difference between observed and predicted values)
def residuals(params):
    a = params
    return (ydata - model(a))/sigma_data

# Define the sum of squared residuals function (objective function)
def sum_of_squares(params):
    return np.sum(residuals(params) ** 2)


def numerical_jacobian(x,a, epsilon=1e-8):
    # Jacobian matrix, with dimensions (N, m)
    N = len(x)
    m = 1
    J = np.zeros((N, m))
    # Perturb a to estimate ∂f/∂a
    J[:, 0] = (model(a + epsilon) - model(a)) / epsilon
    
    return J

# Gradient Descent settings
learning_rate = 1e-8
tolerance = 1e-9
max_iterations = 250000

# Initial guesses for parameters
a_opt = [[] for i in range(1)]
parameter_uncertainties = []
chi_square = [[] for i in range(1)]
# chi squared of 1325 K: 2479315339
# chi squared of 1325 K: 4496661335

fig = plt.figure()
gs = fig.add_gridspec(1)
axs = gs.subplots()

# Gradient Descent loop
for j in range(1):
    for i in range(len(temps)):
        params = [ydata[0]/neutral_rate[i][1]]
        print ('ini param:',params[0])
        mod = neutral_rate[i][5:]
        for iteration in range(max_iterations):
            # Current values of a and b
            a = params[0]
        
            # Calculate partial derivatives using finite differences
            epsilon = 1e-10
            gradient_a = (sum_of_squares(a + epsilon) - sum_of_squares(a)) / epsilon
        
            # Update parameters
            new_params = params - learning_rate * np.array([gradient_a])
        
            # Check for convergence
            converg = np.linalg.norm(new_params - params)
            # print (temps[i], converg)
            if np.linalg.norm(new_params - params) < tolerance:
                print("Converged after", iteration, "iterations")
                break
        
            params = new_params
    
        # Results
        print ('convergence reached:', converg)
        a_opt[j].append(params[0])
        
        chi_square[j].append(np.sum((model(params[0]) - ydata)**2))
        
        J = numerical_jacobian(ydata, params[0])
        W = np.diag(1/sigma_data**2)
        # lamb_small = 1e-9
        # JT_W_J_inv = np.linalg.inv(J.T @ W @ J + lamb_small*np.eye(J.shape[1]))
        JT_W_J_inv = np.linalg.inv(J.T @ W @ J)
        
        covariance_matrix = JT_W_J_inv
        parameter_uncertainties.append((np.sqrt(np.diag(covariance_matrix))))

    axs.scatter(temps[:],chi_square[j], label = j)
    
axs.set_ylabel(r'$\chi^2$')
axs.set_xlabel(r'Temperature (K)')
axs.legend(loc = 'best')
print("Optimal parameters: a =", a_opt)
print ("Parameter uncertainties (standard deviations):", parameter_uncertainties)
print ('Chi squared:', chi_square)


#%%

fig = plt.figure()
gs = fig.add_gridspec(1)
axs = gs.subplots()

for i in range(len(temps)):
    axs.scatter(temps,chi_square[0])

axs.set_ylabel(r'$\chi^2$')
axs.set_xlabel(r'Temperature (K)')


#%%

    fig = plt.figure()
    gs = fig.add_gridspec()
    axs = gs.subplots()

    # data_cooling = {'Storaged time': storage_times, 'mean energies (arb. units)':neutral_rate}

    # df1 = pd.DataFrame(data_cooling)
    # df1.to_excel(exp_path_70 + 'Madrid_RF_cooling_dynamic_neutralRates_c70_{}_K.xlsx'.format(temp), index = False)


    # n_rate_pt = fn.find_nearest(storage_times,x_lr[0])[0]
    
    # axs[1].plot(storage_times,neutral_rate, label = 'temp: ' + str(temp) + ' K,')
    for i in range(len(temps)):
        l = i
        plt.plot(storage_times_rate[l][1:],neutral_rate[l][1:]*a_opt[0][l]*60,lw = 3,
                 alpha = 0.8,label = 'simulation, temp: ' + str(temps[l]) + ' K,')

    data_cooling = {'Storaged time': storage_times_rate[l][1:], 
                    'mean energies (arb. units)':neutral_rate[l][1:]*a_opt[0][i]}
    
    df1 = pd.DataFrame(data_cooling)
    # df1.to_excel(r'D:\c70/' + 'c70-Decay-{}K_corrected_6-31+Gd.xlsx'.format(temps[l]), index = False)


    # axs.plot(storage_times,neutral_rate*y_lr0[0]/neutral_rate[n_rate_pt], label = 'B3LYP/aug-cc-pVDZ, osc./50, temp: ' + str(temp) + ' K,')
    axs.errorbar(x_c70[:], y_c70[:], yerr=err_c70[:],fmt ='.k',
                           capthick=1,capsize=1,elinewidth=1.5,alpha = 0.2, label = 'Spt. decay')
    
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylim([1e0,1e6])
axs.set_xlim([4e-4,1e-1])
axs.set_ylabel(r'Neutral yield (counts s$^{-1}$)')
axs.set_xlabel('time after creation (s)')
axs.legend(loc ='best')
# axs[1].set_yscale('log')
# axs[1].set_xscale('log')
# axs[1].set_ylim([0.9e-3,3e8])
# axs[1].set_xlim([1e-4,1e0])

print ('Simulation took: ' + str(round(time() - atime,4)) + ' s ')

#%%


import matplotlib
matplotlib.rcParams.update({'font.size': 11})

NUM_COLORS = 15
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

selected = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# selected = selected + [18]
pretty_times = ['0', '100 us', '700 us', '1.3 ms',
                '2.4 ms', '4.5 ms', '8.3 ms', '15 ms',
                '28 ms', '50 ms', '0.1 s', '0.2 s',
                '0.3 s', '0.6 s', '1 s', '2 s']

cols = ['black']

for ini, i in enumerate(selected):
    # peak_func = VAD_rate_grid*np.exp(-VAD_rate_grid*(storage_times[i]))
    enes_cm = np.arange(0,len(cooling_dynamics[i]),1)
    enes_eV = enes_cm*wn_to_eV
    if ini == 0:
      ax.plot(enes_eV,cooling_dynamics[i][:],label = pretty_times[ini],
              lw = 2, color = 'darkred')
    else:      
        ax.plot(enes_eV,cooling_dynamics[i][:],label = pretty_times[ini],
            lw = 2)
        # ax.plot(peak_func/max(peak_func), linestyle = 'dashed')

ax.legend(bbox_to_anchor=(0.9, 0.1),
          fancybox=True, shadow=True,fontsize = 10)

ax.set_ylim([0.5e-2,1.02])
ax.set_yscale('linear')
ax.set_ylabel('Relative population', fontsize = 13)
# ax.set_xlabel(r'Excited internal energy (cm$^{-1}$)', fontsize = 13)
ax.set_xlabel(r'Internal energy (eV)', fontsize = 13)

plt.savefig('D:\phd thesis\images/' + 'cooling_dynamics_IVR_SHC_v2_c70-.pdf'
            , dpi = 300, bbox_inches='tight')


#%%


plt.figure(5)
plt.scatter(storage_times, mean_energies)
plt.xscale('log')

#%%

data_cooling = {'Storaged time': storage_times,
                'mean energies (arb. units)': mean_energies
                }

df1 = pd.DataFrame(data_cooling)
df1.to_excel(r'D:\c70\Gsplit_simulation/' + 'G_split_2500eV_cut_1e-3.xlsx.xlsx', index = False)
   

#%%
# pdDF_test_x = pd.read_csv(exp_path_70 + '2200eV_individual_times_for_matrix_trajectories_c70_2024.csv',header = None, delimiter = ',')
# zrx = pdDF_test_x.values

pdDF_test_x = decoupling_times[:] - decoupling_times[0] + cplt[0]
zrx = pdDF_test_x
files_trajectories = glob.glob(exp_path_70 + 'norm_2200eV_all_matrix_trajectories_c70_2024*.csv')
cplt2 = cplt
matrix_trajectories = np.zeros((len(cplt2)-1,10001))
elemental_times = []

# plt.figure()
for i in range(len(cplt2)-1):
    pdDF_test = pd.read_csv(files_trajectories[i],header = None, delimiter = ',')
    zr = pdDF_test.values
    # x_test = zrx[:,0] - zrx[0,0] + cplt2[0]
    x_test = zrx
    # plt.plot(zrx[:,0] - zrx[0,0] + cplt2[i],zr[:,0])
    pt_zero = fn.find_nearest(np.array(x_test),cplt2[i])[0]
    for r in range(10001 - pt_zero):
        matrix_trajectories[i][pt_zero + r] = zr[r]

        
# plt.yscale('log')
# plt.xscale('log')

fig, ax = plt.subplots()
fig1 = plt.gcf()
p = ax.imshow(matrix_trajectories,aspect = 'auto',interpolation='nearest')
cbar = fig1.colorbar(p, pad = 0.15)

plt.figure()
for i in range(len(cplt2)):
    plt.plot(x_test,matrix_trajectories[i], label = str(cplt2[i]) + ' s')

plt.yscale('linear')
plt.xscale('log')


#%%

import glob

# alternative_path = 'D:\c60_2024/'
files_log = glob.glob(exp_path_70 + '2200eV_c70_energy_distribution_at*.csv')
plt.figure(19)
values_cutoff = []
data_1300 = np.loadtxt(exp_path_70 + 'cooling_dynamics-corrected_for_zero_and_first_modes1300_False_norm_pt_1.085s.csv')
x_1300 = data_1300.T[0]
y_1300 = data_1300.T[1]
energy_cut_0 = 2200
divisions_test = np.array([48, 96, 144, 192, 240, 288, 334,
                           412, 460,508, 564, 832, 1252, 1672, 2092, 2200])+1 # by hand

for i in range(1):
    pdDF = pd.read_csv(files_log[-1],header = None, delimiter = '\t')
    zr = pdDF.values
    values_cutoff.append(zr[energy_cut_0,0])
    plt.plot(zr[:,0], alpha = 0.2)
 
for j in range(len(divisions_test)):
    plt.axvline(x = divisions_test[j], color = 'black', alpha = 0.2)

values_cutoff = np.array(values_cutoff)/np.sum(np.array(values_cutoff))
plt.xlim([0,10000])
plt.axvline(x = energy_cut_0, color = 'red')
plt.yscale('log')
plt.ylim([1e-2,1e20])

plt.figure(21)
plt.plot(values_cutoff)


#%%
fig,ax = plt.subplots()
ax.errorbar(middles, np.array(y_storage_times)/(norm_lifetimes*(stop - injection)), yerr=err_storage_times/(norm_lifetimes*(stop - injection)), fmt='ok',markersize = 10,alpha = 0.2, capthick=1.5,capsize=3,elinewidth=1.5)
ax.plot(x_test,
        one_for_all, color = 'purple',lw = 3)
ax.plot(x_1300[:], y_no3[:],lw = 3,color = 'black', label = 'IVR ON')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1,120])
ax.set_ylim(1e3,1e6)

cols = ['k','r','b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r']

#%%

from matplotlib.colors import LogNorm

def double_exp(x,a0,d,a1,tau1):
    return a0*x**d + a1*(np.exp(-x/tau1))

time_element = 9
pt_for_n = fn.find_nearest(np.array(storage_times),middles[time_element])[0]
pt1 = fn.find_nearest(np.array(x_1300),1.085)[0]
y_no3 = np.array(y_1300[:])*y_data[0]/y_1300[pt1]

x_grid = np.arange(1,100,0.01)
pars = [24500,-1,31000,2]
y_guess = double_exp(x_grid,*pars)
# p0, cov0 = curve_fit(double_exp,x_1300[:20],y_no3[:20],[*pars],maxfev = 50000)
y_fit =double_exp(x_grid,*pars)
cols = ['k','r','b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r']

fig,ax = plt.subplots()
nums_traj = len(linear_steps)-1
time_correction = 10.2

ax.errorbar(middles, np.array(y_storage_times)/(norm_lifetimes*(stop - injection)), yerr=err_storage_times/(norm_lifetimes*(stop - injection)), fmt='ok',markersize = 10,alpha = 0.2, capthick=1.5,capsize=3,elinewidth=1.5)
individual_times = []
elemental_times = []
matrix_trajectories = np.zeros((nums_traj,10001))
one_for_all_trajectory =[]

for j in range(nums_traj):
    pt_noivr = fn.find_nearest(np.array(x_grid),cplt[j])[0]
    if j == 0:
        pt_initiate = pt_noivr
    individual_times.append(np.array(decoupling_times[:]) - time_correction + cplt[0])
    ax.plot(np.array(decoupling_times[:]) - time_correction + cplt[j], 
            np.array(total_mean_energy_trajectories[j])*y_fit[pt_noivr]/total_mean_energy_trajectories[j][0],
            lw = 1,linestyle = 'dashed',
            color = cols[j], label = str(temp) + ' K, ' + str(cplt[j]) + ' s')
    
    elemental_times.append(fn.find_nearest(np.array(individual_times[-1]),cplt[j])[0])
    for r in range(10001 - elemental_times[-1]):
        matrix_trajectories[j][elemental_times[-1] + r] = total_mean_energy_trajectories[j][r]*y_fit[pt_noivr]/total_mean_energy_trajectories[j][0]
    
    np.savetxt(exp_path_70 + 'norm_2200eV_all_matrix_trajectories_c70_2024_{}.csv'.format(cplt[j]),np.array(total_mean_energy_trajectories[j])*y_fit[pt_noivr]/total_mean_energy_trajectories[j][0], delimiter = ',')

# np.savetxt(exp_path_70 + '2500eV_individual_times_for_matrix_trajectories_c70_2024.csv',individual_times[0], delimiter = ',')

plt.legend(loc = 'best')
# for j in range(10001):    
#     one_for_all_trajectory.append(np.sum(matrix_trajectories.T[j]))

ax.plot(np.array(decoupling_times[:]) - time_correction + cplt[0],
        one_for_all*y_fit[pt_initiate]/one_for_all[0], color = 'purple',lw = 3)
ax.plot(x_1300[:], y_no3[:],lw = 3,color = 'black', label = 'IVR ON')
# ax.plot(x_grid,y_fit, color = 'pink')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim([1,120])
ax.set_ylim(1e3,4e4)

# np.savetxt(exp_path_70 + '2024_all_trajectories_at_all_times.csv',total_mean_energy_trajectories)
# print (elemental_times)

fig, ax = plt.subplots()
fig1 = plt.gcf()
p = ax.imshow(matrix_trajectories,aspect = 'auto',interpolation='nearest')
cbar = fig1.colorbar(p, pad = 0.15)

plt.figure()
for i in range(nums_traj):
    plt.plot(individual_times[0],matrix_trajectories[i], label = str(cplt) + ' s')

plt.yscale('log')
plt.xscale('log')

# np.savetxt(exp_path_70 + '2500eV_all_matrix_trajectories_c70_2024.csv',
           # matrix_trajectories, delimiter = ',')

# with open(exp_path_70 + '2500eV_all_matrix_trajectories_c70_2024.csv','w') as ar:
#     csv_write = csv.writer(ar,delimiter = ' ')
#     for i in range(len(matrix_trajectories)):
#         csv_write.writerow(matrix_trajectories[i])
    
# with open(exp_path_70 + '2500eV_individual_times_for_matrix_trajectories_c70_2024.csv','w') as ar:
#     csv_write = csv.writer(ar,delimiter = ' ')
#     csv_write.writerow(individual_times[0]) 
        
#%%

one_for_all = []
for j in range(10001):
    counts = np.where(matrix_trajectories.T[j] != 0)[0]
    values_cutoff_updated = values_cutoff[:len(counts)]
    values_cutoff_updated = np.array(values_cutoff_updated)/np.sum(np.array(values_cutoff_updated))
    # print (values_cutoff_updated)
    numerator = np.dot(matrix_trajectories.T[j][:len(counts)],values_cutoff_updated)
    # numerator =  np.average(matrix_trajectories.T[j][:len(counts)],values_cutoff_updated)
    one_for_all.append(numerator/np.sum(values_cutoff_updated))

one_for_all = np.array(one_for_all)
# print (np.shape(np.array(one_for_all)))
# one_for_all = np.array(one_for_all)
# print (len(one_for_all))

#%%
    
    # neutral_rate = cdf.neutral_rate_function(energy_grid,np.array(storage_times),cooling_dynamics,VAD_rate_grid2)

    divisions_test = [440,880]
    total_mean_energies = np.zeros(10001)
    fig,ax = plt.subplots()
    # time_element = 9
    # pt_for_n = fn.find_nearest(np.array(storage_times),middles[time_element])[0]

    for r in range(len(divisions[:1])):
        print ('starting with: ' + str(divisions[r]) + ' cm-1')
        storage_times, all_energy_trajectories, mean_energies, occupation, ini_sum,end_sum = cdf.probability_energy_means(50000,
                                                                                                                            2500,
                                                                                                                            f_n,
                                                                                                                            f_i,
                                                                                                                            rho_states,
                                                                                                                            r'D:\c70\June_19/' + freq_e)
    
        time_element = 9
        pt_for_n = fn.find_nearest(np.array(storage_times),middles[time_element])[0]
        
        
        if r == 0:
            ax.errorbar(middles, np.array(y_storage_times)/(norm_lifetimes*(stop - injection)), yerr=err_storage_times/(norm_lifetimes*(stop - injection)), fmt='ok',markersize = 10,alpha = 0.2, capthick=1.5,capsize=3,elinewidth=1.5)
        
        ax.plot(storage_times[0:], 
                np.array(mean_energies[0:])*(y_storage_times[time_element]/(stop[time_element] - injection[time_element]))/mean_energies[pt_for_n],
                alpha = 0.4, label = str(divisions[r]) + ' cm-1')
        
        total_mean_energies += np.array(mean_energies)*energy_slices_norm[r]


    ax.plot(storage_times[0:], 
            np.array(total_mean_energies[0:])*(y_storage_times[time_element]/(stop[time_element] - injection[time_element]))/total_mean_energies[pt_for_n],
            lw = 3,
            color = 'purple', label = str(temp) + ' K')
    
    # data_for_save = list()
    # for j in range(len(storage_times)):
    #     data_for_save.append([storage_times[j],total_mean_energies[j]])    
        
    # np.savetxt(exp_path_70 + 'cooling_dynamics-NO_IVR_shuffle_half_energy_slices_at_5s.csv',data_for_save)

    # for i in range(10):
    #     ax.plot(storage_times[0:],
    #             np.array(all_energy_trajectories[i][0:])*(y_storage_times[time_element]/(stop[time_element] - injection[time_element]))/all_energy_trajectories[i][pt_for_n],
    #             )
   
    # ax.plot(storage_times[0:], np.array(mean_energies[0:])*(y_storage_times[7]/mean_energies[pt_for_n]), label = str(temp) + ' K')

    # ax.plot(storage_times[1:], np.array(mean_energies[1:]), label = str(temp) + ' K')

    
    # plt.scatter(storage_times, centers)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.xscale([])
    ax.set_xlabel('Storage time (s)')
    ax.set_ylabel(r'Integrated Counts')
    ax.legend(loc = 'best')
    ax.set_xlim([0.99,100])
    ax.set_ylim([8e2,1e5])
    
    
    # # # ax2.plot(storage_times[1:], np.array(mean_energies[1:]), label = str(temp) + ' K')
    # # # ax2.tick_params(axis='y', labelcolor='red')
#%%
    from scipy.signal import find_peaks as fp
    
    slice_of_cooling = cooling_dynamics[-1][1:260]
    slice_for_divisions = fp(slice_of_cooling,distance = 25)[0] + 1 # add +1 to coincide with the frequency energies
    slice_for_divisions_further = np.arange(420,4200,420) - 8 # added -8 to coincide with the max of the distribution, but maybe I should take it out
    divisions = slice_for_divisions.tolist() + slice_for_divisions_further.tolist()
    
    plt.figure(15)
    plt.plot(cooling_dynamics[-1][1:])
    plt.xlim([0,6000])
    energy_slices = []
    for i in range(len(divisions)):
        plt.axvline(x =divisions[i], linestyle = 'dashed', color = 'black', alpha = 0.2)
        energy_slices.append(cooling_dynamics[-1][divisions[i]])
    
    energy_slices_norm = energy_slices[:]/max(energy_slices)
    plt.xlabel('Energy (cm-1)')
    print (energy_slices_norm)
    print (divisions)

    
#%%
    data_for_save = list()
    for j in range(len(storage_times)):
        data_for_save.append([storage_times[j],mean_energies[j]])    
        
    np.savetxt(exp_path_70 + 'cooling_dynamics-NO_IVR_shuffle_{}_1380eV_3.8s.csv'.format(temp),data_for_save)

    for i in range(10):
        data_for_save = []
        for j in range(len(storage_times)):
            data_for_save.append([storage_times[j],all_energy_trajectories[i][j]])
        np.savetxt(exp_path_70 + 'cooling_dynamics-NO_IVR_shuffle_individuals_{}_{}_1380eV_3.8s.csv'.format(temp,i),data_for_save)


    # np.savetxt(exp_path_70 + 'cooling_dynamics-corrected_for_zero_and_first_modes{}_{}_normalization_point_2DMatrix.csv'.format(temp,switch_on[tn]),cooling_dynamics)

    # pt_c = fn.find_nearest(np.array(t_test),storage_times[1])[0]
    
    # plt.figure(21)
    # plt.plot(t_test,h_test)
    # plt.plot(storage_times[1:], np.array(centers[1:])*h_test[pt_c]/centers[1], label = str(temp) + ' K')
    # plt.plot(storage_times[1:], np.array(mean_energies[1:])*h_test[pt_c]/mean_energies[1], label = str(temp) + ' K')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim([0.99,100])

    print ('Simulation took: ' + str(round(time() - atime,4)) + ' s ')
    
    
    # plt.close()
    # plt.figure(101)
    # for i in range(9):
    #     plt.plot(cooling_dynamics[i*5], label = str(round(storage_times[i*5],2)))
    # # plt.plot(cooling_dynamics[23], label = str(storage_times[23]))
    # # plt.plot(cooling_dynamics[24], label = str(storage_times[24]))
    # # plt.plot(cooling_dynamics[25], label = str(storage_times[25]))
    # # plt.plot(cooling_dynamics[26], label = str(storage_times[26]))
    # # plt.plot(cooling_dynamics[27], label = str(storage_times[27]))
    # # plt.plot(cooling_dynamics[28], label = str(storage_times[28]))
    # plt.xlim([-1000,125000])
    # plt.ylim([1e-1,1e2])
    # plt.yscale('log')
    # plt.legend(loc = 'best')
    
    # plt.figure(25)
    # # plt.plot(storage_times[47:],n_rate, label = 'experiment')
    # n_rate_pt = fn.find_nearest(storage_times,x_lr[4])[0]
    # # if tn == 0:
    # plt.plot(storage_times,neutral_rate*y[4]/neutral_rate[n_rate_pt], label = 'wB97XD,nu1, nu2 x 2 temp: ' + str(temp) + ' K,')
   
    # plt.errorbar(x_lr[:], y_lr0[:], yerr=err_lr0[:],fmt ='.k',
    #                 capthick=1,capsize=1,elinewidth=1.5,alpha = 0.2, label = 'Spt. decay')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylabel('Neutral yield (counts)')
    # plt.xlabel('Storage time (s)')
    # plt.xlim([1e-4,0.5e0])
    # plt.legend(loc = 'best')
    # plt.grid() 
    
    #%%
    data_for_save = list()
    for j in range(len(storage_times)):
        data_for_save.append([storage_times[j],neutral_rate[j]])    
       
    np.savetxt(exp_path_70 + 'cooling_dynamics_100ms_wB97_{}_nu1_only.csv'.format(temp),data_for_save)
   
 #    print ('Simulation took: ' + str(round(time() - atime,4)) + ' s ')

#%%

''' Temperature and Excitation energy (Heat capacity) '''

def heat_capacity(x,beta, theta,n):
    kb = 8.6173e-5
    cmax = (3*n - 6)*kb
    return cmax*(1 - np.exp(-beta*x + theta))

e_vib = np.arange(0.1,22,1e-1)
h = 0.1e-2
pars60 = [0.0138,7.4]
parsT = [1,0,70]

temps = np.arange(5,2000,10)

y_guessT = heat_capacity(temps,*parsT)
energies_negativeIon,temp_relation_negativeIon = cdf.microcan_temp(e_vib,rho_states, h)
excitation_energy = cdf.partition_f(temps,f_n)
# poly = np.polyfit(temp_relation_negativeIon, energies_negativeIon, deg=12)
poly_f = np.polyfit(temps, excitation_energy, deg=12)
# poly_f = np.polyfit(temp_relation_c70_negIon,
                    # energies_c70_negIon, deg=10)

elem_line = fn.find_nearest(temp_relation_negativeIon,1000)[0]
p60, cov60 = curve_fit(cdf.line_temp,temp_relation_negativeIon[elem_line:],energies_negativeIon[elem_line:],[*pars60],maxfev=10000)
# pT, covT = curve_fit(heat_capacity)

energies_c70_negIon, temp_relation_c70_negIon = cdf.microcan_temp(e_vib, rho_states, h)
elem_line70 = fn.find_nearest(temp_relation_c70_negIon,1000)[0]
p70, cov70 = curve_fit(cdf.line_temp,temp_relation_c70_negIon[elem_line70:],energies_c70_negIon[elem_line70:],[*pars60],maxfev=10000)


plt.figure(107)
# plt.plot(temps,y_guessT,color = 'yellow')
# plt.scatter(temp_relation_negativeIon,energies_negativeIon, s=17,marker = 'o',facecolors='none', edgecolors='forestgreen', label = 'c60-, B3LYP/6-311G')
# plt.plot(temp_relation_negativeIon,np.polyval(poly,temp_relation_negativeIon),color = 'pink', label='4th-polynomial fit')
plt.plot(temps,np.polyval(poly_f,temps),color = 'pink', label='4th-polynomial fit')
# plt.plot(temps,cdf.line_temp(temps,*p60), color = 'black',linestyle = 'dashed', label = str(round(p60[1],1)) +  ' + ' + str(round(p60[0],4)) + '(T - 1000)' )
plt.scatter(temps,excitation_energy,s=12, label = 'canonical')
plt.scatter(temp_relation_c70_negIon,energies_c70_negIon, s=17,marker = '^',facecolors='none', edgecolors='lightcoral', label = 'c70-, B3LYP/6-311G')
plt.plot(temps,cdf.line_temp(temps,*p70), color = 'red',linestyle = 'dashed', label = r'$7.2 + 0.0136(T -1000)$')

temp_energy = np.polyval(poly_f,temps)
temp_decouple = fn.find_nearest(temp_energy,0.3)[0]
print (temps[temp_decouple])

plt.legend(loc = 'best')
plt.xlabel('Temperature (K)')
plt.ylabel('Excitation energy (eV)')
# plt.ylim([-0.5,24])
# plt.xlim([50,2000])
plt.grid(True)

#%%

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sympy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.colors import LogNorm


transformation = lambda x:np.polyval(poly_f,x)
x_tr = lambda x: 7.4 + 0.0138*(x-1000)


plt.rcParams["figure.figsize"] = (10,6)
matplotlib.rcParams.update({'font.size': 15})

temps = np.arange(1,2000,1)
poly_temps = transformation(temps)

elem_ene1 = fn.find_nearest(poly_temps,46*wn_to_eV)[0]
elem_ene2 = fn.find_nearest(poly_temps,10000*wn_to_eV)[0]

start_IVR_off = mean_energies[28]*wn_to_eV # at 3.47 s
elem_start_IVR_off = fn.find_nearest(poly_temps,start_IVR_off)[0]

print (temps[elem_ene1],temps[elem_ene2])
    
cols = ['k','r','b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r'
        ,'b','m','y','k','r','b','m','y','k','r']

cdict1 = {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1),
                         (1, 0.5, 0.5)),
             'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1),
                         (0.91, 0, 0), (1, 0, 0)),
             'blue':  ((0., 0.17, 0.17), (0.11, 1, 1), (0.34, 1, 1),
                         (0.65, 0, 0), (1, 0, 0))}

jet_new = LinearSegmentedColormap('BlueRed1',
                                    cdict1)


int_method ='gaussian'

pt_i = 1.2e-1
cooling_dynamics_copy = cooling_dynamics
for i in range(len(cooling_dynamics_copy)):
    cooling_dynamics_copy[i][cooling_dynamics_copy[i] < 1e-1] = pt_i
    
fig, ax = plt.subplots()
# plt.figure(5)
fig1 = plt.gcf()
nonlinear_ax = ax.twinx()
# nonlinear_ax.set_yscale('function', functions=(transformation,np.log))
nonlinear_ax.set_ylim([temps[elem_ene1],temps[elem_ene2]])

p=ax.imshow(np.fliplr(cooling_dynamics_copy[20:]).T,aspect = 'auto',cmap =jet_new,interpolation=int_method,
            norm = LogNorm(vmin = pt_i, vmax = 113),extent=[storage_times[20],storage_times[-1],46*wn_to_eV,len(cooling_dynamics[20])*wn_to_eV])
            # extent=[1.5,10,0,len(cooling_dynamics[10])])

plt.axvline(x = 3.47, color = 'white')
plt.axhline(y = temps[elem_start_IVR_off], color = 'white')

ax.set_xscale('log')
ax.set_yscale('linear')

cbar = fig1.colorbar(p, pad = 0.15)
cbar.ax.set_ylabel('Normalized population (arb.units)',fontsize = 15)

num_ticks = 10
ticks = np.linspace(temps[elem_ene1],temps[elem_ene2], 10)
# nonlinear_ax.set_yticks(ticks)
# nonlinear_ax.set_yticklabels([f'{ticks[i]:.0f}' for i in range(len(ticks))])

ax.set_xlabel('Storge time (s)',fontsize = 15)
ax.set_ylabel('Excitation energy distribution (eV)',fontsize = 15)
nonlinear_ax.set_ylabel('Temperature (K)')

# ax.set_yticks([],fontsize = 15)
# ax.set_xticks([],fontsize = 15)
ax.set_ylim([46*wn_to_eV,10000*wn_to_eV])
plt.tight_layout()

#%%

pi = 7
plt.figure(101)
x_ev = np.arange(0,len(cooling_dynamics[0]),1)
# for i in range(9):
for i in range(20):
    # cooling_dynamics[i][cooling_dynamics[i] < 1e-1] = 1e-1
    plt.plot(x_ev,cooling_dynamics[i], label = str(storage_times[i]))
# plt.plot(cooling_dynamics[pi], label = str(storage_times[pi]))
# plt.plot(cooling_dynamics[24], label = str(storage_times[24]))
# plt.plot(cooling_dynamics[25], label = str(storage_times[25]))
# plt.plot(cooling_dynamics[26], label = str(storage_times[26]))
# plt.plot(cooling_dynamics[27], label = str(storage_times[27]))
# plt.plot(x_ev,cooling_dynamics[0], color = 'black')
# plt.xlim([-1000,125000])
plt.ylim([1e-1,1e3])
plt.yscale('linear')
plt.legend(loc = 'best')

#%%

from periodictable import formula
import scipy.constants as const

molecule_amu = formula('OH').mass
ion_mass = molecule_amu * const.atomic_mass
ion_velocity = np.sqrt(2 * 10e3 * const.value('electron volt') / ion_mass)

# vel = 3.369e5
# vel = 5.327e5

c = 299792458
nu0 = c/680.109e-9

nu = nu0*(1 - ion_velocity/c)

print (c/nu)


#%%



