# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:55:54 2023

@author: J Edu

"""

import numpy as np
import func as fn
from scipy.misc import derivative


wn_to_eV = 1.23984e-4  # multiply by to convert Wavenumbers to electronvolts
kb_eV = 8.617333262e-5 #eV K-1
mean_alpha = 76.5 # amstrong

a0_to_angstrom = 0.529  # 1 a0 = 0.529e-10m
hartree_to_eV = 27.21138386
m_mol = 720*1.660538921e-27 # Mass of molecule in u
m_e = 5.485799e-4  # u, electron mass

mu_emol = m_e*m_mol/(m_e+m_mol) # kg, reduced mass
hbar = 6.62607015e-34  # J*s

def get_frequencies(file):
    fs = []
    with open(file, 'r') as f:
        for i in f:
            # if i.startswith(' Frequencies'):
            if i.startswith('Frequencies'):
                a_string = i.split()
                for j in range(3):
                    fs.append(float(a_string[-j-1]))
    return fs

def get_frequencies_space(file):
    fs = []
    with open(file, 'r') as f:
        for i in f:
            # if i.startswith(' Frequencies'):
            if i.startswith(' Frequencies'):
                a_string = i.split()
                for j in range(3):
                    fs.append(float(a_string[-j-1]))
    return fs

def get_excited_states(file):
    fs = list()
    osc = list()
    osc_float = list()
    with open(file, 'r') as f:
        for i in f:
            if i.startswith(' Excited State'):
                a_string = i.split()
                fs.append(float(a_string[4]))
                osc.append(a_string[8])
        for j in range(len(osc)):
            osc_float.append(float(osc[j][2:]))
    return np.array(fs), np.array(osc_float)

def get_intensities(file):
    fi = list()
    with open(file,'r') as f:
        for i in f:
            if i.startswith(' IR Inten'):
                a_string = i.split()
                for j in range(3):
                    fi.append(float(a_string[-j-1]))
    return fi

def get_lifetimes(freqs,ints):
    lifetimes = list()
    for j in range(len(freqs)):
        if ints[j] > 0.0001:
            lifetimes.append(1/(0.000000125*ints[j]*freqs[j]**2))
        else:
            lifetimes.append(1/(0.000000125*1e-6*freqs[j]**2))
    return lifetimes

def energy_temp(temp):
    return 7.4 + 0.0138*(temp - 1000)

def temp_ene_c70(x):
    return 8.4 + 0.0157*(x - 1000)

def line_temp(x,m,b):
    return b + m*(x - 1000)

def line_temp_simple(x,m,b):
    return b + m*(x)

def stepfunction(x,onset):
    return 0.5*(np.sign(x-onset)+1)

def g_dist(ene,sigma,mu):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(ene-mu)**2/(2*sigma**2))

def g_dist_a(ene,a,sigma,mu):
    return a*np.exp(-(ene-mu)**2/(2*sigma**2))

def bs_dens(Evib, vib_wn):
    '''
    SUM OF DENSITY OF STATES:
    Beyer1971'Algorithum 448. Number of Multiply Restricted Partitions[A1]'
    P is the number of ways a given amount of vibratinal energy can be
    distributed among the quantised modes of a polyatomic moleclule

    Evib_ev: total vibrational energy (we convert to wavenumbers)
    vib_wn: vibratinal energy levels for system (anion or neutral).
    It is an array.(as calcualted by Gaussian, direct input into code)
    '''
    N0 = int(Evib // wn_to_eV)
    # N0 = Evib.astype(int)
    print('this is N0: ', N0)

    N = N0
    C = vib_wn.astype(int)  # C has to have a discrete (integer) energy scale for the vibrational levels. 
    #Wavenumbers are suitable for this.
    
    # assert N>=C.max(), 'algorithum fails if any levels are inaccessible'
    if N <= np.max(C):  # N has to be larger than the largest value of C for the algorithm to work.
        N = np.max(C) + 1

    P = np.zeros(N)
    P[0] = 1  # Ground state is not included in the algorithm and is added manually.

    for i in C:
        J = i - 1  # -1 to fit Pythons array indices (staring from 0, Fortran starts from 1)
        P[J] += 1
        for M in range(J+1, N):
            P[M] += P[M-J]
    # import pdb; pdb.set_trace()
    Ps = np.zeros(N)
    for j in range(len(Ps)-1):
        Ps[j+1] = np.sum(P[:j])  # The inegrated (accumulative sum) density.
    # Should not be used, just for testing.
    # print (P/wn_to_eV)
    # print (' ')
    # print (Ps/wn_to_eV)
    return P/wn_to_eV # Dimension is 1/E, so we change it to 1/eV and not 1/wn.

def microcan_temp(enes,rhos,h):
    der_f = list()
    energies = list()
    for ene in enes:
        en_wn1 = int((ene + h)//wn_to_eV)
        en_wn2 = int((ene)//wn_to_eV)
        term = np.log(rhos[en_wn1]) - np.log(rhos[en_wn2])
        if term > 0:
            der_f.append(term/h)
            energies.append(ene)
        # else:
        #     der_f.append(1)

    kb_eV = 8.617333262e-5 #eV K-1
    return energies, (np.array(der_f))**(-1)/kb_eV

def partition_f(temps,freqs):
    ene_ave = np.zeros(len(temps))
    for j in range(len(temps)):
        for i in range(len(freqs)):
            ene = freqs[i]*wn_to_eV
            ene_ave[j] += ene/(np.exp(ene/(kb_eV*temps[j]))-1)
    return ene_ave

def Sigma_Ec(E_electron):
    '''Calculate the cross section for electron attachment. Use Langevin cross
    section in atomic units. sigma=(2*pi*q/v)*sqrt(alpha/mu). Which in our case simplifies to
    sigma=pi*sqrt(2*alpha/E_electron). E_electron is an array so this is an array'''
    alpha = mean_alpha * (1/a0_to_angstrom)**3  # mean_alpha given in angstrom^3 convert to a0^3
    E_electron = E_electron * (1/hartree_to_eV)  # E_electron in eV convert to hartree
    sigma = np.pi * np.sqrt(2*alpha/E_electron)  # this is in atomic units
    sigma_SI = sigma * (a0_to_angstrom * 1e-10)**2  # convert value to m^2
    sigma_constant = 4*np.pi*(0.7e-9)**2
    return sigma_constant # when want constant value

# def function_heat_capacity(energy,zero_point,n,freqs):
#     func = lambda x:np.log(level_densities_haarhoff(energy,zero_point,n,freqs))
#     return 

def get_heat_capacity(energy,zero_point,n,freqs):
    k_eV = 8.617333e-5/wn_to_eV #wn K-1
    x_energies = list()
    y_derivative = list()
    func = lambda x:np.log(level_densities_haarhoff(x,zero_point,n,freqs))
    for ene in energy:
        y_values = func(ene)
        if y_values > 0:
            x_energies.append(ene)
            num = derivative(func,ene, dx=1e-6)
            y_derivative.append(num)
    return np.array(x_energies), (1/k_eV)*(1/np.array(y_derivative))
    
''' rate constant funcs '''

def get_IR_total(energies,freqs,lifetimes,level_densities):
    print ('Inside IR_total rate constant ')
    k_total = np.zeros((len(energies),2))
    enes_n = freqs*wn_to_eV # here i change to eV to 
    k_total.T[0] = energies
    for j in range(len(energies)):
        for i in range(len(freqs)):
            m_int = int(energies[j]/enes_n[i])
            delta_m_int = energies[j]/enes_n[i] - int(energies[j]/enes_n[i])
            m = 0
            ki = 0
            while m < m_int -1:
                m_wv = int((m + delta_m_int)*freqs[i])
                m += 1
                ki += level_densities[m_wv]
            rho_index = int(energies[j] // wn_to_eV)
            k_total[j][1] += ki*(1/lifetimes[i])/level_densities[rho_index]
    return k_total

def get_IR_total_Mark(energies,freqs,intense, level_densities):
    k_total = np.zeros((len(energies),2))
    enes_n = freqs*wn_to_eV 
    k_total.T[0] = energies
    As = (0.000000125**2)*freqs**2*intense
    for j in range(len(energies)):
        # print ('energy: ', str(energies[j]),' eV')
        for i in range(len(enes_n)):
            if intense[i] != 0:
                m_int = int(energies[j] // enes_n[i])
                m = 1
                ki = 0
                while m <= m_int:
                    nu_wv = int(energies[j] - m*enes_n[i])
                    ki += level_densities[nu_wv]
                    m += 1
                rho_index = int(energies[j] // wn_to_eV)
                k_total[j][1] += As[i]*ki/level_densities[rho_index]
    return k_total


#%%

def get_IR_total_wn(energies,freqs,lifetimes,level_densities):
    print ('Inside IR_total rate constant ')
    k_total = np.zeros((len(energies),2))
    k_total.T[0] = energies
    for j in range(len(energies)):
        for i in range(len(freqs)):
            m_int = int(energies[j]/freqs[i])
            delta_m_int = energies[j]/freqs[i] - int(energies[j]/freqs[i])
            m = 0
            ki = 0
            while m < m_int -1:
                m_wv = int((m + delta_m_int)*freqs[i])
                m += 1
                ki += level_densities[m_wv]
            rho_index = int(energies[j])
            k_total[j][1] += ki*(1/lifetimes[i])/level_densities[rho_index]
    return k_total

#%%

def stepfunc(x,limit):
    """
    Step function. Equal to 0 if x < limit, else 1.
    """
    return 0.5*(np.sign(x-limit)+1)

def get_IR_total_Mark_wn(energies,freqs,intense, level_densities):
    k_total = np.zeros((len(energies),2))
    k_total.T[0] = energies
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        for i in range(len(freqs)):
            if intense[i] != 0:
                ki = 0
                nu = 1
                m_int = int(energies[j] // freqs[i])
                while nu <= m_int:
                    nu_wv = int(energies[j] - nu*freqs[i])
                    ki += level_densities[nu_wv]
                    nu += 1
                k_total[j][1] += As[i]*ki/level_densities[energies[j]]
    return k_total

def get_IR_total_Haar(energies,zero_point,n,freqs,intense):
    k_total = np.zeros(len(energies))
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        print ('IR rate constant of Haarhoff, energy_{}: '.format(j), energies[j], ' eV')
        try:
            for i in range(len(freqs)):
                if intense[i] != 0:
                    ki = 0
                    nu = 1
                    m_int = int(energies[j] // freqs[i])
                    while nu <= m_int:
                        ki += level_densities_haarhoff(energies[j] - nu*freqs[i],zero_point,n,freqs)
                        nu += 1
                    den = level_densities_haarhoff(energies[j],zero_point,n,freqs)
                    k_total[j] += As[i]*ki/den
        except OverflowError as oe:
            print ('there is an Overflow in energy {}: '.format(j), energies[j], ki, den)
    return k_total


def get_IR_singles_Haar(energies,zero_point,n,freqs,intense):
    k_singles = np.zeros((len(energies),len(freqs)))
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        print ('IR rate constant of Haarhoff, energy_{}: '.format(j), energies[j], ' eV')
        for i in range(len(freqs)):
            if intense[i] != 0:
                ki = 0
                nu = 1
                m_int = int(energies[j] // freqs[i])
                while nu <= m_int:
                    ki += level_densities_haarhoff(energies[j] - nu*freqs[i],zero_point,n,freqs)
                    nu += 1
                k_singles[j][i] = As[i]*ki/level_densities_haarhoff(energies[j],zero_point,n,freqs)
    return k_singles

#%%

def get_IR_total_Haar2(energies,zero_point,n,freqs,intense):
    k_total = np.zeros((len(energies),2))
    k_total.T[0] = energies
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        print ('IR rate constant of Haarhoff, energy_{}: '.format(j), energies[j], ' eV')
        try:
            for i in range(len(freqs)):
                if intense[i] != 0:
                    ki = 0
                    nu = 1
                    m_int = int(energies[j] // freqs[i])
                    while nu <= m_int:
                        ki += level_densities_haarhoff(energies[j] - nu*freqs[i],zero_point,n,freqs)
                        nu += 1
                    den = level_densities_haarhoff(energies[j],zero_point,n,freqs)
                    k_total[j][1] += As[i]*ki/den
        except OverflowError as oe:
            print ('there is an Overflow in energy {}: '.format(j), energies[j], ki, den)
    return k_total

def get_IR_singles_Haar2(energies,zero_point,n,freqs,intense):
    k_singles = np.zeros((len(energies),len(freqs)+1))
    k_singles.T[0] = energies
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        print ('IR rate constant of Haarhoff, energy_{}: '.format(j), energies[j], ' eV')
        for i in range(len(freqs)):
            if intense[i] != 0:
                ki = 0
                nu = 1
                m_int = int(energies[j] // freqs[i])
                while nu <= m_int:
                    ki += level_densities_haarhoff(energies[j] - nu*freqs[i],zero_point,n,freqs)
                    nu += 1
                k_singles[j][i+1] = As[i]*ki/level_densities_haarhoff(energies[j],zero_point,n,freqs)
    return k_singles

#%%

from numba import jit
from numba import vectorize, float64

# n is the number of oscillators
# eta is epsilon/epsilon_z 
# epsilon is the total vibrational energy in excess of the
# epsilon_z (the zero point energy) 
# alpha^m = mean of the mth powered frequencies / mean of frequencies to the power of m

@jit(nopython=True)
def average(subset):
    n = len(subset)
    get_sum = sum(subset)
    mean = get_sum/n
    return mean

@jit(nopython=True)
def lambda_haar(freqs):
    l_inverse = 1
    for nu in freqs:
        l_inverse *= nu/average(freqs)
    return l_inverse, average(freqs)

@jit(nopython=True)
def level_densities_haarhoff(energy,zero_point,n,freqs):
    h_wn = 4.135667696*10e-15/wn_to_eV
    l_inverse, nu_bar = lambda_haar(freqs)
    eta = energy/zero_point
    alpha_2 = average(freqs**2)/nu_bar**2
    beta = ((n-1)*(n-2)*alpha_2 - n**2)/(6*n)
    braquet_term = (1 + eta/2)*(1 + 2/eta)**(eta/2)
    braquet_term_2 = 1- 1/(1 + eta)**2
    factor = (2/(np.pi*n))**(1/2)*((1-1/(12*n))/(l_inverse*(1 + eta)*h_wn*nu_bar))
    # try:
    rho = factor*braquet_term**n*braquet_term_2**beta
    # except OverflowError:
        # print ('Overflow in Haarhoff expression ', factor,braquet_term**n,braquet_term_2**beta)
    return rho

#%%

def get_IR_singles_wn(energies,freqs,intense,level_densities): 
    print ('Inside IR_singles rate constant ')
    k_singles = np.zeros((len(energies),len(freqs)+1))
    k_singles.T[0] = energies
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        for i in range(len(freqs)):
            m_int = int(energies[j]/freqs[i])
            delta_m_int = energies[j]/freqs[i] - int(energies[j]/freqs[i])
            m = 0
            ki = 0
            while m < m_int -1:
                m_wv = int((m + delta_m_int)*freqs[i])
                m += 1
                ki += level_densities[m_wv]
            rho_index = energies[j]
            k_singles[j][i+1] = ki*(As[i])/level_densities[rho_index]
    return k_singles

def get_IR_singles_Mark_wn(energies,freqs,intense,level_densities):
    k_total = np.zeros((len(energies),len(freqs)+1))
    k_total.T[0] = energies
    As = 0.000000125*freqs**2*intense
    for j in range(len(energies)):
        for i in range(len(freqs)):
            if intense[i] != 0:
                m_int = int(energies[j] // freqs[i])
                m = 1
                ki = 0
                while m <= m_int:
                    nu_wv = int(energies[j] - m*freqs[i])
                    ki += level_densities[nu_wv]
                    m += 1
                k_total[j][i+1] = As[i]*ki/level_densities[energies[j]]
    return k_total



#%%

def get_IR_singles(energies,freqs,lifetimes,level_densities): 
    print ('Inside IR_singles rate constant ')
    enes_n = freqs*wn_to_eV
    k_singles = np.zeros((len(energies),len(freqs)+1))
    k_singles.T[0] = energies
    for j in range(len(energies)):
        for i in range(len(freqs)):
            m_int = int(energies[j]/enes_n[i])
            delta_m_int = energies[j]/enes_n[i] - int(energies[j]/enes_n[i])
            m = 0
            ki = 0
            while m < m_int -1:
                m_wv = int((m + delta_m_int)*freqs[i])
                m += 1
                ki += level_densities[m_wv]
            rho_index = int(energies[j] // wn_to_eV)
            k_singles[j][i+1] = ki*(1/lifetimes[i])/level_densities[rho_index]
    return k_singles

def get_RF(energies,enes_n,As,level_densities):
    print ('Inside RF rate constant ')
    k_RF = np.zeros((len(energies),2))
    k_RF.T[0] = energies 
    for j in range(len(energies)):
        for i in range(len(enes_n)):
            m_wv = int((energies[j] - enes_n[i])//wn_to_eV)
            ki = level_densities[m_wv]*stepfunction(energies[j],enes_n[i])
            rho_index = int(energies[j] // wn_to_eV)
            k_RF[j][1] = ki*(As[i])/level_densities[rho_index]
    return k_RF

def get_RF_s(energies,enes_n,As,level_densities):
    print ('Inside RF rate constant ')
    k_RF = np.zeros((len(energies),len(enes_n)+1))
    k_RF.T[0] = energies 
    for j in range(len(energies)):
        for i in range(len(enes_n)):
            m_wv = int((energies[j] - enes_n[i])//wn_to_eV)
            ki = level_densities[m_wv]*stepfunction(energies[j],enes_n[i])
            rho_index = int(energies[j] // wn_to_eV)
            k_RF[j][i+1] = ki*(As[i])/level_densities[rho_index]
    return k_RF

def get_K_VAD(energies,ea,prefactor,level_densities_p,level_densities_d):
    print ('Inside VAD rate constant ')

    k_VAD = np.zeros((len(energies),2))
    k_VAD.T[0] = energies
    for j in range(len(energies)):
        if energies[j] > ea:
            index_p = int(energies[j]//wn_to_eV)
            index_d = int((energies[j] - ea)//wn_to_eV)
            ratio = level_densities_d[index_d]/level_densities_p[index_p]
            k_VAD[j][1] = prefactor*ratio
    return k_VAD

def get_old_k_VAD(E_vib, EA, dens_neutral, dens_anion):
    ''' INPUT to k_VAD:
        E_vib: vibrational energy - an array and defines energy window for integration
        J: choosen rotational state for rotational energy
        EA: electron affinity
        dens_neutral: calculated sum of density of stated for neutral
        dens_anion: calculated sum of density of stated for anion)

        Other things needed:
        B_neutral: rotational constant for neutral structure from Gaussian (GHz)
        B_anion: rotational constant anion structure from Gaussian

        k_VAD: returns a value k_EJ and energy. We will make the sum over all
        states of epsilon.
        '''
 
    EA_J = EA
    eV_to_J = 1.602176565e-19

    # integration step, energy step. Since we integrate by sum this is step for the sum
    de = 0.0001

    # here we calculate k_VAD using all the stuff above
    k_EJ = np.zeros(len(E_vib))  #
    for n, Ev in enumerate(E_vib):  # enumerate adds an index and returns the value for that index.

        # arange returns evenly spaced values within an interval arange(start, stop, step)
        integration_energies = np.arange(1e-25, Ev-EA_J, de) #it was before modification integration_energies = np.arange(1e-25, Ev-EA_J, de)
        epsilon = integration_energies  # epsilon is the KE of the leaving electron
        E_neutral_rho = Ev - EA_J - epsilon  # this is the energy put into the density of states calc.
        mEp = len(epsilon)
        if dens_anion[(Ev//wn_to_eV).astype(int)] == 0:
            k_EJ[n] = 0  # For special cases where the density of states in the
                                         # denominator is 0 for a given energy..
        else:
            index_neut = ((E_neutral_rho)/wn_to_eV).astype(int)
            index_anion = ((Ev)/wn_to_eV).astype(int)
            # The term ((Ev_neutral_rho//wn_to_eV).astype(int)
            # returns the integer wavenumber value of the energy E_neutral_rho,
            # which is used as the index when searching in the density tables.
            # Integrate by summing.
            k_EJ[n] = np.sum((Sigma_Ec(epsilon) * epsilon * dens_neutral[index_neut] / dens_anion[index_anion])*de)

    k_EJ *= (2*mu_emol/(np.pi**2*hbar**3)*eV_to_J**2)  # include the constants
    print (2*mu_emol/(np.pi**2*hbar**3)*eV_to_J**2)
    return k_EJ
    
#%%

def IR_rates_Mark_script(freq,intense,level_densities):
    A = 0.000000125 * intense * freq ** 2
    p = level_densities
    k = np.zeros((len(freq),len(level_densities)))
    for s in range(len(freq)): 
    	if intense[s] != 0:
    		n = 1
    		while n < 2**17 / freq[s]:
    			k[s,n * freq[s]:] += np.nan_to_num(A[s] * np.asarray(p[:-n * freq[s]]) / np.asarray(p[n * freq[s]:]))	#here the rate coefficients are calculated
    			n = n + 1 
    		nu = s+1	
    return k


    #%%

''' Cooling dynamics '''

kb_wn = 0.695034800

def get_initial_Boltzmann(energies,ini_temp,level_densities):
    g = np.zeros((len(energies),2))
    for i in range(len(energies)):
        rho_index = int(energies[i] // wn_to_eV)
        g[i][1] = level_densities[rho_index]*np.exp(-energies[i]/(kb_eV*ini_temp))
        g[i][0] = energies[i]
    return g

def get_initial_Boltzmann_wn(energies,ini_temp,level_densities):
    g = np.zeros((len(energies),2))
    for i in range(len(energies)):
        rho_index = int(energies[i])
        g[i][1] = level_densities[rho_index]*np.exp(-energies[i]/(kb_wn*ini_temp))
        g[i][0] = energies[i]
    return g

def get_Boltzmann(energies,a,ini_temp,level_densities):
    g = np.zeros(len(energies))
    for i in range(len(energies)):
        rho_index = int(energies[i] // wn_to_eV)
        g[i] = level_densities[rho_index]*np.exp(-energies[i]/(kb_eV*ini_temp))
    return a*g/max(g)

def g_evolution_dt(energies,freqs,k_singles,g_0,dt,no_steps):
    enes_nu = freqs*wn_to_eV
    dynamic_g = np.zeros((no_steps,len(energies)))
    dynamic_g[0] = g_0[:,1]/max(g_0[:,1])
    for j in range(no_steps-1):
        for ek,ene in enumerate(energies):
            w1 = 0
            w2 = 0
            el0 = fn.find_nearest(g_0[:,0],ene)[0]
            for i in range(len(enes_nu)):
                el1 = fn.find_nearest(g_0[:,0],ene + enes_nu[i])[0]
                w1 += np.exp(-k_singles[el0][i]*dt)
                w2 += dynamic_g[j][el1]*(1 - np.exp(-k_singles[el1][i]*dt))
            dynamic_g[j+1][ek] = dynamic_g[j][ek]*w1 + w2
    return dynamic_g