#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# author: MingChao Ji
# email: mingchao.ji@fysik.su.se
# date created: 2019-10-09 16:04:40
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# global functions

import const
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.signal import find_peaks
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import glob
import os.path


# linear function
def line(x, k,p):
    return k * x **(p)


# sine function
def sine(t, amp, w, phase, c):
    return amp * np.sin(w * t + phase) + c


# cosine function
def cosine(t, amp, w, phase, c):
    return amp * np.cos(w * t + phase) + c


# exponential function
def expo(x, a, tau):
    return a*np.exp(-x/tau)


# exponential function for the lifetime fit
def expo_lifetime(x, amp, tau, c):
    return amp * np.exp(-x / tau) + c

def lifetime_La(x,amp,amp1,tau0,tau1):
    # tau2 = 595.67
    return amp*np.exp(-x/tau0) + amp1*np.exp(-x/tau1) 
# power-law function
def power(x, amp, f, h):
    return amp * x ** f + h

def TOB(ene,cte,alpha,l,threshold):
    ene = np.array(ene,dtype='complex')
    k = np.sqrt(2*(ene-(threshold)))
    expression = np.zeros(len(ene))
    for i in range(len(ene)):
        if k[i] > 0:               
            b = np.real((2*np.sqrt(alpha)*k[i])/(l+1/2)**2)
            F1 = sp.hyp2f1(1/2,3/2,2,((1-b)/(1+b)))
            expression[i] += ((l + 1/2)/2)*((1-b)/np.sqrt(1+b))*F1
    return cte*1/(1+np.exp(2*np.pi*np.array(expression)))

def WGMERT(ene,ND0,ND2,alpha,K_22_0,K_22_2,threshold):
    ene = np.array(ene,dtype='complex')
    k = np.sqrt(2*(ene-(threshold)))
    hbar = 1
    omega = ene/hbar
    expression = np.zeros(len(ene))
    angular_m = np.array([0,2])
    kds = np.array([K_22_0,K_22_2])
    nds = np.array([ND0,ND2])
    for nl,l in enumerate(angular_m):
        for i in range(len(ene)):
            if k[i] > 0:
                num = nds[nl]*omega[i]*k[i]**(2*l+1)
                gamma_ff = np.pi*alpha**(3/4)*k[i]**2/3
                gamma_gf = alpha**(-1/4)
                den = (gamma_ff - kds[nl]*gamma_gf)**2
                expression[i] += np.real(num/den)
            else:
                expression[i] += 0
    return np.array(expression)

def WGMERT_au(k,ND0,ND2,alpha,K_22):
    hbar = 1
    # omega = ene/hbar
    expression = np.zeros(len(k))
    angular_m = np.array([2,0])
    nds = np.array([ND0,ND2])
    for nl,l in enumerate(angular_m):
        for i in range(len(k)):
            num = nds[nl]*k[i]**(2*l+1)
            gamma_ff = np.pi*alpha**(3/4)*k[i]**2/3
            gamma_gf = alpha**(-1/4)
            den = (gamma_ff - K_22*gamma_gf)**2
            expression[i] = num/den
    return expression

# power-exp
def pow_exp(x, amp1, f1, tau1):
    return amp1 * x ** f1 * np.exp(-x/tau1)

def pow2_exp(x, amp, f1,tau,amp2,f2):
    return amp * x**(f1) * np.exp(-x/tau) + amp2*x**(f2)

def pow_exp_f(x, amp, tau):
    return amp * x ** -1 * np.exp(-x/tau)


# dual (broken) power-law function
def broken_power(x, amp, f1, xs, f2, h):
    return np.piecewise(x, [x < xs, x >= xs], [lambda x: amp * x ** f1,
                                               lambda x: amp * x ** f2 * xs ** (f1 - f2) + h])



# rate coefficient of MN
def rate_coefficient(x):
    # y = np.piecewise(x, [x <= 1.0, 1.0 < x], [lambda x: 0.25 * x ** -0.5, 0.25])
    y = np.piecewise(x, [x <= 2.0, 2.0 < x], [lambda x: 0.25 * x ** -0.5, 0.175])
    # y = np.piecewise(x, [x <= 3.0, 3.0 < x], [lambda x: 0.25 * x ** -0.5, 0.144])
    return y


# gaussian function
def gaussian(x, amp, ctr, sig):
    return amp * np.exp(-np.power((x - ctr) / sig, 2.) / 2)


def gaussian_multi(x, params):
    y = np.zeros_like(x)
    print(params)
    for i in range(0, len(params), 3):
        amp = params[i]
        ctr = params[i + 1]
        sig = params[i + 2]
        y = y + amp * np.exp(-np.power((x - ctr) / sig, 2.) / 2)
    return y


# a special function to fit the square shape to gaussian-like shape
# def square(x, amp, ctr, hw, tau):
#     return amp/(2 * np.exp(-hw/tau) * np.cosh((x-ctr)/tau) + np.exp(-2 * hw/tau) + 1)

def square(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        amp = params[i]
        ctr = params[i + 1]
        hw = params[i + 2]
        tau = params[i + 3]
        y = y + amp / (2 * np.exp(-hw / tau) * np.cosh((x - ctr) / tau) + np.exp(-2 * hw / tau) + 1)
    return y


    # function used by J. U. Andersen et al. Eur Phys J D 25, 139 (2003) to fit the space_charge_lifetime decay of amino acids.
def spontaneous(x, amp, tau, delta):
    # return amp * (x/tau)**sig / (np.exp(x/tau)-1) + bgd
    return amp * (x/tau) ** delta / (np.exp(x / tau) - 1)

def spontaneous_bg(x, amp, tau, delta, bg):
    # return amp * (x/tau)**sig / (np.exp(x/tau)-1) + bgd
    return amp * (x/tau) ** delta / (np.exp(x / tau) - 1) + bg

# time constant tau vary with time
def spont_new(x, amp, tau0, gamma, delta, bgd):
    tau = tau0 * (1 + x * gamma)
    return amp * (x/tau) ** delta / (np.exp(x / tau) - 1) + bgd

def spont_edu(x, amp, delta, tau0, gamma,bg):
    tau = tau0 * (1 + x * gamma)
    return amp * x**delta * np.exp(-x/tau) +  bg

def spont_edu2(x, amp, amp2, delta, delta2, tau0, gamma,bg):
    tau = tau0 * (1 + x * gamma)
    return amp * x**delta * np.exp(-x/tau) + amp2*x**delta2 + bg

# M. Gatchell et al. Journal of Physics: Conference Series, 488, 012040 (2014) to fit lifetime
# amp: normalization factor, gamma: ion loss rate due to space charge effect, kappa: ion loss rate due to residual gas
def space_charge_lifetime(x, amp, gamma, kappa):
    return amp / ((1 + gamma / kappa) * np.exp(x * kappa) - gamma / kappa)

def recursive_fit(model,x,y,sigma,p0,bounds,mxf,epsilon=1e-9):

    # initialize using a non weighted fit
    p_prev,p_cov = curve_fit(model,x,y,p0=p0)
    dp = 1

    # p_prev = p
    while dp > epsilon :
        p,p_cov = curve_fit(model,x,y,sigma=sigma,p0=p_prev, maxfev = mxf)
        # print(p_cov)
        dp = max(abs(p-p_prev))

        p_prev = p
        # print('ding',dp)

    return p,p_cov

'''
numpy functions

'''

def np_array(lista):
    li0 = lista.to_numpy()
    li = li0.flatten()
    return np.array(li[:])

def normalization(lista):
    lista_0 = np.array(lista)
    lista_f = lista_0/max(lista_0)
    return lista_f

def csv_to_np(file):
    pdDF = pd.read_csv(file,header = None, delimiter = '\t')
    return np.array(pdDF.values)

def csv_to_np_coma(file):
    pdDF = pd.read_csv(file,header = None, delimiter = ',')
    return np.array(pdDF.values)

def csv_to_np2(file):
    pdDF = pd.read_csv(file,header = None, delimiter = '\t')
    zr = pdDF.values
    return zr[:,0], zr[:,1], zr[:,2]

def csv_to_np3(file):
    pdDF = pd.read_csv(file,header = None, delimiter = '\t')
    zr = pdDF.values
    return zr[:,0], zr[:,1], zr[:,2], zr[:,3]

def data_xyerr(file):
    zr = csv_to_np(file)
    d_0 =np.array(zr)
    return d_0[:,0], d_0[:,1], d_0[:,2], d_0[:,3]

def data_2d(file):
    zr = csv_to_np(file)
    d_0 = np.array(zr)
    return d_0[:,0], d_0[:,1]

def data_2d_coma(file):
    zr = csv_to_np_coma(file)
    d_0 = np.array(zr)
    return d_0[:,0], d_0[:,1]

def data_to_np(alist, n1, n2):
    z = np.array(alist)
    return z[n1:n2,0], z[n1:n2,1], z[n1:n2,2]

def bg_norm_pars(file,elem):
    zr = csv_to_np(file)
    d_0 =np.array(zr)
    return d_0[:,0][0], d_0[:,1][elem], d_0[:,2][elem], d_0[:,3][elem], d_0[:,4][elem], d_0[:,5][elem]

def bg_norm_pars_laser(file):
    zr = csv_to_np(file)
    d_0 =np.array(zr)
    return d_0[:,0][0], d_0[:,1], d_0[:,2], d_0[:,3], d_0[:,4]

def load_LPTS_file(file):
    zr = csv_to_np(file)
    d_0 =np.array(zr)
    return d_0[:,0], d_0[:,1], d_0[:,2], d_0[:,3], d_0[:,4]

def load_LPTS_pow_file(file):
    zr = csv_to_np(file)
    d_0 =np.array(zr)
    return d_0[:,0], d_0[:,1]


def ea_weighted(eas,eas_err):
    avg = np.sum(eas*eas_err)/np.sum(eas_err)
    std = np.sqrt((1/len(eas))*np.sum(eas_err*(eas - avg)**2))
    return avg, std

'''
end of numpy functions
'''

# destruction cross section
cell_length = 7.0  # cm
temperature = 300  # K
# [J] = Pa * m^3 = mbar * 10^-2 * cm^6
gc = cell_length / (const.boltzmann_J_K * temperature) / 1.0e4

# x: pressure in gas cell, mbar, cs: cross section, cm^2
def destruct_cs(x, inte, cs):
    return inte * np.exp(-x * cs * gc)


# cal ion speed from kinetic energy, eV
def ke2speed(ke_eV, mass_kg):
    return np.sqrt(2 * ke_eV * const.eV_J / mass_kg)


# cal kinetic energy from ion speed, m/s
def speed2ke(ion_speed, mass_kg):
    return 0.5 * mass_kg * ion_speed * ion_speed / const.eV_J


# cal photon energy from wavelength
def lambda2ev(lambda_nm):
    return const.planck_eVs * const.speed_of_light * 1.0e9 / lambda_nm


# cm-1 to ev
def pcm2ev(freq):
    return const.planck_eVs * const.speed_of_light * 1.0e2 * freq


# cal wavelength, nm from photon energy, eV
def ev2lambda(energy_eV):
    return const.planck_eVs * const.speed_of_light * 1.0e9 / energy_eV


# Kernel Density Estimation with Scipy
def density(grid, data, dens_factor=0.005):
    # data : numpy.array, `n x p` dimensions, representing n points and p variables.
    # grid : numpy.array, Data points at which the desity will be estimated.
    # out : numpy.array, Density estimate. Has `m x 1` dimensions
    kde = gaussian_kde(data.T, dens_factor)
    return kde.evaluate(grid.T)


def count_circ_area(circ_area, img_array):
    # circ_area = [xc, yc, r]
    count = 0
    for i in range(len(img_array[:, 0])):
        if np.sqrt((img_array[i, 0] - circ_area[0]) ** 2 + (img_array[i, 1] - circ_area[1]) ** 2) <= circ_area[2]:
            count += 1
    return count


# function to calc num of counts located in certain area of an img
# circ_area = [xc, yc, r]
# rect_area = [x1, y1, x2, y2] to define the area
# img_array = np.array([X_list, Y_list]).T

# def a function to index the position of a num in another (an) list
def offset(num, _list):
    _list.append(num)
    _list.sort()
    idx = _list.index(num)
    # remove the num to initialize the list
    _list.remove(num)
    return idx

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def norm_factor(cycle_list, energy_list, No_wavelengths,d):
    cycles = []
    for i in range(No_wavelengths):
        cycle, bg, bgstd, bgstep = bg_norm_pars(cycle_list[i+d],elem = -1) 
        cycles.append(cycle)
    c_max = max(cycles)
    cycles = np.array(cycles)
    e_max = max(energy_list)
    e = 6 # specific case for each of these experiments 
    energies = np.array(energy_list[d + e:No_wavelengths+d + e])
    fac = (c_max/cycles)*(e_max/energies)
    return fac

'''
binning functions
'''
def bin_log_norm(z,start):
    z = np.array(z)
    x = z[:,0]
    y = z[:,1]
    err = z[:,2]
    xbin = z[:,3]
    data = []
    for i in range(len(x)):
        fac = start/xbin[i]
        data.append([x[i],y[i]*fac,err[i]*fac])
    return data

def binArray(data_x,data_y,n,method):   
    dim = len(data_x)/10
    if method == 'log':
        sizes = np.logspace(np.log10(1e-5),np.log10(dim),num=n) # 
    if method == 'linear':
        step = int(len(data_x)//n)
        sizes = np.ones(step)*step
    binArray = []
    x_elem = []
    c = 0
    for i in sizes:
        elem, val = find_nearest(data_x, i)
        if not c == elem and i < dim:
            ys = data_y[c:elem+1]
            binArray.append(ys.mean())
            x_elem.append(val)
        c = elem
    return x_elem, binArray

# function to rebin the data with factor by default 2
def rebin(x, y, by=2):
    """Reduce the number of bins in tof spectrum:
    x - time; y - Intensity;
    by - number of columns to merge into a new column"""
    nl = len(x)
    newl = int(nl / by)

    newx = np.linspace(x[0], x[(nl - 1) * by], num=newl)
    newy = np.zeros(newl, dtype=float)

    for i in range(0, by):
        newy += y[0 + i:newl * by + i:by]

    new_xy = np.array([newx, newy]).T
    # output a new array with 1st col newx, 2nd newy
    return new_xy


# calc the hist of a data series
def histo(data, n_bins):
    hist, bin_edge = np.histogram(data, bins=n_bins)
    bin_wd = np.diff(bin_edge)
    x_center = bin_edge[:-1] + bin_wd / 2
    hist_data = np.array([x_center, hist]).T
    return hist_data


# function to introduce equal bin-width on log scale
def logbin(datafile, index_col_x, index_col_y, num_bins):
    bin_start = datafile[0, index_col_x]
    bin_stop = datafile[-1, index_col_x]

    bin_edges = np.logspace(np.log10(bin_start), np.log10(bin_stop), num_bins + 1, base=10)

    # keep the sum of times, which we will average later
    binned_x_sum = np.zeros(num_bins)

    # define an empty array into which the binned data will go with the right number of elements
    binned_y_sum = np.zeros(num_bins)

    # how many things ended up in each bin; avoid divide by zeros if nothing
    # arrives
    binned_n = np.zeros(num_bins) + 1e-9

    n_rows = datafile.shape[0]

    currentbin = 0
    i = 0

    while i < n_rows and currentbin < num_bins:
        x = datafile[i, index_col_x]
        y = datafile[i, index_col_y]

        if x <= bin_edges[currentbin + 1]:
            # x is in the current bin
            binned_y_sum[currentbin] += y
            binned_x_sum[currentbin] += x
            binned_n[currentbin] += 1
            i += 1
        else:
            # make a step to the next bin as x does not belong in this bin
            currentbin += 1

    binned_x = binned_x_sum / binned_n
    binned_y = binned_y_sum / binned_n
    combined = np.array([binned_x, binned_y, binned_n]).T

    nonzero_rows = combined[:, 1] > 0
    combined = combined[nonzero_rows, :]

    return combined

# use the following way to export data,
# new_data = func.logbin(datafile, index_col_x, index_col_y, num_bins)


# function to calculate the sum of period peaks like laser induced delayed peaks.
# Function parameters are given by channel, i.e. real-time / x_bin
def pksum(counts, start, width, period, end, x_bin):
    # calculate num of peaks in the seleted time range
    num_pks = int(round((end - start) / period))

    # create lists to put x_value, peak_left, peak_right, peak_sum.
    x_value_list = []
    y_value_list = []
    peak_left_list = []
    peak_right_list = []

    for n in range(num_pks):
        peak_left = int(round(start + n * period))
        peak_right = int(round(peak_left + width))
        # take the center of peak as x value
        x_value = int((peak_left + peak_right) / 2) * x_bin
        y_value = np.sum(np.array(counts)[peak_left:peak_right + 1])

        peak_left_list.append(peak_left)
        peak_right_list.append(peak_right)
        x_value_list.append(x_value)
        y_value_list.append(y_value)

    combined = np.array([x_value_list, peak_left_list, peak_right_list, y_value_list]).T

    # nonzero_rows = combined[:, 3] > 0
    # combined = combined[nonzero_rows, :]
    return combined
# use the following way to export data,
# peak_sum = func.pksum(ch_count, calc_start, gate_width, int(rev_time_s)+1, calc_end, 1.0e-6)


''' Questionarie functions 
'''

def filesQ(lista):
    dim = len(lista)
    print ('This is the initial list: ', lista)
    r = input('I have counted ' + str(dim) + ' elements. Do you want them all or customized (0 all 1 singles or 2 sublist)? ')
    if r =='1':
        how_many = input('how many do you want me to handle? ')
        new_list = list()
        for i in range(int(how_many)):
            new_list.append(input('write file ' + str(i+1) + ' : '))
    elif r =='2':
        slics = input('slice it (give: ini-end-jump): ')
        s = slics.split('-')
        new_list = lista[int(s[0]):int(s[1]):int(s[2])]
    else:
        new_list = lista
    print ('so i am taking this list: ' , new_list)
    return np.array(new_list)

'''
peak_function should be checked every time for height and distance parameters
'''
def peak_sum(counts, start, width, period, end):
    # calculate num of peaks in the selected time range
    num_pks = int(round((end - start) / period))
    # create lists to put x_value, peak_left, peak_right, peak_sum.
    x_value_list = []
    peak_right_list = []
    peak_centers = []
    for n in range(num_pks):
        peak_left = start + n * period
        peak_right = peak_left + width
        peak_center = peak_left + width/2
        # take the center of peak as x value
        x_value = (peak_left + peak_right) / 2
        # y_value = np.sum(np.array(counts)[peak_left:peak_right + 1])
        peak_right_list.append(peak_right)
        peak_centers.append(peak_center)
        x_value_list.append(x_value)
    combined = np.array([x_value_list, peak_right_list]).T
    return combined

def period_function(x,y,min_diff,max_diff,maxy,dist,n1,n2):
    peaks = find_peaks(y,height = max(y)/3,prominence = maxy,distance=dist)
    laser_peaks= peaks[0][n1:]
    periods = []
    for i in range(len(laser_peaks)-1):
        diff = abs(x[laser_peaks[i+1]]-x[laser_peaks[i]])
        if diff < max_diff:
            periods.append(diff)
    return np.average(periods), laser_peaks

# this function will find the frequency of the laser version 1
# def find_laserPeak(datax,datay,p,m,n,period):
#     t0 = datax[p]
#     times = []
#     periods = []
#     elems = []
#     for i in range(n):
#         fsy = datay[p-m:p+m]
#         fsx = datax[p-m:p+m]
#         fp = max(fsy)
#         elemy, valy = find_nearest(fsy,fp)
#         times.append(fsx[elemy])
#         elems.append(find_nearest(datax,times[i])[0]) 
#         tn = t0 + (i+1)*period
#         p, valx = find_nearest(datax,tn)
#     for t in range(len(times)-1):
#         diff = abs(times[t+1] - times[t])
#         if (1 - (1/10))*period < diff < (1 + (1/10))*period:
#             periods.append(diff)
#     return np.average(periods), elems


# version 2
def find_laserPeak(datax,datay,p,m,bd,period):
    t0 = datax[p]
    times = []
    periods = []
    elems = []
    valx = 0
    tn = t0
    while valx < bd - 0.1:
        if p-m > 0:
            fsy = datay[p-m:p+m]
            fsx = datax[p-m:p+m]
        else:
            fsy = datay[p:p+m]
            fsx = datax[p:p+m]
        fp = max(fsy)
        elemy, valy = find_nearest(fsy,fp)
        times.append(fsx[elemy])
        elems.append(find_nearest(datax,fsx[elemy])[0]) 
        tn = tn + period
        p, valx = find_nearest(datax,tn)
    for t in range(len(times)-1):
        diff = abs(times[t+1] - times[t])
        if (1 - (1/10))*period < diff < (1 + (1/10))*period:
            periods.append(diff)
    return np.average(periods), elems

def findValleys(xdata,ydata,pi,rev,n):
    valleys = []
    ti = xdata[pi]
    tf = ti + rev + 1e-6
    span = 6
    for i in range(n):
        # print ('pi ',pi)
        pf,val2 = find_nearest(xdata,tf)
        y = ydata[pi:pf]
        x = xdata[pi:pf]
        ymax = min(y)
        elemy, valy = find_nearest(y,ymax)
        time = x[elemy]
        elemx, valx = find_nearest(xdata,time)
        # if  elemx > pi:
        for i in range(span):
            valleys.append(elemx - span//2 + i)
            # print (elemy,elemx)
        # else:
            # print ('else ',elemy, elemx)
        pi = pf
        tf += rev
    return valleys

def find_Valleys_1(xdata,ydata,pi):
    valleys = []
    ti = xdata[pi]
    tf = ti + rev + 1e-6
    span = 6
    for i in range(n):
        # print ('pi ',pi)
        pf,val2 = find_nearest(xdata,tf)
        y = ydata[pi:pf]
        x = xdata[pi:pf]
        ymax = min(y)
        elemy, valy = find_nearest(y,ymax)
        time = x[elemy]
        elemx, valx = find_nearest(xdata,time)
        # if  elemx > pi:
        for i in range(span):
            valleys.append(elemx - span//2 + i)
            # print (elemy,elemx)
        # else:
            # print ('else ',elemy, elemx)
        pi = pf
        tf += rev
    return valleys

def backSubtraction(datax,datay,bd,direction,Np):
    elem_bd, x_bd = find_nearest(datax,bd)
    skip = 0
    if direction == 0:
        y = datay[elem_bd-Np-skip:elem_bd-skip]
    elif direction == 1:
        y = datay[elem_bd+skip:elem_bd+Np+skip]
    yb = np.mean(y)
    ystd = np.sqrt(abs(np.sum(y)))*1/(len(y))
    # ystd = np.std(y)
    return yb, ystd, elem_bd

def backTail(datax,datay,signal_f,signal_i,laser_interval,beam_dump):
    tailsx = []
    tailsy = []
    tf = round(signal_f,6) - 1e-5
    ti = round(signal_i,6)
    h = 0
    while tf < beam_dump:
        elem_i, val_i = find_nearest(datax,ti)
        elem_f, val_f = find_nearest(datax,tf)
        ls = datax[elem_i:elem_f]
        for k,val in enumerate(ls):
            tailsy.append(datay[elem_i + k])
            tailsx.append(val)
        tf += laser_interval
        ti += laser_interval
    return np.array(tailsx), np.array(tailsy)

def getPreviousPoints(xdata,ydata,err,x,y,e,ni,pi):
    x_prev = xdata[ni:pi]
    y_prev = ydata[ni:pi]
    err_prev = err[ni:pi]
    x2 = np.array(x_prev.tolist() + x.flatten().tolist())
    y2 = np.array(y_prev.tolist() + y.flatten().tolist())
    err2 = np.array(err_prev.tolist() + e.flatten().tolist()) 
    return x2, y2, err2

def load_metadata(file,sheet,columns):
    pd_data = pd.read_excel(file,sheet_name = sheet)
    sets = list()
    for col in columns:
        df_s = pd.DataFrame(pd_data,columns = [col])
        sets.append(np.array(df_s).flatten())
    return sets

def ready_data(path,method,exp,chs):
    file_list = list()
    ps_cycle_file_list = list()
    path_test = r'C:\Users\Admin\Nextcloud\Documents\Jose PhD project\2022v13_Azulene_Naphthalene\SptDecay/'
    path_ps_test = r'C:\Users\Admin\Nextcloud\Documents\Jose PhD project\2022v13_Azulene_Naphthalene\SptDecay\pars/'
    # print (path + method + exp)
    if os.path.exists(path):
        file_list = glob.glob(path_test + method + '*.csv')
        # file_list = glob.glob(path + '*.csv')
    else:
        print ('path does not exist')
    # print (path + method + exp)
    print ('In ready data! \n')
    print (path + 'pars/')
    print ('')
    # ps_cycle_file_list = glob.glob(path + 'pars/' + method + exp + '*.csv')
    ps_cycle_file_list = glob.glob(path_ps_test + method + '*.csv')
    
    ps_cycles = ps_cycle_file_list[:]
    jumps = len(chs)
    all_data = []
    for i in range(jumps):
        all_data.append(file_list[i::jumps])
    return all_data, ps_cycles

def plotting(x,y,err,xmin,xmax,ymin,ymax,k,d,labels,col,mode,alpha):
    matplotlib.rcParams.update({'font.size': 17})
    plt.figure(k, figsize = [12,7])
    if mode == 1:
        plt.errorbar(x,y,yerr=err,capthick=1.5,ls = 'none',capsize=3,elinewidth=1.5, alpha = 0.5)
        plt.plot(x,y, alpha = 0.6, label = labels,color = col)
    if mode == 0:
        plt.scatter(x,y,s = 25,alpha = 0.3,label = labels)
        plt.errorbar(x,y,yerr=err,capthick=1.5,ls = 'none',capsize=3,elinewidth=1.5, alpha = 0.3)

    plt.yscale('log')
    plt.ylim([ymin,ymax])
    plt.xscale('log')
    plt.xlim([xmin,xmax])
    plt.legend(loc='best')
    plt.xlabel('Storage time [s]')
    plt.ylabel('Neutral yield [arb. units]')
