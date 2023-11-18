import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
import pandas as pd
from numba import jit

# matplotlib.rcParams.update({'font.size': 15})
# plt.rcParams["figure.figsize"] = (8,5)


def stepfunc(x,limit):
    """
    Step function. Equal to 0 if x < limit, else 1.
    """
    return 0.5*(np.sign(x-limit)+1)

def wigner_law(enes,ea,A,C):
    enes = np.array(enes,dtype='complex')
    express = np.sqrt(enes - ea)*stepfunc(enes, ea)
    return C + A*np.real(express)

def wigner_law_l(enes,ea,A,C,l):
    enes = np.array(enes,dtype='complex')
    express = (enes - ea)**(l+1/2)*stepfunc(enes, ea)
    return C + A*np.real(express)

def wigner_law_2_th(enes,ea,A1,A2,d1,C):
    enes = np.array(enes,dtype='complex')
    eas = [0,d1]
    fs = np.array([A1,A2])
    express = 0
    for k in range(len(eas)):
        express += fs[k]*np.sqrt(enes - (ea + eas[k]))*stepfunc(enes, (ea + eas[k]))
    return C + np.real(express)

def wigner_law_3_th(enes,A1,A2,A3,ea,d1,d2,C):
    enes = np.array(enes,dtype='complex')
    eas = [0,d1,d2]
    fs = np.array([A1,A2,A3])
    express = 0
    for k in range(len(eas)):
        express += fs[k]*np.sqrt(enes - (ea + eas[k]))*stepfunc(enes, (ea + eas[k]))
    return C + np.real(express)

def wigner_law_4_th(enes,A1,A2,A3,A4,d0,ea,d1,d2,C):
    enes = np.array(enes,dtype='complex')
    eas = [0,d1,d2,d0]
    fs = np.array([A1,A2,A3,A4])
    express = 0
    for k in range(len(eas)):
        express += fs[k]*np.sqrt(enes - (ea + eas[k]))*stepfunc(enes, (ea + eas[k]))
    return C + np.real(express)

def wigner_law_4_th_2(enes,A1,A2,A3,A4,d0,ea,d1,d2,C):
    enes = np.array(enes,dtype='complex')
    eas = [0,d1,d2,d0]
    fs = np.array([A1,A2,A3,A4])
    ls = np.array([0,0,0,0])
    express = 0
    for k in range(len(eas)):
        express += fs[k]*(enes - (ea + eas[k]))**(ls[k]+1/2)*stepfunc(enes, (ea + eas[k]))
    return C + np.real(express)

def wigner_hfs(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    # freqs relative to the F=2 hyperfine level, in MHz. Energies in eV
    fs = np.array([5,4,3,2])
    h_eV = 4.135667696e-15
    f3 = 151.21
    f4 = f3 + 201.2 
    f5 = f4 + 251
    Ehfs = np.array([f5,f4,f3,0])*(1e6)*h_eV
    express = 0
    for k,f in enumerate(fs):
        express += (2*f +1)*np.sqrt(E-(threshold+Ehfs[k]*1e3))*stepfunc(E,threshold + Ehfs[k]*1e3)
    return C + A*np.real(express)

def wigner_hfs_abs(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    # freqs relative to the F=2 hyperfine level, in MHz. Energies in eV
    fs = np.array([5,4,3,2])
    h_eV = 4.135667696e-15
    f3 = 151.21
    f4 = f3 + 201.2 
    f5 = f4 + 251
    Ehfs = np.array([f5,f4,f3,0])*(1e6)*h_eV
    express = 0
    for k,f in enumerate(fs):
        express += (2*f +1)*np.sqrt(E-(threshold+Ehfs[k]*1e3) + np.abs(E-(threshold+Ehfs[k]*1e3)))
    return C + A*np.real(express)

def wigner_c60_2(E,threshold,A1,A2,C):
    E = np.array(E,dtype='complex')
    fs = np.array([A1,A2])
    h_eV = 4.135667696e-15
    f3 = abs(2.6635 - 2.78)
    Ehfs = np.array([f3,0])*(1e6)*h_eV
    express = 0
    for k,f in enumerate(fs):
        express += (2*f +1)*np.sqrt(E-(threshold+Ehfs[k]*1e3))*stepfunc(E,threshold + Ehfs[k]*1e3)
    return C + np.real(express)

def wigner_c60(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    ph = np.sqrt(E-(threshold))*stepfunc(E,threshold)
    return C + A*np.real(ph)


def wigner_c60_p(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    ph = (E-(threshold))**(3/2)*stepfunc(E,threshold)
    return C + A*np.real(ph)

def wigner_c60_malpha(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    k = np.sqrt(2*(E-(threshold)))
    me = 1
    a0 = 1
    alpha = 76.5/(0.529**3)
    # alpha = 0
    l = 0
    ph = list()
    for i in range(len(k)):
        if k[i]*a0 > 0:
            # ph.append((E[i]-(threshold))**(l+1/2)*(1 - (4*alpha*2*(E[i]-(threshold))*np.log(k[i]*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold))
            ph.append((E[i]-(threshold))**(l+1/2)*(1 + (4*alpha*2*(E[i]-(threshold))*np.log(k[i]*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold))

        else:   
            ph.append((E[i]-(threshold))**(l+1/2)*(1)*stepfunc(E[i],threshold))
    return C + A*np.real(ph)

def wigner_c60_m(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    k = np.sqrt(2*(E-(threshold)))
    me = 1
    a0 = 1
    # alpha = 76.5/(0.529**3)
    alpha = 0
    l = 0
    ph = list()
    for i in range(len(k)):
        if k[i]*a0 > 0:
            # ph.append((E[i]-(threshold))**(l+1/2)*(1 - (4*alpha*2*(E[i]-(threshold))*np.log(k[i]*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold))
            ph.append((E[i]-(threshold))**(l+1/2)*(1 + (4*alpha*2*(E[i]-(threshold))*np.log(k[i]*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold))

        else:   
            ph.append((E[i]-(threshold))**(l+1/2)*(1)*stepfunc(E[i],threshold))
    return C + A*np.real(ph)

def wigner_c60_m_alpha(E,threshold,A,C,alpha,l,d1):
    # E = np.array(E,dtype='complex')
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    # alpha = 76.5/(0.529177210903**3)
    # l = 0
    for i in range(len(E)):
        k1 = E[i]-threshold
        flat = stepfunc(E[i],threshold)
        if k1*a0 > 1e-6:
            k = np.sqrt(2*k1)
            den = a0*(2*l+3)*(2*l+1)*(2*l-1)
            num = 4*alpha*2*k1*np.log(k*a0)
            ph[i] = k1**(l+1/2)*(1 + num/den + d1*k1**2)
    return C + A*ph
    
def wigner_c60_alpha_quadrupole(E,threshold,A,C,alpha,l,d1,Q):
    # E = np.array(E,dtype='complex')
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    # alpha = 76.5/(0.529177210903**3)
    # l = 0
    j = 3/2
    for i in range(len(E)):
        k1 = E[i]-threshold
        if k1*a0 > 1e-7:
            k = np.sqrt(2*k1)
            den = a0*(2*l+3)*(2*l+1)*(2*l-1)
            num = k1*np.log(k*a0)
            qterm = (19/20)*((j+1)*(2*j+3)/(5*j*(2*j-1)))*Q - 2*alpha
            ph[i] = k1**(l+1/2)*(1 + qterm*num + d1*k1**2)
    return C + A*ph

def wigner_c60_farley(E,threshold,c2,c1):
    r0 = 1
    hbar = 1
    ph = np.zeros(len(E))
    for i in range(len(E)):
        k1 = E[i]-threshold
        if k1 > 1e-7:
            k = np.sqrt(2*k1)
            gamma = np.sqrt(2*threshold)
            x = gamma*r0
            b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
            b3 = -2*x*(3*x + 6)**(-1)*(x**4 + 5*x**3 + 15*x**2 + 30*x + 30)
            ph[i] = ((E[i])/gamma**4)*(b1*(k/gamma) + b3*(k/gamma)**3)
    return c1 + c2*ph

def farley(ene,c2,r0,threshold):
    hbar = 1
    ph = np.zeros(len(ene))
    for i in range(len(ene)):
        k1 = ene[i]-threshold
        if k1 > 0:
            k = np.sqrt(2*k1)
            gamma = np.sqrt(2*threshold)
            x = gamma*r0
            b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
            b3 = -2*x*(3*x + 6)**(-1)*(x**4 + 5*x**3 + 15*x**2 + 30*x + 30)
            term1 = c2*(ene[i]/gamma**4)*(b1*(k/gamma))
            term2 = (ene[i]/gamma**4)*(b3*(k/gamma)**3)
            ph[i] = term1 + term2
    return ph

def farley0(ene,c2,r0,threshold):
    ph = np.zeros(len(ene))
    for i in range(len(ene)):
        k1 = ene[i]-threshold
        if k1 > 0:
            k = np.sqrt(2*k1)
            gamma = np.sqrt(2*threshold)
            x = gamma*r0
            b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
            ph[i] = ((ene[i])/gamma**4)*(b1*(k/gamma))
    return c2*ph

def wigner_alpha(ene,D0,D2,alpha,threshold):
    ene = np.array(ene,dtype='complex')
    k = np.real(np.sqrt(2*(ene-(threshold))))
    # me = 1
    a0 = 1
    ph = np.zeros(len(ene))
    # alpha = 76.5/(0.529177210903**3)
    angular_m = np.array([0,2])
    ds = np.array([D0,D2])
    for ln, l in enumerate(angular_m):
        for i in range(len(ene)):
            if ene[i]-threshold > 0:
                den = a0*(2*l+3)*(2*l+1)*(2*l - 1)
                num = 4*alpha*k[i]**2*np.log(k[i]*a0)
                # ph[i] = k1**(l+1/2)*(1 - num/den + d1*k1**2)
                ph[i] += ds[ln]*(k[i]**(2*l+1)*(1 - num/den))
            else:
                ph[i] = 0
    return ph

def Brink(x):
    return x*(1 + 589.5*x**2*np.log(x))

def malley(ene,alpha,threshold):
    # me = 1
    a0 = 1
    l = 0
    ph = np.zeros(len(ene))
    for i in range(len(ene)):
        if ene[i]-threshold > 0:
            k = (2*(ene[i]-threshold))**(l+1/2)
            den = a0*(2*l+3)*(2*l+1)*(2*l - 1)
            num = 4*alpha*k**2*np.log(k*a0)
            ph[i] += k*(1 - num/den)
        else:
            ph[i] = 0
    return ph

def malley_2(ene,alpha,threshold):
    # me = 1
    a0 = 1
    angular_momentum = np.array([0,2])
    ph = np.zeros(len(ene))
    for l in angular_momentum:
        for i in range(len(ene)):
            if ene[i]-threshold > 0:
                k = (2*(ene[i]-threshold))**(l+1/2)
                den = a0*(2*l+3)*(2*l+1)*(2*l - 1)
                num = 4*alpha*k**2*np.log(k*a0)
                ph[i] += k*(1 - num/den)
            else:
                ph[i] = 0
    return ph


def wigner_alpha_quadrupole(E,A,C,alpha,l,Q):
    # E = np.array(E,dtype='complex')
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    # alpha = 76.5/(0.529177210903**3)
    # l = 0
    j = 3/2
    for i in range(len(E)):
        k1 = E[i]
        if k1*a0 > 1e-8:
            k = np.sqrt(2*k1)
            den = a0*(2*l+3)*(2*l+1)*(2*l-1)
            num = k1*np.log(k*a0)
            qterm = (19/20)*((j+1)*(2*j+3)/(5*j*(2*j-1)))*Q - 2*alpha
            # ph[i] = k1**(l+1/2)*(1 + qterm*num + d1*k1**2)
            ph[i] = k1**(l+1/2)*(1 + qterm*num)
    return C + A*ph

def wigner_farley(E,threshold):
    r0 = 1
    hbar = 1
    ph = np.zeros(len(E))
    for i in range(len(E)):
        k1 = E[i]
        if k1 > 1e-8:
            k = np.sqrt(2*k1)
            gamma = np.sqrt(2*threshold)
            x = gamma*r0
            b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
            b3 = -2*x*(3*x + 6)**(-1)*(x**4 + 5*x**3 + 15*x**2 + 30*x + 30)
            ph[i] = ((E[i]+threshold)/gamma**4)*(b1*(k/gamma) + b3*(k/gamma)**3)
    return ph

def wigner_farley_0(E,threshold):
    r0 = 1
    hbar = 1
    ph = np.zeros(len(E))
    for i in range(len(E)):
        k1 = E[i]
        if k1 > 1e-8:
            k = np.sqrt(2*k1)
            gamma = np.sqrt(2*threshold)
            x = gamma*r0
            b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
            b3 = -2*x*(3*x + 6)**(-1)*(x**4 + 5*x**3 + 15*x**2 + 30*x + 30)
            ph[i] = ((E[i]+threshold)/gamma**4)*(b1*(k/gamma))
    return ph

# @jit(nopython=True)
def wigner_c60_m_alpha_hfs(E,threshold,A,C,d1):
    # E = np.array(E,dtype='complex')
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    # alpha = 76.5/(0.529177210903**3)
    alpha = 1655
    l = 0
    fs = np.array([5,4,3,2])
    h_eV = 4.135667696e-15
    f3 = 151.21
    f4 = f3 + 201.2 
    f5 = f4 + 251
    Ehfs = np.array([f5,f4,f3,0])*(1e6)*h_eV
    for kf,f in enumerate(fs):
        for i in range(len(E)):
            k1 = E[i]-(threshold+Ehfs[kf]*1e3)
            flat = stepfunc(E[i],threshold+ Ehfs[kf]*1e3)
            if k1*a0 > 1e-7:
                k = np.sqrt(2*k1)
                ph[i] += (2*f +1)*k1**(l+1/2)*(1 - (4*alpha*2*k1*np.log(k*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1))+d1*k1)*flat    
            # else:   
                # ph[i] += (2*f +1)*k1**(l+1/2)*(1)*flat
    return C + A*ph

def wigner_c60_Qm_alpha_hfs(E,threshold,A,C,Q,d1):
    # E = np.array(E,dtype='complex')
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    # alpha = 76.5/(0.529177210903**3)
    alpha = 1655
    l = 0
    j=3/2
    fs = np.array([5,4,3,2])
    h_eV = 4.135667696e-15
    f3 = 151.21
    f4 = f3 + 201.2 
    f5 = f4 + 251
    Ehfs = np.array([f5,f4,f3,0])*(1e6)*h_eV
    for kf,f in enumerate(fs):
        for i in range(len(E)):
            k1 = E[i]-(threshold+Ehfs[kf]*1e3)
            flat = stepfunc(E[i],threshold+ Ehfs[kf]*1e3)
            if k1*a0 > 1e-7:
                k = np.sqrt(2*k1)
                den = a0*(2*l+3)*(2*l+1)*(2*l-1)
                num = k1*np.log(k*a0)
                qterm = (19/20)*((j+1)*(2*j+3)/(5*j*(2*j-1)))*Q - 2*alpha
                ph[i] += (2*f +1)*k1**(l+1/2)*(1 + qterm*num + d1*k1)*flat
    return C + A*ph


def wigner_c60_Farley_hfs(E,threshold,c1,c2):
    # E = np.array(E,dtype='complex')
    r0 = 1
    me = 1
    a0 = 1
    ph = np.zeros(len(E))
    fs = np.array([5,4,3,2])
    h_eV = 4.135667696e-15
    f3 = 151.21
    f4 = f3 + 201.2 
    f5 = f4 + 251
    Ehfs = np.array([f5,f4,f3,0])*(1e6)*h_eV
    for kf,f in enumerate(fs):
        for i in range(len(E)):
            k1 = E[i]-(threshold+Ehfs[kf]*1e3)
            flat = stepfunc(E[i],threshold+ Ehfs[kf]*1e3)
            if k1*a0 > 1e-7:
                k = np.sqrt(2*k1)
                gamma = np.sqrt(2*threshold)
                x = gamma*r0
                b1 = 2*x*(x**2 + 3*x + 3)**2/(x+2)
                b3 = -2*x*(3*x + 6)**(-1)*(x**4 + 5*x**3 + 15*x**2 + 30*x + 30)
                ph[i] += (2*f +1)*((E[i])/gamma**4)*(b1*(k/gamma) + b3*(k/gamma)**3)*flat
    return c2 + c1*ph

def wigner_c60_m_2alpha(E,threshold,A1,A2,C):
    E = np.array(E,dtype='complex')
    fs = np.array([A1,A2])
    me = 1
    a0 = 1
    eV_to_hartree = 0.0367493
    t2 = np.abs(2.6835 - 2.7494)*eV_to_hartree
    Eths = np.array([t2,0])
    alpha = 76.5/(0.529**3)
    # alpha = 0
    l = 0
    ph = list()
    for i in range(len(E)):
        express = 0
        for kl,f in enumerate(fs):
            ek = E[i]-(threshold+Eths[kl])
            if ek*a0 > 0:
                express += f*(E[i]-(threshold+Eths[kl]))**(l+1/2)*(1 + (4*alpha*2*(E[i]-(threshold+Eths[kl]))*np.log(np.sqrt(2*(E[i]-(threshold+Eths[kl])))*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold+Eths[kl])
            else:   
                express += f*(E[i]-(threshold+Eths[kl]))**(l+1/2)*(1)*stepfunc(E[i],threshold+Eths[kl])
        ph.append(express)
    return C + np.real(ph)

def wigner_c60_m_2(E,threshold,A1,A2,C):
    E = np.array(E,dtype='complex')
    fs = np.array([A1,A2])
    me = 1
    a0 = 1
    eV_to_hartree = 0.0367493
    t2 = np.abs(2.6835 - 2.7494)*eV_to_hartree
    Eths = np.array([t2,0])
    # alpha = 76.5/(0.529**3)
    alpha = 0
    l = 0
    ph = list()
    for i in range(len(E)):
        express = 0
        for kl,f in enumerate(fs):
            ek = E[i]-(threshold+Eths[kl])
            if ek*a0 > 0:
                express += f*(E[i]-(threshold+Eths[kl]))**(l+1/2)*(1 + (4*alpha*2*(E[i]-(threshold+Eths[kl]))*np.log(np.sqrt(2*(E[i]-(threshold+Eths[kl])))*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))*stepfunc(E[i],threshold+Eths[kl])
            else:   
                express += f*(E[i]-(threshold+Eths[kl]))**(l+1/2)*(1)*stepfunc(E[i],threshold+Eths[kl])
        ph.append(express)
    return C + np.real(ph)

def wigner_c60_m_abs(E,threshold,A,C):
    E = np.array(E,dtype='complex')
    k = np.sqrt(2*(E-(threshold)) + np.abs(E-(threshold)))
    me = 1
    a0 = 1
    # alpha = 76.5/(0.529**3)
    alpha = 0
    l = 0
    ph = (E -(threshold) + np.abs(E-(threshold)))**(l+1/2)*(1 + (4*alpha*2*(E-(threshold))*np.log(k*a0))/(a0*(2*l+3)*(2*l+1)*(2*l-1)))
    return C + A*np.real(ph)

def pol(k):
    me = 1
    a0 = 1
    alpha = 76.5/(0.529**3)
    l=0
    ph = (k)**(2*l+1)*(4*alpha*k**2)*np.log(k*a0)/(a0*(2*l+3)*(2*l+1)*(2*l-1))

    return ph

def gaussian(x,mu,sigma):
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma/sigma)

def gaussian2(x,mu):
    sigma =  1e-6
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma/sigma)


def point_convolution(x,samplerange,N_samples,threshold,A,C,sigma,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_hfs(t,threshold,A,C)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]


def point_convolution_simple(x,samplerange,N_samples,threshold,A,C,sigma,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_law(t,threshold,A,C)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]


def point_convolution_simple_l(x,samplerange,N_samples,threshold,A,C,l,sigma,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_law_l(t,threshold,A,C,l)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

''' with sigma fixed'''
def point_convolution2(x,samplerange,N_samples,threshold,A,C,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_hfs(t,threshold,A,C)
    g = lambda t : gaussian2(t,0)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

def point_convolution_c60(x,samplerange,N_samples,threshold,A,C,sigma,d1,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_c60_m_alpha_hfs(t,threshold,A,C,d1)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

def point_convolutionQ_c60(x,samplerange,N_samples,threshold,A,C,Q,sigma,d1,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_c60_Qm_alpha_hfs(t,threshold,A,C,Q,d1)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

def point_convolutionFarley_c60(x,samplerange,N_samples,threshold,A,C,sigma,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_c60_Farley_hfs(t,threshold,A,C)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

def point_convolution_simple_hfs(x,samplerange,N_samples,threshold,A,C,sigma,fft=False) :
    """
    Calculate the convolution between a Wigner s-threshold (l+1/2) and
    a Gaussian distribution at x.
    
    Convolution function has been reimplemented. It seems like the old one
    placed the threshold incorrectly, even though the resulting functions are
    identical. Exactly why is still unknown. I am guessing it was related to
    centering the Gaussian on the threshold. This implementation only considers
    a distribution centered at 0 which is sufficient.

    Arguments:
    x - point where the convolution is calculated
    samplerange - sample range for the convolved functions. For fitting this should
        extend outside the data range, setting it to the full scan width seems
        to work well. As long as no edge effects are visible it should be fine.
    N_samples - number of sample points on the interval [x-range,x+range] to sample
        the convolved functions. Must be an ODD number so that x is centered in the
        sampling interval. Seems to work ok in the 2000-10000 range but keep
        this in mind.
    threshold - see function wigner
    A - see function wigner
    C - see function wigner
    sigma - see function gaussian
    fft - If True, use FFT convolution from scipy.signal, else use numpy convolve.
        Have not really found any difference in performance for current applications
        yet.

    Returns:
    c - convolution calculated at x
    """

    assert N_samples%2 == 1, "N_samples must be odd!"

    # Calculate sample points. Gaussian is centered around 0
    samples_gaussian = np.linspace(-samplerange,samplerange,N_samples)

    # Wigner centered around x
    samples = samples_gaussian + x 
    
    # Step size is used to normalize
    ds = samples[1] - samples[0]
    w = lambda t : wigner_hfs(t,threshold,A,C)
    g = lambda t : gaussian(t,0,sigma)

    # I chose to use the "valid" mode for the convolution, which means
    # that only points with complete overlap (same samples from
    # both convolved functions) is returned. For the present case, the
    # number of samples of both functions are the same so they only overlap
    # completely in one point, so we only get 1 value back from convolve,
    # which is the convolution at x.
    if fft :
        c = convolve(w(samples),g(samples_gaussian),mode='valid') * ds
    else :
        c = np.convolve(w(samples),g(samples_gaussian),mode='valid') * ds

    return c[0]

def fit_point_convolution(x,y,y_delta):
    """
    Small example of a fit. Recursive fitting not implemented yet.
    """
    sigma_fix = 1e-6
    N_samples = 3001
    start = x[0] 
    stop = x[-1]
    scan_width = stop-start

    # generic start guesses
    p0 = (
        start + scan_width * .5,
        y[-1],
        y[0],
        scan_width * .1
    )


    # Fit with free parameters
    fit = lambda x,f_th,A,C,sigma: point_convolution(x,scan_width,N_samples,f_th,A,C,sigma)
    # It is important to vectorize the fit function, since curve_fit needs to be able to pass
    # arrays as input. This way x in point_convolution will be treated as a single number always.    
    fit_vectorized = np.vectorize(fit)
    p,p_cov = curve_fit(fit_vectorized,x,y,sigma=y_delta,p0=p0)
    print('fit free\n',p,p_cov) 

    # Fit with fixed sigma
    p0_fixed_sigma = p0[0:3]
    fit_fixed_sigma = lambda x,f_th,A,C: point_convolution(x,scan_width,N_samples,f_th,A,C,sigma_fix)
    # It is important to vectorize the fit function, since curve_fit needs to be able to pass
    # arrays as input. This way x in point_convolution will be treated as a single number always.    
    fit_fixed_sigma_vectorized = np.vectorize(fit_fixed_sigma)
    p_fixed_sigma,p_fixed_sigma_cov = curve_fit(fit_fixed_sigma_vectorized,
    x,y,sigma=y_delta,p0=p0_fixed_sigma)
    print('fit fixed sigma\n',p_fixed_sigma,p_fixed_sigma_cov)
    
    # Fit wigner only
    p_wigner,p_wigner_cov = curve_fit(wigner_hfs,
    x,y,sigma=y_delta,p0=p0_fixed_sigma)
    print('fit wigner\n',p_wigner,p_wigner_cov)

    # just some plotting to show the results
    x_fit = np.linspace(start,stop,N_samples)
    y_fit = fit_vectorized(x_fit,*p)
    y_fit_fixed_sigma = fit_fixed_sigma_vectorized(x_fit,*p_fixed_sigma)
    y_fit_wigner = wigner_hfs(x_fit,*p_wigner)


    plt.plot(x_fit,y_fit,color='C0',label='$\sigma$ free')
    plt.plot(x_fit,y_fit_fixed_sigma,color='C1',label=f'$\sigma=${sigma_fix}')
    plt.plot(x_fit,y_fit_wigner,color='C2',linestyle='--',label='Wigner')

    plt.axvline(p[0],color='C0',linestyle='--')
    plt.axvline(p_fixed_sigma[0],color='C1',linestyle='-')
    plt.axvline(p_wigner[0],color='C2',linestyle='--')
    
    plt.errorbar(x,y,yerr=y_delta,color='k',linestyle=' ',marker='.',capsize=3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Events')

    plt.legend()
    # plt.savefig('./figures/new_convolution.pdf',bbox_inches='tight')
    # plt.savefig('./figures/new_convolution.png',bbox_inches='tight',dpi=250)
    

    plt.show()


def test_fit() :
    # input for Moas big data file
    datapath = './data/binned_data.csv'
    file_ind = 76
    cycle_ind = 5

    data = pd.read_csv(datapath)
    data = data.astype({'cycle_index':int,'file_index':int})

    dataset = data[data['file_index'] == file_ind]
    scanset = dataset[dataset['cycle_index']==cycle_ind]

    x = np.array(scanset['freq'])
    y = np.array(scanset['sig'])
    y_delta =  np.sqrt(y)

    fit_point_convolution(x,y,y_delta)

def main() : 
    test_fit()

if __name__ == '__main__' :
    main()