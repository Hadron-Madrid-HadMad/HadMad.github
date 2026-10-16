"""G2 wave Global Fits"""
# Parametrization and parameters for the three Global Fits given in Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c

# UNDER 0.85 GeV

mpi=0.13957
s0=1.65**2
sh=2.4**2


"""Global Fit I"""
# b0,b1,b2,Bh2,Bh3,Delta
par_1=-4.57e6,-62.15e6,-77.05e6,188e6,86e6,0.329
err_par_1=0.09e6,0.13e6,0.19e6,10e6,21e6,0.023

"""Global Fit II"""
# b0,b1,b2,Bh2,Bh3,Delta
par_2=-4.56e6,-62.18e6,-76.98e6,187e6,110e6,0.340
err_par_2=0.09e6,0.13e6,0.19e6,10e6,21e6,0.023

"""Global Fit III"""
# b0,b1,b2,Bh2,Bh3,Delta
par_3=-4.57e6,-62.11e6,-77.00e6,194e6,120e6,0.319
err_par_3=0.09e6,0.13e6,0.19e6,10e6,21e6,0.023

# We define the phase-shift, which has two conformal pieces

def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))

def q(s,m):
    return c.sqrt((s/4)-m**2)

def v(s):
    return (c.sqrt(s)-c.sqrt(s0-s))/(c.sqrt(s)+c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

def v2(s):
    return (c.sqrt(s)-c.sqrt(sh-s))/(c.sqrt(s)+c.sqrt(sh-s))

def w2(s):
    return v2(s+1j*10**(-18))

wm=w2(0.85**2)

# Let's define the matching coefficients Bh0 and Bh1

def Bh0(b0,b1,b2):
    return b0 + b1*w(0.85**2) + b2*w(0.85**2)**2

def Bh1(b1,b2):
    return (b1+2*b2*w(0.85**2))*(s0/sh)*(np.sqrt(sh-0.85**2)/np.sqrt(s0-0.85**2))*((0.85+np.sqrt(sh-0.85**2))/(0.85+np.sqrt(s0-0.85**2)))**2

# The phase-shift is given by

def phi42(s,b0,b1,b2,Bh2,Bh3,Delta):
    if s<0.85**2:
        return mpi**8*c.sqrt(s)*s*(b0 + b1*w(s) + b2*w(s)**2)/(2*q(s,mpi)**9*(4*(mpi**2 + Delta**2)-s))
    else:
        return mpi**8*c.sqrt(s)*s*(Bh0(b0,b1,b2) + Bh1(b1,b2)*(w2(s)-wm) + Bh2*(w2(s)-wm)**2+ Bh3*(w2(s)-wm)**3)/(2*q(s,mpi)**9*(4*(mpi**2 + Delta**2)-s))

def deltaf2(s,b0,b1,b2,Bh2,Bh3,Delta):
    return 180*np.arctan(1/phi42(s,b0,b1,b2,Bh2,Bh3,Delta)).real/np.pi

# This is the final S2 Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    b0,b1,b2,Bh2,Bh3,Delta=params
    return (c.exp(np.pi*1j*deltaf2(s,b0,b1,b2,Bh2,Bh3,Delta)/90)-1)/(2j*sigma(s,mpi))


def S(s,params):
    return 1+ 2j*sigma(s,mpi)*tf(s,params)

def etaf(s,params):
    return abs(S(s,params))

def deltaf(s,params):
    b0,b1,b2,Bh2,Bh3,Delta=params
    return deltaf2(s,b0,b1,b2,Bh2,Bh3,Delta)