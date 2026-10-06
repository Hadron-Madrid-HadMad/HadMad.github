"""S2 wave Global Fits"""
# Parametrization and parameters for the three Global Fits given in Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c


mpi=0.13957
s0=1.05**2
sh=2.3**2

"""Global Fit I"""
# b0,b1,z2,Bh2,Bh3,eps
par_1=-76.5,-57,0.142,290,-2466,0.32
err_par_1=2.8,11,0.004,109,355,0.06

"""Global Fit II"""
# b0,b1,z2,Bh2,Bh3,eps
par_2=-80.2,-64,0.147,340,-2489,0.33
err_par_2=2.8,11,0.004,109,355,0.06

"""Global Fit III"""
# b0,b1,z2,Bh2,Bh3,eps
par_3=-78.1,-58,0.145,327,-2615,0.30
err_par_3=2.8,11,0.004,109,355,0.06


# First, we define the phase-shift, which has two conformal pieces

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

def Bh0(b0,b1):
    return b0 + b1*w(0.85**2)

def Bh1(b1):
    return b1*(s0/sh)*(np.sqrt(sh-0.85**2)/np.sqrt(s0-0.85**2))*((0.85+np.sqrt(sh-0.85**2))/(0.85+np.sqrt(s0-0.85**2)))**2

# The phase-shift is given by

def phi20(s,b0,b1,z2,Bh2,Bh3):
    if s<0.85**2:
        return c.sqrt(s)*mpi**2*(b0 + b1*w(s))/(2*q(s,mpi)*(s-2*z2**2))
    else:
        return c.sqrt(s)*mpi**2*(Bh0(b0,b1) + Bh1(b1)*(w2(s)-wm) + Bh2*(w2(s)-wm)**2 + Bh3*(w2(s)-wm)**3)/(2*q(s,mpi)*(s-2*z2**2))

def deltaf2(s,b0,b1,z2,Bh2,Bh3):
    return 180*np.arctan(1/phi20(s,b0,b1,z2,Bh2,Bh3)).real/np.pi

# Now we define the elasticity

def etaf2(s,eps):
    if s<0.915**2:
        return 1
    else:
        return 1-eps*(1-(0.915**2/s))**(3/2)

# This is the final S2 Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    b0,b1,z2,Bh2,Bh3,eps=params
    return (etaf2(s,eps)*c.exp(np.pi*1j*deltaf2(s,b0,b1,z2,Bh2,Bh3)/90)-1)/(2j*sigma(s,mpi))

def S(s,params):
    return 1+ 2j*sigma(s,mpi)*tf(s,params)

def deltaf(s,params):
    return 90*c.phase(S(s,params))/np.pi

def etaf(s,params):
    b0,b1,z2,Bh2,Bh3,eps=params
    return  etaf2(s,eps)