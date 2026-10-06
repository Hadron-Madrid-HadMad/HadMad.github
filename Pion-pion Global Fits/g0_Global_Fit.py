"""G0 wave Global Fits"""
# Parametrization and parameters for the three Global Fits given in Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c


mpi=0.13957
sm=1.4**2
s0=1.75**2
alpha=0.5
s_ine=(1.05)**2


"""Global Fit"""
# Gg,mg,Rg,xg
par= 0.206,  2.011,  4.671, 0.16
err_par=0.080, 0.023, 0, 0.03


def q(s,m):
    return c.sqrt((s/4)-m**2)

def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))
    
# First, the Breit-Wigner amplitude, valid above 1.4 GeV

def q0(mg):
    return q(mg**2,mpi)

def D(s):
    return 11025 + 1575*s**2 + 135*s**4 + 10*s**6 + s**8

def G(s,Gg,mg,Rg):
    return Gg*(q(s,mpi)/q0(mg))**9*D(q0(mg)*Rg)/D(q(s,mpi)*Rg)

def tbw(s,Gg,mg,Rg,xg):
    return xg*mg*G(s,Gg,mg,Rg)/(sigma(s,mpi)*(mg**2-s-1j*mg*G(s,Gg,mg,Rg)))

def Sbw(s,Gg,mg,Rg,xg):
    return 1+ 2j*sigma(s,mpi)*tbw(s,Gg,mg,Rg,xg)

def etabw(s,Gg,mg,Rg,xg):
    return abs(Sbw(s,Gg,mg,Rg,xg))

def deltabw(s,Gg,mg,Rg,xg):
    return 90*c.phase(Sbw(s,Gg,mg,Rg,xg))/np.pi

# Now, let's define the low-energy conformal amplitude, matched to the B-W amplitude at 1.4 GeV

def v(s):
    return (c.sqrt(s)-alpha*c.sqrt(s0-s))/(c.sqrt(s)+alpha*c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

# We define the following matching structures for the phase-shift

def fac(s):
    return c.sqrt(s)/(2*q(s,mpi)**9)*mpi**8

def cotdbw_eff(s,Gg,mg,Rg,xg):
    return 1/(np.tan(c.phase(Sbw(s,Gg,mg,Rg,xg)).real/2)*fac(s))

def derC(Gg,mg,Rg,xg):
    return (cotdbw_eff(sm+10**(-5),Gg,mg,Rg,xg)-cotdbw_eff(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

derW=(w(sm+10**(-5))-w(sm-10**(-5)))/(2*10**(-5))

# The matching coefficients b0 and b1 are given by

def b1(Gg,mg,Rg,xg):
    return derC(Gg,mg,Rg,xg)/derW

def b0(Gg,mg,Rg,xg):
    return cotdbw_eff(sm,Gg,mg,Rg,xg) - b1(Gg,mg,Rg,xg)*w(sm)

#  The conformal phase-shift is defined

def phi(s,Gg,mg,Rg,xg):
    return mpi**8*(b0(Gg,mg,Rg,xg) + b1(Gg,mg,Rg,xg)*w(s))/(sigma(s,mpi)*q(s,mpi)**8)

def deltac(s,Gg,mg,Rg,xg):
    return 180*np.arctan(1/phi(s,Gg,mg,Rg,xg))/np.pi

#  The phase-shift in the whole range is

def delta(s,Gg,mg,Rg,xg):
    if s<sm:
        return deltac(s,Gg,mg,Rg,xg).real
    else:
        return deltabw(s,Gg,mg,Rg,xg).real
    
# We define the following matching coefficients for the elasticity
    
def eps(Gg,mg,Rg,xg):
    return (1-etabw(sm,Gg,mg,Rg,xg))/(1-(s_ine/sm))**(9/2)

def derE(Gg,mg,Rg,xg):
    return (etabw(sm+10**(-5),Gg,mg,Rg,xg)-etabw(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

def r(Gg,mg,Rg,xg):
    return -(sm*derE(Gg,mg,Rg,xg)/(1-etabw(sm,Gg,mg,Rg,xg)) +(9/2)*s_ine/(sm-s_ine))

#  The elasticity in the whole range is
    
def eta(s,Gg,mg,Rg,xg):
    if s<s_ine:
        return 1
    elif s<sm:
        return 1-eps(Gg,mg,Rg,xg).real*(1-(s_ine/s))**(9/2)*(1+r(Gg,mg,Rg,xg).real*(1-(sm/s)))
    else:
        return etabw(s,Gg,mg,Rg,xg).real
    
# This is the final G0 Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    Gg,mg,Rg,xg=params
    return (eta(s,Gg,mg,Rg,xg)*c.exp(np.pi*1j*delta(s,Gg,mg,Rg,xg)/90)-1)/(2j*sigma(s,mpi))

def deltaf(s,params):
    Gg,mg,Rg,xg=params
    return delta(s,Gg,mg,Rg,xg)

def etaf(s,params):
    Gg,mg,Rg,xg=params
    return eta(s,Gg,mg,Rg,xg)