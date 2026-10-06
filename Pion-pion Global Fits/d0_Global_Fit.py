"""D0 wave Global Fits"""
# Parametrization and parameters for the three Global Fits given in Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c

mpi=0.13957
mk=0.4957
s0=1.05**2
sh=1.45**2

"""Global Fit I"""
# b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4
par_1=12.34,10.12,4.5,0.258,0.94,1.2746,0.9,-12.18,-1.94,-3.01,0.22,3.71,-4.3
err_par_1=0.13,0.15,1.8,0.015,0.03,0.0008,0,0.17,0.05,0.03,0.13,0.22,0.4

"""Global Fit II"""
# b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4
par_2=12.42,10.00,33,0.317,1.10,1.2756,2*mk,-39.7,-19.28,-8.86,11.99,-25.49,16.3
err_par_2=0.13,0.15,4,0.026,0.04,0.0008,0,0.5,0.14,0.07,0.11,0.23,0.4

"""Global Fit III"""
# b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4
par_3=12.48,10.02,34,0.322,1.16,1.2754,2*mk,-82.4,-42.65,-14.38,10.20,-18.8,14.2
err_par_3=0.13,0.15,5,0.024,0.03,0.0008,0,0.5,0.14,0.07,0.20,0.4,0.7

# AMPLITUDE UNDER 1.4 GeV

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

em=2*mk

wm=w2(em**2)

# Let's define the matching coefficients b0h and b1h

def b1h(b1,b2h):
    return (b1*(s0/sh)*(np.sqrt(sh-em**2)/np.sqrt(s0-em**2))*((em+np.sqrt(sh-em**2))/(em+np.sqrt(s0-em**2)))**2 -2*b2h*wm)

def b0h(b0,b1,b2h):
    return b0 + b1*w(em**2)-b1h(b1,b2h)*wm-b2h*wm**2

# The phase-shift below 1.4 GeV is given by

def phi30(s,b0,b1,b2h,mf2):
    if s<em**2:
        return np.sqrt(s)*mpi**2*(mf2**2 - s)*(b0 + b1*w(s))/(2*q(s,mpi)**5) # First conformal below KK
    else:
        return np.sqrt(s)*mpi**2*(mf2**2 - s)*(b0h(b0,b1,b2h) + b1h(b1,b2h)*w2(s) + b2h*w2(s)**2)/(2*q(s,mpi)**5) # Second conformal

def deltaf1(s,b0,b1,b2h,mf2):
    a=180*np.arctan(1/phi30(s,b0,b1,b2h,mf2)).real/np.pi
    if a>0:
        return a
    else:
        return a+180
    
# Now we define the elasticity below 1.4 GeV

def etaf1(s,eps,r,mf2,me):
    if s<me**2:  # Note that the inelastic threshold "me" depends on the Global Fit
        return 1
    else:
        return 1-eps*((1-(me**2/s))/((1-(me**2/mf2**2))))**(5/2)*(1+r*(1-((s-me**2)/(mf2**2-me**2))))

# The total amplitude below 1.4 GeV is given by

def tf1(s,b0,b1,b2h,eps,r,mf2,me):
    return (etaf1(s,eps,r,mf2,me)*c.exp(np.pi*1j*deltaf1(s,b0,b1,b2h,mf2)/90)-1)/(2j*sigma(s,mpi))

# AMPLITUDE OVER 1.4 GeV

sm=1.4**2

def x(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

# Numeric derivatives to compute the matching coefficients

derX=(x(sm+10**(-5))-x(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b2h,mf2):
    return (deltaf1(sm+10**(-5),b0,b1,b2h,mf2)-deltaf1(sm-10**(-5),b0,b1,b2h,mf2))/(2*10**(-5))

def derE(eps,r,mf2,me):
    return (etaf1(sm+10**(-5),eps,r,mf2,me)-etaf1(sm-10**(-5),eps,r,mf2,me))/(2*10**(-5))

def Delta(b0,b1,b2h,mf2,d2,d3,d4):
    return (derD(b0,b1,b2h,mf2)/derX) + 4*d2 - 9*d3 + 16*d4

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def c4(s):
    return 8*s**4 - 8*s**2 + 1

# We start with the phase-shift

def deltaf2(s,b0,b1,b2h,mf2,d2,d3,d4):
    return deltaf1(sm,b0,b1,b2h,mf2) + Delta(b0,b1,b2h,mf2,d2,d3,d4)*(c1(x(s))+1) + d2*(c2(x(s))-1) + d3*(c3(x(s))+1) + d4*(c4(x(s))-1)

def deltafr(s,b0,b1,b2h,mf2,d2,d3,d4):
    if s<sm:
        return deltaf1(s,b0,b1,b2h,mf2)
    else:
        return deltaf2(s,b0,b1,b2h,mf2,d2,d3,d4)    
    
# Finally, we define the elasticity above 1.4 GeV

qm=q(sm,mpi)

def Q(s):
    return (s/sm)-1

def eps0(eps,r,mf2,me): 
    return c.sqrt(-c.log(etaf1(sm,eps,r,mf2,me)))

def eps1(eps,r,mf2,me): 
    return -sm*derE(eps,r,mf2,me)/(2*eps0(eps,r,mf2,me)*etaf1(sm,eps,r,mf2,me))                             

def etaf2(s,eps,r,mf2,me,eps2,eps3,eps4):
    return np.exp(-(eps0(eps,r,mf2,me) + eps1(eps,r,mf2,me)*Q(s) + eps2*Q(s)**2 + eps3*Q(s)**3 + eps4*Q(s)**4)**2)

def etafr(s,eps,r,mf2,me,eps2,eps3,eps4):
    if s<sm:
        return etaf1(s,eps,r,mf2,me)
    else:
        return etaf2(s,eps,r,mf2,me,eps2,eps3,eps4).real
    
# This is the final D0 Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4=params
    return (etafr(s,eps,r,mf2,me,eps2,eps3,eps4)*c.exp(np.pi*1j*deltafr(s,b0,b1,b2h,mf2,me,d2,d3,d4)/90)-1)/(2j*sigma(s,mpi))

def deltaf(s,params):
    b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4=params
    return deltafr(s,b0,b1,b2h,mf2,d2,d3,d4)

def etaf(s,params):
    b0,b1,b2h,eps,r,mf2,me,d2,d3,d4,eps2,eps3,eps4=params
    return etafr(s,eps,r,mf2,me,eps2,eps3,eps4)