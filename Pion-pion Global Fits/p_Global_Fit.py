"""P wave Global Fits"""
# Parametrization and parameters for the three Global Fits given in Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c

mpi=0.13957
mk=0.4957
s0=1.43**2
alpha=0.3
mpi0=mpi
mom=0.78266


"""Global Fit I"""
# b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4
par_1=1.11,-0.833,0.80,2.74,-1.82,0.396,-0.5825,0.2616,-0.0120,0.7707,-64.64,-46.95,-31.14,-10.82,-0.928,-0.026,0.321,0.131
err_par_1=0.05,0.013,0.04,0.08,0.10,0.003,0.0008,0.0005,0.0003,0.0012,0.26,0.08,0.04,0.03,0.018,0.009,0.004,0.003

"""Global Fit II"""
# b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4
par_2=1.20,-1.193,1.47,3.17,-2.71,0.4475,-0.6936,0.3529,-0.0459,0.7697,0.3,0.61,0,0,0,-0.19,-0.024,0.0033
err_par_2=0.06,0.016,0.05,0.09,0.11,0.0015,0.0008,0.0005,0.0003,0.0012,0.4,0.10,0,0,0,0.012,0.003,0.0016

"""Global Fit III"""
# b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4
par_3=1.18,-1.144,1.30,3.26,-2.72,0.4638,-0.6850,0.3151,-0.0223,0.7696,0.6,0.22,0,0,0,-0.282,-0.067,-0.0171
err_par_3=0.06,0.017,0.05,0.10,0.14,0.0019,0.0011,0.0007,0.0005,0.0012,0.5,0.13,0,0,0,0.013,0.004, 0.0018


# AMPLITUDE UNDER 1.4 GeV

# First, the conformal amplitude

def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))

    
def q(s,m):
    return c.sqrt((s/4)-m**2)

def v(s):
    return (c.sqrt(s)-alpha*c.sqrt(s0-s))/(c.sqrt(s)+alpha*c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

def phi11(s,b0,b1,b2,b3,b4,mrho):
    tt=c.sqrt(s)*(mrho**2 - s)*(((2*mpi**3)/(mrho**2*c.sqrt(s)))+b0 + b1*w(s)+ b2*w(s)**2+ b3*w(s)**3+ b4*w(s)**4)/(2*q(s,mpi)**3)
    if s.imag>=0:
        return tt
    else:
        return-tt

def tc2(s,b0,b1,b2,b3,b4,mrho):
    return 1/(sigma(s,mpi)*(phi11(s,b0,b1,b2,b3,b4,mrho)-1j))

def Sc2(s,b0,b1,b2,b3,b4,mrho):
    return 1+ 2j*sigma(s,mpi)*tc2(s,b0,b1,b2,b3,b4,mrho)

def deltac2(s,b0,b1,b2,b3,b4,mrho):
    a= 90*c.phase(Sc2(s,b0,b1,b2,b3,b4,mrho))/np.pi
    if a>0:
        return a
    else:
        return a+180
    
# Now, we implement the inelastic amplitude

sm1=(mpi0+mom)**2

def nu(s,m1,m2):
    return c.sqrt((s-(m1+m2)**2)*(s-(m1-m2)**2))

def J(s,m1,m2):
    Delta=m1**2-m2**2
    Sigma=m1**2+m2**2
    return (2+(Delta/s -Sigma/Delta)*np.log(m2**2/m1**2)+(nu(s,m1,m2)*(c.log((nu(s,m1,m2)-s+Delta)/(nu(s,m1,m2)+s+Delta))+c.log((nu(s,m1,m2)-s-Delta)/(nu(s,m1,m2)+s-Delta)))/s))/(2*np.pi)

mp=(mpi0+mom)

def R(s):
    return s/mp**2 -1

def deltain(s,k0,k1,k2,k3):
    q2=(s-(mpi0+mom)**2)*(s-(mpi0-mom)**2)/(4*s)
    return  J(s,mpi0,mom)*(k0+k1*R(s)+k2*R(s)**2+k3*R(s)**3)*q(s,mpi)**3*q2/(c.sqrt(s)*mpi**2)/(mp**2)


def tin2(s,k0,k1,k2,k3):
    return  (np.exp(2j*deltain(s,k0,k1,k2,k3))-1)/(2j*sigma(s,mpi))

def Sin2(s,k0,k1,k2,k3):
    return 1+ 2j*sigma(s,mpi)*tin2(s,k0,k1,k2,k3)

# Let's define the total amplitude below 1.4 GeV

def S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return Sc2(s,b0,b1,b2,b3,b4,mrho)*Sin2(s,k0,k1,k2,k3)

def etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return abs(S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))


def deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    a= 90*c.phase(S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/np.pi
    if a>0:
        return a
    else:
        return a+180

def tf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return  (etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*np.exp(1j*deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*90/np.pi)-1)/(2j*sigma(s,mpi))

# AMPLITUDE OVER 1.4 GeV

sm=1.4**2

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def c4(s):
    return 8*s**4 - 8*s**2 + 1

def c5(s):
    return 16*s**5 - 20*s**3 + 5*s

def c6(s):
    return 32*s**6 - 48*s**4 + 18*s**2-1

def c7(s):
    return 64*s**7 - 112*s**5 + 56*s**3 - 7*s

def x(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

# Numeric derivatives to compute the matching coefficients

derX=(x(sm+10**(-5))-x(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return (deltaf1(sm+10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)-deltaf1(sm-10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/(2*10**(-5))

def derE(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return (etaf1(sm+10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)-etaf1(sm-10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/(2*10**(-5))

# We start with the phase-shift

def Delta(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6):
    return (derD(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)/derX) + 4*d2 - 9*d3 +16*d4 - 25*d5 +36*d6

def deltaf2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6):
    return deltaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho) + Delta(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6)*(c1(x(s))+1) + d2*(c2(x(s))-1) + d3*(c3(x(s))+1) + d4*(c4(x(s))-1) + d5*(c5(x(s))+1) + d6*(c6(x(s))-1) 

def deltafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6):
    if s.real<sm:
        return deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)
    else:
        return deltaf2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6)

# Finally, we define the elasticity above 1.4 GeV

def eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho): # Caution, this eps0 is different from the one in the paper, everything is rewritten in a different way
    return np.sqrt(-np.log(etaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)))

def eps1(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    return -derE(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)/(2*eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*etaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*derX) +4*eps2 -9*eps3 +16*eps4                             
    
def eta2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    return np.exp(-(eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho) + eps1(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)*(c1(x(s))+1) + eps2*(c2(x(s))-1) + eps3*(c3(x(s))+1) + eps4*(c4(x(s))-1))**2)

def etafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    if s.real<sm:
        return etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)
    else:
        return eta2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4).real
    
# This is the final P Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4=params 
    return (etafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)*c.exp(np.pi*1j*deltafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6)/90)-1)/(2j*sigma(s,mpi))

def deltaf(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4=params
    return deltafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6)

def etaf(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d2,d3,d4,d5,d6,eps2,eps3,eps4=params
    return etafr(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)