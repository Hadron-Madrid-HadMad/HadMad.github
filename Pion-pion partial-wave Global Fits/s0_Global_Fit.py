"""S0 wave Global Fits"""
# Parametrization given in Eur.Phys.J.C 79, 1008 (2019) (See also the erratum)
# Parameters for the three Global Fits from Phys.Rev.D 111, 074003 (2025)

import numpy as np
import cmath as c

mpi=0.13957
mk=0.4957
s0=4*mk**2
sm=1.4**2

"""Global Fit I"""
# b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4
par_1=11.4,-0.6,18.6,-6.7,-20.2,5.6,0.137,0.996,0.025,5.04,-4.36,-0.05,-0.28,-11.9,0,0,13.7,0,0
err_par_1=0.3,1.1,2.7,3.1,3.7,4.8,0.028,0,0,0.28,0.16,0.16,0.06,3.7,0,0,4.0,0,0

"""Global Fit II"""
# b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4
par_2=12.1,-1.4,14.8,-4.9,-18.9,0.5,0.137,0.996,0.025,5.02,-4.71,0.01,-0.36,-11.1,0,0,81.7,-183.8,-51
err_par_2=0.3,0.8,1.5,1.5,1.3,4.5,0.028,0,0,0.08,0.08,0.18,0.04,6.2,0,0,2.4,8.5,25

"""Global Fit III"""
# b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4
par_3=11.9,-1.0,16.5,-5.3,-22.7,5.6,0.137,0.996,0.025,5.28,-4.64,0.18,-0.37,73.3,27.4,-0.27,171.7,-1041,1678
err_par_3=0.3,0.9,1.7,1.6,1.2,2.8,0.028,0,0,0.08,0.04,0.07,0.04,1.5,0.4,0.2,2,8,31

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
    return (c.sqrt(s)-c.sqrt(s0-s))/(c.sqrt(s)+c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

def phi00(s,b0,b1,b2,b3,b4,b5,z0):
    return c.sqrt(s)*mpi**2*((z0**2/(mpi*c.sqrt(s)))+b0 + b1*w(s)+ b2*w(s)**2+ b3*w(s)**3+ b4*w(s)**4+ b5*w(s)**5)/(2*q(s,mpi)*(s-(z0**2/2)))

def tc2(s,b0,b1,b2,b3,b4,b5,z0):
    return 1/(sigma(s,mpi)*(phi00(s,b0,b1,b2,b3,b4,b5,z0)-1j))

# Now, we implement the f0(980) amplitude

def J(s,m):
    return (2+sigma(s,m)*c.log((sigma(s,m)-1)/(sigma(s,m)+1)))/np.pi

def w1(s):
    return (2*(c.sqrt(s)-2*mk)/(1.5-2*mk))-1

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def f(s,k0,k1,k2,k3):
    return mk**2*(k0+k1*c1(w1(s))+k2*c2(w1(s))+k3*c3(w1(s)))

def sp(rsp,isp):
    return (rsp - isp*1j)**2

def fr(rsp,isp,k0,k1,k2,k3):
    return f(sp(rsp,isp),k0,k1,k2,k3).real

def fi(rsp,isp,k0,k1,k2,k3):
    return f(sp(rsp,isp),k0,k1,k2,k3).imag

def Jkr(rsp,isp):
    return J(sp(rsp,isp),mk).real

def Jki(rsp,isp):
    return J(sp(rsp,isp),mk).imag

def Jpr(rsp,isp):
    return J(sp(rsp,isp),mpi).real

def Jpi(rsp,isp):
    return J(sp(rsp,isp),mpi).imag

def sigmar(rsp,isp):
    return sigma(sp(rsp,isp),mpi).real

def sigmai(rsp,isp):
    return sigma(sp(rsp,isp),mpi).imag

def sr(rsp,isp):
    return sp(rsp,isp).real

def si(rsp,isp):
    return sp(rsp,isp).imag

def d(rsp,isp):
    return Jpi(rsp,isp)*sr(rsp,isp) + Jpr(rsp,isp)*si(rsp,isp) + 2*(si(rsp,isp)*sigmai(rsp,isp) - sr(rsp,isp)*sigmar(rsp,isp))

def G(rsp,isp,k0,k1,k2,k3):
    return -((fi(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp) + fr(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp) + si(rsp,isp))/d(rsp,isp))

def M(rsp,isp,k0,k1,k2,k3):
    return ((((fi(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp) + fr(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp))*(si(rsp,isp)*(Jpi(rsp,isp) - 2*sigmar(rsp,isp)) - sr(rsp,isp)*(Jpr(rsp,isp) + 2*sigmai(rsp,isp)))) + (Jpi(rsp,isp) - 2*sigmar(rsp,isp))*(si(rsp,isp)**2 + sr(rsp,isp)**2))/d(rsp,isp)) - (fi(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp) - fr(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp))

def tf2(s,rsp,isp,k0,k1,k2,k3):
    return s*G(rsp,isp,k0,k1,k2,k3)/(M(rsp,isp,k0,k1,k2,k3)-s-J(s,mpi)*s*G(rsp,isp,k0,k1,k2,k3)-J(s,mk)*f(s,k0,k1,k2,k3))

# Let's define the total amplitude below 1.4 GeV

def t12(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    return tc2(s,b0,b1,b2,b3,b4,b5,z0) + tf2(s,rsp,isp,k0,k1,k2,k3) + 2j*sigma(s,mpi)*tc2(s,b0,b1,b2,b3,b4,b5,z0)*tf2(s,rsp,isp,k0,k1,k2,k3)

def S1(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    return 1+ 2j*sigma(s,mpi)*t12(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)

def eta11(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    return abs(S1(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3))

def delta11(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    a=90*c.phase(S1(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3))/np.pi
    if s>1.1**2 and a<0:
        return  a+360
    elif a<0 or s>0.98**2:
        return a+180
    else:
        return a

# AMPLITUDE OVER 1.4 GeV

# We start with the phase-shift

def w2(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

derW=(w2(sm+10**(-5))-w2(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    return (delta11(sm+10**(-5),b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)-delta11(sm-10**(-5),b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3))/(2*10**(-5))

def derE(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3):
    return (eta11(sm+10**(-5),b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)-eta11(sm-10**(-5),b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3))/(2*10**(-5))

def Delta(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    return (derD(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)/derW) + 4*d0 - 9*d1 + 16*d2

def c4(s):
    return 8*s**4 - 8*s**2 + 1

def delta2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    return delta11(sm,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3) + Delta(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2)*(c1(w2(s))+1) + d0*(c2(w2(s))-1) + d1*(c3(w2(s))+1) + d2*(c4(w2(s))-1)

def deltaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    if s<sm:
        return delta11(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)
    else:
        return delta2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2)    
    
# Finally, we define the elasticity above 1.4 GeV

qm=q(sm,mpi)

def Q(s):
    return (q(s,mpi)/qm)-1

def eps0(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3): 
    return c.sqrt(-c.log(eta11(sm,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)))

def eps1(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3): 
    return -4*qm**2*derE(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)/(eps0(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)*eta11(sm,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3))                              

def eta2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4):
    return np.exp(-(eps0(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3) + eps1(b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)*Q(s) + eps2*Q(s)**2 + eps3*Q(s)**3 + eps4*Q(s)**4)**2)

def etaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4):
    if s<sm:
        return eta11(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3)
    else:
        return eta2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4).real

# This is the final S0 Global Fit parametrization, and the corresponding phase-shift and elasticity functions

def tf(s,params):
    b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return (etaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4)*c.exp(np.pi*1j*deltaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2)/90)-1)/(2j*sigma(s,mpi))

def deltaf(s,params):
    b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return deltaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2)

def etaf(s,params):
    b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return etaf2(s,b0,b1,b2,b3,b4,b5,z0,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4)