# Copyright 2021 Philip Schwedler
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This code is partially based on the source code of the Python Package
# sdeint (https://github.com/mattja/sdeint) written by Matthew J. Aburn & Yoav Ram.

from __future__ import absolute_import
import numpy as np
import numbers
from sdeint.wiener import deltaW, Ikpw, Iwik
from sdeint.integrate import (Error, SDEValueError, _check_args)
from .wiener_extension import (Imr, Jmr, Ihatkp, Itildekp, Iweakkp)

def _check_args_extension(f, G, y0, tspan, dW=None, IJ=None, Xi=None, dxG=None):
    
    # check the shape of G; since we do not know if the input is valid
    # at all, take a try-catch statement
    try:
        if callable(G):
        # then G must be a function returning a d x m matrix
            Gtest = G(y0, tspan[0])
        
            # if G returning array of shape(k,) then we assume that we are in the diagonal case
            # if we really have k=d will be tested in _check_args below
            if Gtest.ndim == 1:
                Gcheck = lambda y,t: np.diag(G(y,t))
            else:
                Gcheck = G
        else:
            Gcheck = G
    except:
        print("Unexpected error: ")
        raise
    (d, m, f, Gcheck, y0, tspan, dW, IJ)=_check_args(f, Gcheck, y0, tspan, dW, IJ)
    message = """G is a matrix of shape (%d, %d), therefore dxG
	             has to be of shape (%d, %d, %d)""" % (d, m, d, d, m)
    if dxG is not None:
        # then dxG must be a function returning a d x d x m matrix
        if callable(dxG):
            Gtest = dxG(y0, tspan[0])
            if Gtest.ndim != 3 or Gtest.shape[0] != d or Gtest.shape[1] != d or Gtest.shape[2] != m:
                raise SDEValueError(message)
    message = """From function G, it seems m==%d. If present, the optional
              parameter Xi must be an array of shape (len(tspan)-1, m) giving
              m independent Wiener increments for each time interval.""" % m
    if Xi is not None:
        if not hasattr(Xi, 'shape') or Xi.shape != (len(tspan) - 1, m):
            raise SDEValueError(message)

    return (d, m, f, G, y0, tspan, dW, IJ, Xi, dxG)



def itoMilstein(f, G, dxG, y0, tspan, Imethod=Imr, dW=None, I=None):
    """
	- Created by Philip Schwedler, 05/2021
	- Implements the Milstein order 1.0 strong approximation scheme for
	Ito equations dy = f(y,t)dt + G(y,t) dW(t).
	
    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments
	
    Args:
      f: callable(y, t) returning (d,) array
	     Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
	     Matrix-valued function to define the noise coefficients of the system
	  dxG: callable(y, t) returning (d,d,m) array
	    Tensor-valued function to define the derivatives of the noise
		coefficients of the system. Each element dxG[i,:,:] defines
		a matrix of shape (d,m) representing the derivatives of the 
		coefficients of G w.r.t the i-th coordinate.
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): Sequence of equally spaced time points
      IJmethod (callable): which function to use to generate repeated
        integrals. N.B. for an Ito equation, must use an Ito version here
        (either Ikpw or Iwik).
      dW: optional array of shape (len(tspan)-1, d). 
      IJ: optional array of shape (len(tspan)-1, m, m).
        Optional arguments dW and IJ are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.
		
    Returns:
      y: array, with shape (len(tspan), len(y0))
	  
    Raises:
      SDEValueError
	  
    See also:
	  G. N. Milstein (1974), Approximate integration of stochastic differential equations
    """
    (d, m, f, G, y0, tspan, dW, I, __Xi, dxG) = _check_args_extension(f, G, y0, tspan, dW, I, None, dxG)
    #(d, m, f, G, y0, tspan, dW, I) = _check_args(f, G, y0, tspan, dW, I)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h) # shape (N, m)
    if I is None: 
        # pre-generate repeated stochastic integrals for each time step.
        # Must give I_ij for the Ito case or J_ij for the Stratonovich case:
        __, I = Imethod(dW, h) # shape (N, m, m)
    else:
        I = I
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    Gn = np.zeros((d, m), dtype=y.dtype)
    for n in range(0, N-1):
        tn = tspan[n]
        h = tspan[n+1] - tn
        Yn = y[n] # shape (d,)
        Ik = dW[n,:] # shape (m,)
        Iij = I[n,:,:] # shape (m, m)
        Gn = G(Yn, tn) # shape (d, m)
        dxGn = dxG(Yn, tn) # shape (d, d, m)	
        fnh = f(Yn, tn)*h	
        Yn1 = Yn + fnh + np.dot(Gn, Ik)
        GnIij = np.dot(Gn, Iij).flatten()
        sum1 = np.array([np.dot(GnIij, dxGn[:,k,:].flatten()) for k in range(0,d)])
        y[n+1] = Yn1 + sum1
    return y
	
	

def itoSRIC2(f, G, y0, tspan, dW=None):
    """
    - Created by Philip Schwedler, 06/2021
    - Implements the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
    algorithms SRIC2 to integrate an Ito equation dy = f(y,t)dt + G(y,t)dW(t)
    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.
    This algorithm is suitable for Ito systems with an commutative noise
    coefficient matrix G. The algorithm has order 2.0 convergence for the
    deterministic part alone and order 1.0 strong convergence for the complete
    stochastic system (Theorem 6.3 in Roessler2010).
	
    Parameters
    -----------
    
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system
      G: The d x m coefficient function G can be given in two different ways:
         You can provide a single function G(y, t) that returns an array of
         shape (d, m). In this case the entire matrix G() will be evaluated
         2m+1 times at each time step so complexity grows quadratically with m.
         Alternatively you can provide a list of m functions g(y, t) each
         defining one column of G (each returning an array of shape (d,).
         In this case each g will be evaluated 3 times at each time step so
         complexity grows linearly with m. If your system has large m and
         G involves complicated functions, consider using this way.
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d). 
    
    Returns
    -------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    		 
    Raises
    ------
      SDEValueError
      
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    (d, m, f, G, y0, tspan, dW, __I, __Xi, __dxG) = _check_args_extension(f, G, y0, tspan, dW, None, None, None)
    #(d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
    have_separate_g = (not callable(G)) # if G is given as m separate functions
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h) # shape (N, m)
    
	# calculate matrix of increment products
    I =  np.einsum('ij,ik->ijk', dW, dW) # shape (N,m,m)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    Gn = np.zeros((d, m), dtype=y.dtype)
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        sqrth = np.sqrt(h)
        Yn = y[n] # shape (d,)
        Ik = dW[n,:] # shape (m,)
        Iij = I[n,:,:] # shape (m, m)
        fnh = f(Yn, tn)*h # shape (d,)
        if have_separate_g:
            for k in range(0, m):
                Gn[:,k] = G[k](Yn, tn)
        else:
            Gn = G(Yn, tn)
        sum1 = np.dot(Gn, Iij)/(2*sqrth) # shape (d, m)
        H20 = Yn + fnh # shape (d,)
        H20b = np.reshape(H20, (d, 1))
        H2 = H20b + sum1 - 0.5*sqrth*Gn # shape (d, m)
        H30 = Yn
        H3 = H20b - sum1 + 0.5*sqrth*Gn
        fn1h = f(H20, tn1)*h
        Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(Gn, Ik)
        if have_separate_g:
            for k in range(0, m):
                Yn1 += 0.5*sqrth*(G[k](H2[:,k], tn1) - G[k](H3[:,k], tn1))
        else:
            for k in range(0, m):
                Yn1 += 0.5*sqrth*(G(H2[:,k], tn1)[:,k] - G(H3[:,k], tn1)[:,k])
        y[n+1] = Yn1
    return y
	
	
	
def itoSRID2(f, G, y0, tspan, dW=None, Xi=None):
    """
    - Created by Philip Schwedler, 06/2021
    - Implements the Roessler2010 order 1.5 strong Stochastic Runge-Kutta
    algorithms SRIC2 to integrate an Ito equation dy = f(y,t)dt + G(y,t)dW(t)
    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.
    This algorithm is suitable for Ito systems with an diagonal noise
    coefficient matrix G. The algorithm has order 3.0 convergence for the
    deterministic part alone and order 1.5 strong convergence for the complete
    stochastic system (Theorem 6.5 in Roessler2010).
	- In the case m=1, it implements the Roessler2010 order 1.5 strong
    Stochastic Runge-Kutta algortihm SRI2W1 to integrate an Ito equation
    with scalaer noise dy = f(y,t)dt + G(y,t)dW(t) where y is d-dimensional
    vector variable, f is a vector-valued function, G is a (d x 1 matrix-valued)
	d-dimensional vector variable giving the noise coefficients and
	dW(t) is a vector of 1 independent Wiener increments.
	The algorithm has order 3.0 convergence for the
    deterministic part alone and order 1.5 strong convergence for the complete
    stochastic system (Theorem 6.4 in Roessler2010).
	
    Parameters
    -----------
    
      f: A function f(y, t) returning an array of shape (d,)
        Vector-valued function to define the deterministic part of the system
      G: A function G(y, t) returning an array of shape (d,)
        Vector-valued function which defines the diagonal entries of the noise
        coefficient of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d). 
      Xi: optional array of shape (len(tspan)-1, d).
    
    Returns
    -------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    		 
    Raises
    ------
      SDEValueError
      
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    (d, m, f, Gcheck, y0, tspan, dW, __I, Xi, __dxG) = _check_args_extension(f, G, y0, tspan, dW, None, Xi, None)
    #(d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
	
    if m != d and m != 1:
        raise SDEValueError("""itoSRID2 approximation scheme is only capable of
                               diagonal noise coefficient or general noise coefficient
                               with scalar noise""")
    if m == 1 and d != 1:
        G = lambda y,t: Gcheck(y,t).reshape(d)
    else:
	    G = Gcheck
		
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h) # shape (N-1, m)
    # calculate diagonal stochastic integrals
    if Xi is None:
        # pre-generate normal distributed random variables needed for
        # the approximation of I_(k,0)
        Xi = np.random.normal(0.0, np.sqrt(h), (N -1 , m)) # shape (N-1, m)
    Ik0 =  0.5*h*(dW + (1.0/np.sqrt(3.0))*Xi) # shape (N-1,m)
    Ikk = 0.5*(np.power(dW,2) - h) # shape (N-1,m)
    Ikkk = (1.0/6.0)*(np.power(dW,3) - 3.0*h*dW) # Hermite Polynomial representation
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        sqrth = np.sqrt(h)
        Yn = y[n] # shape (d,)
        Ikn = dW[n,:] # shape (d,)
        Ik0n = Ik0[n,:] # shape (d,)
        Ikkn = Ikk[n,:] # shape (d,)
        Ikkkn = Ikkk[n,:] # shape(d,)
        fnh = f(Yn, tn)*h # shape (d,)
        Gn = G(Yn, tn) # shape(d,)
        H20 = Yn + fnh # shape(d,)
        fn1h = f(H20, tn1)*h # shape(d,)
        H21 = Yn + 0.25*fnh - 0.25*sqrth*Gn # shape(d,)
        Gn1 = G(H21, 0.25*tn1 + 0.75*tn) # shape(d,)
        H30 = Yn + 0.25*fnh + 0.25*fn1h + (1.0/h)*np.multiply(Gn + 0.5*Gn1, Ik0n) # shape (d,)
        fn2h = f(H30, 0.5*(tn+tn1))*h # shape (d,)
        H31 = Yn + fnh + sqrth*Gn # shape(d,)
        Gn2 = G(H31, tn1) # shape(d,)
        H41 = Yn + 0.25*fn2h + sqrth*(2.0*Gn - Gn1 + 0.5*Gn2) # shape(d,)
        Gn3 = G(H41, 0.25*tn1 + 0.75*tn) # shape(d,)
        sum1 = np.multiply(-Ikn + (1.0/sqrth)*Ikkn + 2.0*(1.0/h)*Ik0n - 2.0*(1.0/h)*Ikkkn, Gn) # shape(d,)
        sum2 = np.multiply((4.0/3.0)*Ikn - (4.0/3.0)*(1.0/sqrth)*Ikkn - (4.0/3.0)*(1.0/h)*Ik0n + (5.0/3.0)*(1.0/h)*Ikkkn, Gn1) # shape(d,)
        sum3 = np.multiply((2.0/3.0)*Ikn + (1.0/3.0)*(1.0/sqrth)*Ikkn - (2.0/3.0)*(1.0/h)*Ik0n - (2.0/3.0)*(1.0/h)*Ikkkn, Gn2) # shape(d,)
        sum4 = np.multiply((1.0/h)*Ikkkn, Gn3) # shape(d,)
        Yn1 = Yn + (1.0/6.0)*fnh + (1.0/6.0)*fn1h + (2.0/3.0)*fn2h + sum1 + sum2 + sum3 + sum4 # shape(d,)
        y[n+1] = Yn1
    return y
	
	
	
def itoSRA3(f, G, y0, tspan, dW=None, Xi=None):
    """
    - Created by Philip Schwedler, 06/2021
    Use the Roessler2010 order 1.5 strong Stochastic Runge-Kutta algorithm
    SRA3 to integrate an Ito equation dy = f(y,t)dt + G(t)dW(t)
    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.
    This algorithm is suitable for Ito systems with an additive noise
    coefficient matrix G. The algorithm has order 3.0 convergence for the
    deterministic part alone and order 1.5 strong convergence for the complete
    stochastic system.
    
    Parameters
    ----------
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system
      G: A function G(t) returning an array of shape (d,m)
         Matrix-valued function to define the additive noise part of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d).
      Xi: optional array of shape (len(tspan)-1, d). 
      
    Returns
	-------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
		 
    Raises
	------
      SDEValueError
	  
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return _Roessler2010_SRA3(f, G, y0, tspan, dW, Xi)


def stratSRA3(f, G, y0, tspan, dW=None, Xi=None):
    """
	- Created by Philip Schwedler, 06/2021
    Use the Roessler2010 order 1.5 strong Stochastic Runge-Kutta algorithm
    SRA3 to integrate an Stratonovich equation dy = f(y,t)dt + G(t)\circ dW(t)
    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.
    This algorithm is suitable for Ito systems with an additive noise
    coefficient matrix G. The algorithm has order 3.0 convergence for the
    deterministic part alone and order 1.5 strong convergence for the complete
    stochastic system.
    
    Parameters
    ----------
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system
      G: A function G(t) returning an array of shape (d,m)
         Matrix-valued function to define the additive noise part of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d).
      Xi: optional array of shape (len(tspan)-1, d). 
      
    Returns
	-------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
		 
    Raises
	------
      SDEValueError
	  
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return _Roessler2010_SRA3(f, G, y0, tspan, dW, Xi)


def _Roessler2010_SRA3(f, G, y0, tspan, dW=None, Xi=None):
    """
    - Created by Philip Schwedler, 06/2021
    - Implements the Roessler2010 order 1.5 strong Stochastic Runge-Kutta
    algorithms SRA3 for Ito equations and for Stratonovich equations. 
    (Theorem 6.6 in Roessler2010).
    
    Parameters
    ----------
      f: A function f(y, t) returning an array of shape (d,)
      G: A function G(y, t) that returns an array of shape (d, m).
      y0: array of shape (d,) giving the initial state
      tspan (array): Sequence of equally spaced time points
      dW: optional array of shape (len(tspan)-1, m). 
      Xi: optional array of shape (len(tspan)-1, m).
        Optional arguments dW and Xi are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.
    
    Returns
    -------
      y: array, with shape (len(tspan), len(y0))
    
    Raises
    ------
      SDEValueError
    
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    (d, m, f, __G, y0, tspan, dW, __I, Xi, __dxG) = _check_args_extension(f, lambda y,t: G(t), y0, tspan, dW, None, Xi, None)
    #(d, m, f, __, y0, tspan, dW, IJ) = _check_args(f, lambda y,t: G(t), y0, tspan, dW, IJ)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h) # shape (N, m)
    # calculate stochastic integrals
    if Xi is None:
        # pre-generate normal distributed random variables needed for
        # the approximation of I_(k,0)
        Xi = np.random.normal(0.0, np.sqrt(h), (N -1 , m)) # shape (N-1, m)
    Ik0 =  0.5*h*(dW + (1.0/np.sqrt(3.0))*Xi) # shape (N-1,m)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        Yn = y[n] # shape (d,)
        Ik = dW[n,:] # shape (m,)
        Ik0n = Ik0[n,:,:] # shape (m,)
        fnh = f(Yn, tn)*h # shape (d,)
        H20 = Yn + fnh # shape (d,)
        fn1h = f(H20, tn1)*h # shape (d,)
        Gn = G(tn) # shape (d,m)
        GnI = np.dot(Gn, Ik0n) # shape (d,)
        Gn1 = G(tn1) # shape (d,m)
        Gn1I = np.dot(Gn1, Ik0n) # shape (d,)
        H30 = Yn + 0.25*fnh + 0.25*fn1h + (1.0/h)*Gn1I + 0.5*(1.0/h)*GnI # shape (d,)
        fn2h = f(H30, 0.5*(tn1+tn))*h # shape (d,)
        Yn1 = Yn + (1.0/6.0)*fnh + (1.0/6.0)*fn1h + (2.0/3.0)*fn2h + (1.0/h)*Gn1I + np.dot(Gn1, Ik) # shape (d,)
        y[n+1] = Yn1
    return y
	
	
def itoSRI2W1(f, G, y0, tspan, dW=None, Xi=None):
    """
    - Created by Philip Schwedler, 06/2021
    - It implements the Roessler2010 order 1.5 strong
    Stochastic Runge-Kutta algortihm SRI2W1 to integrate an Ito equation
    with scalaer noise dy = f(y,t)dt + G(y,t)dW(t) where y is d-dimensional
    vector variable, f is a vector-valued function, G is a (d x 1 matrix-valued)
	d-dimensional vector variable giving the noise coefficients and
	dW(t) is a vector of 1 independent Wiener increments.
    This algorithm is suitable for Ito systems with an general noise
    coefficient matrix G with scalar noise. 
	The algorithm has order 3.0 convergence for the
    deterministic part alone and order 1.5 strong convergence for the complete
    stochastic system (Theorem 6.4 in Roessler2010).
	
    Parameters
    -----------
    
      f: A function f(y, t) returning an array of shape (d,)
        Vector-valued function to define the deterministic part of the system
      G: A function G(y, t) returning an array of shape (d,1)
        Matrix-valued function which defines the noise
        coefficient of the system with scalar noise
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, 1). 
      Xi: optional array of shape (len(tspan)-1, 1).
    
    Returns
    -------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    		 
    Raises
    ------
      SDEValueError
      
    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return itoSRID2(f, G, y0, tspan, dW, Xi)
	
	

def itoTamedEuler(f, G, y0, tspan, dW=None):
    """
	- Created by Philip Schwedler, 05/2021
	- Use the tamed Euler-Maruyama algorithm to integrate the Ito equation
    dy = f(y,t)dt + G(y,t) dW(t)
    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments
    Args:
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d). This is for advanced use,
        if you want to use a specific realization of the d independent Wiener
        processes. If not provided Wiener increments will be generated randomly
    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    Raises:
      SDEValueError
    See also:
      Hutzenthaler, Jentzen and Kloeden (2012) Strong Convergence of an explicit
      numerical method for SDEs with nonglobally Lipschitz continuous coefficients
      
    """
    (d, m, f, G, y0, tspan, dW, __IJ, __Xi, __dxG) = _check_args_extension(f, G, y0, tspan, dW, None, None, None)
    #(d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h)
    y[0] = y0;
    for n in range(0, N-1):
        tn = tspan[n]
        yn = y[n]
        dWn = dW[n,:]
        fn = f(yn, tn)
        y[n+1] = yn + fn*h / (1+h*np.linalg.norm(fn)) + G(yn, tn).dot(dWn)
    return y
	
	
	
def itoRI5(f, G, y0, tspan, Ihat=None, Itilde=None):
    """
    - Created by Philip Schwedler, 06/2021
    - Implements the Roessler2009 order 2.0 weak Stochastic Runge-Kutta
    algorithms RI5 to integrate an expectation of a functional of the
    Ito equation dy = f(y,t)dt + G(y,t)dW(t)
    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.
    This algorithm is suitable for Ito systems with an general noise
    coefficient matrix G. The algorithm has order 3.0 convergence for the
    deterministic part alone and order 2.0 strong convergence for the complete
    stochastic system (Theorem 5.1 in Roessler2009).
	
    Parameters
    -----------
    
      f: A function f(y, t) returning an array of shape (d,)
        Vector-valued function to define the deterministic part of the system
      G: A function G(y, t) returning an array of shape (d,m)
        Vector-valued function which defines the diagonal entries of the noise
        coefficient of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      Ihat: optional array of shape (len(tspan)-1, m). 
        Three point distributed random variable to compute the weak
        approximations of the iterated stochastic integrals
      Itilde: optional array of shape (len(tspan)-1, m).
        Two point distributed random variable to compute the weak
        approximations of the iterated stochastic integrals

    Returns
    -------
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    		 
    Raises
    ------
      SDEValueError
      
    See also:
      A. Roessler (2009) Second Order Runge-Kutta Methods For
        Ito Stochastic Differential Equations
    """
    (d, m, f, G, y0, tspan, __dW, __I, __Xi, __dxG) = _check_args_extension(f, G, y0, tspan, None, None, None, None)
    #(d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
    have_separate_g = (not callable(G)) # if G is given as m separate functions
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    # weak approximations of twofold iterated stochastic integrals
    if Ihat is None:
        # pre-generate three point distributed random variables:
        Ihat = Ihatkp(N - 1, m, h) # shape (N-1, m)
    if Itilde is None:
        # pre-generate two point distributed random variables:
        Itilde = Itildekp(N - 1, m - 1, h) # shape (N-1, m-1)
    Iij = Iweakkp(N, m, h, Ihat, Itilde) # shape (N-1,m,m)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    Gn = np.zeros((d, m), dtype=y.dtype)
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        sqrth = np.sqrt(h)
        Yn = y[n] # shape (d,)
        Ihatn = Ihat[n,:] # shape (m,)
        Iijn = Iij[n,:,:] # shape (m,m)
        fnh = f(Yn, tn)*h # shape (d,)
        if have_separate_g:
            for k in range(0, m):
                Gn[:,k] = G[k](Yn, tn) # shape (d,m)
        else:
            Gn = G(Yn, tn) # shape (d,m)
        sum1 = np.dot(Gn, Ihatn) # shape (d,)
        H20 = Yn + fnh + (1.0/3.0)*sum1 # shape (d,)
        fn1h = f(H20, tn1)*h # shape (d,)
        H30 = Yn + (25.0/144.0)*fnh + (35.0/144.0)*fn1h - (5.0/6.0)*sum1 # shape (d,)
        fn2h = f(H30, (5.0/12.0)*tn1 + (7.0/12.0)*tn)*h # shape (d,)
        Yn1 = Yn + 0.1*fnh + (3.0/14.0)*fn1h + (24.0/35.0)*fn2h # shape (d,)
        sum2 = np.dot(Gn, np.transpose(Iijn)) # shape (d,m)
        if have_separate_g:
            for k in range(0, m):
                Gnk = Gn[:,k] # shape (d,)
                H2k = Yn + 0.25*fnh + 0.5*Gnk*sqrth # shape (d,)
                Gn1k = G[k](H2k, 0.25*tn1 + 0.75*tn) # shape (d,)
                H3k = Yn + 0.25*fnh - 0.5*Gnk*sqrth # shape (d,)
                Gn2k = G[k](H3k, 0.25*tn1 + 0.75*tn) # shape (d,)
                sum3 = (1.0/sqrth)*(sum2[:,k] - Gnk*Iijn[k,k]) # shape (d,)
                Hhat2k = Yn + sum3 # shape (d,)
                Gn3k = G[k](Hhat2k, tn) # shape (d,)
                Hhat3k = Yn - sum3 # shape (d,)
                Gn4k = G[k](Hhat3k, tn) # shape (d,)
                Yn1 += (Gnk-Gn1k-Gn2k)*Ihatn[k]+(1.0/sqrth)*(Gn1k-Gn2k)*Iijn[k,k]+(0.5*Gnk-0.25*Gn3k-0.25*Gn3k)*Ihatn[k]+0.5*(Gn3k-Gn4k)*sqrth # shape (d,)
        else:
            for k in range(0, m):
                Gnk = Gn[:,k] # shape (d,)
                H2k = Yn + 0.25*fnh + 0.5*Gnk*sqrth # shape (d,)
                Gn1k = G(H2k, 0.25*tn1 + 0.75*tn)[:,k] # shape (d,)
                H3k = Yn + 0.25*fnh - 0.5*Gnk*sqrth # shape (d,)
                Gn2k = G(H3k, 0.25*tn1 + 0.75*tn)[:,k] # shape (d,)
                sum3 = (1.0/sqrth)*(sum2[:,k] - Gnk*Iijn[k,k]) # shape (d,)
                Hhat2k = Yn + sum3 # shape (d,)
                Gn3k = G(Hhat2k, tn)[:,k] # shape (d,)
                Hhat3k = Yn - sum3 # shape (d,)
                Gn4k = G(Hhat3k, tn)[:,k] # shape (d,)
                Yn1 += (Gnk-Gn1k-Gn2k)*Ihatn[k]+(1.0/sqrth)*(Gn1k-Gn2k)*Iijn[k,k]+(0.5*Gnk-0.25*Gn3k-0.25*Gn3k)*Ihatn[k]+0.5*(Gn3k-Gn4k)*sqrth # shape (d,)
        y[n+1] = Yn1
    return y