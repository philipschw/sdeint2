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
from .wiener_extension import Imr


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
    (d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
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
	


def itoMilstein(f, G, y0, tspan, IJmethod=Imr, dW=None, IJ=None, dxG=None):
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
    if dxG==None:
        raise SDEValueError('Required argument dxG for the Milstein scheme is missing.')
    (d, m, f, G, y0, tspan, dW, IJ) = _check_args(f, G, y0, tspan, dW, IJ)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h) # shape (N, m)
    if IJ is None: 
        # pre-generate repeated stochastic integrals for each time step.
        # Must give I_ij for the Ito case or J_ij for the Stratonovich case:
        __, I = IJmethod(dW, h) # shape (N, m, m)
    else:
        I = IJ
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
		print(GnIij)
		print(dxGn[:,0,:].flatten())
        sum1 = np.array([np.dot(GnIij, dxGn[:,k,:].flatten()) for k in range(0,d)])
        y[n+1] = Yn1 + sum1
    return y