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

"""
- Comment created by Matthew J. Aburn
- 05/2021: Modification by Philip Schwedler: Adding method of Mrongowius and
           Rössler (2021)

Simulation of standard multiple stochastic integrals, both Ito and Stratonovich
I_{ij}(t) = \int_{0}^{t}\int_{0}^{s} dW_i(u) dW_j(s)  (Ito)
J_{ij}(t) = \int_{0}^{t}\int_{0}^{s} \circ dW_i(u) \circ dW_j(s)  (Stratonovich)
These multiple integrals I and J are important building blocks that will be
used by most of the higher-order algorithms that integrate multi-dimensional
SODEs.

Strong approximation
--------------------
We implement the method of Mrongowius and Rössler (2021) which improves
the method of Wiktorsson (2001) by needing significantly less computational
effort than Wiktorsson by approximating the the truncation term by some
appropiate multivariate Gaussian random variable (see Section 5.2 of
Mrongowius and Rössler (2021) for a detailed discussion).

Weak approximation
------------------
We implement the method of Kloeden and Platen (1992) presented on page 225.

References:
  P. Kloeden, E. Platen and I. Wright (1992) The approximation of multiple
    stochastic integrals
  M. Wiktorsson (2001) Joint Characteristic Function and Simultaneous
    Simulation of Iterated Ito Integrals for Multiple Independent Brownian
    Motions
  J. Mrongowius and A. Rössler (2021) On the Approximation and Simulation
    of Iterated Stochastic Integrals and the Corresponding Levy Areas in 
    Terms of a Multidimensional Brownian Motion
  P. Kloden and E. Platen (1992) Numerical Solution of Stochastic Differential
    Equations, Third Printing
"""

import numpy as np
from sdeint.wiener import (deltaW, _dot, _vec, _unvec, _kp, _P, _K, _AtildeTerm, _a)

numpy_version = list(map(int, np.version.short_version.split('.')))
if numpy_version >= [1,10,0]:
    broadcast_to = np.broadcast_to
else:
    from ._broadcast import broadcast_to
	

def _AtildeTerm1(N, h, m, dW, Km0, Pm0):
    """
	- Created by Philip Schwedler, 05/2021
	- first part in the truncation term split [Mrongowius and Rössler 2021, p7, (22))]
	"""
    M = m*(m-1)//2
    psi_n1 = np.random.normal(0.0, 1.0, (N, m, 1))
    factor1 = np.dot(Km0, Pm0 - np.eye(m**2))
    factor1 = broadcast_to(factor1, (N, M, m**2))
    factor2 = _kp(dW, psi_n1)
    return (np.sqrt(h/2.0)/np.pi)*_dot(factor1, factor2)
	

def Imr(dW, h, n=5):
    """
	- Created by Philip Schwedler, 05/2021
	- Matrix I approximating repeated Ito integrals for each of N time
    intervals, using the method of Mrongowius and Rössler (2021).
	
    Parameters
	---------
      dW: array of shape (N, m)
        giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h: float
	    time step size
      n: int, optional
	    number of terms to take in the series expansion before truncation
		
    Returns
	--------
      (Atilde, I): where Atilde is an array of shape (N,m(m-1)//2,1)
                   giving the area integrals used, and
				   I is an array of shape (N, m, m) giving an m x m matrix
				   of repeated Ito integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1)) # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise(ValueError)
    if m == 1:
        return (np.zeros((N, 1, 1)), (dW*dW - h)/2.0)
		
	# Define permutation matrix and selection matrix [Mrongowius and Roessler (2021), p6]
    Pm0 = _P(m)
    Km0 = _K(m)
    M = m*(m-1)//2
	
	# Atilde term in [Mrongowius and Roessler (2021), p6, (18)]
    Atilde_n = _AtildeTerm(N, h, m, 1, dW, Km0, Pm0)
    for k in range(2, n+1):
	    # kth term in the sum for Atilde [Mrongowius and Roessler (2021), p6, (18)]
        Atilde_n += _AtildeTerm(N, h, m, k, dW, Km0, Pm0)
    Atilde_n = (h/(2.0*np.pi))*Atilde_n # approximation after n terms
	
	# First Summand from the tail-term [Mrongowius and Roessler (2021), p7, (22)]
    an = _a(n)**0.5
    Atilde_n1 = an*_AtildeTerm1(N, h, m, dW, Km0, Pm0)
	
	# Second Summand from the tail-term [Mrongowius and Roessler (2021), p8, (26)]
    psi_n2 = np.random.normal(0.0, 1.0, (N, M, 1))
    Atilde_n2 = (h/(np.pi*np.sqrt(2.0)))*an*psi_n2
    Atilde = Atilde_n + Atilde_n1 + Atilde_n2 # our final approximation of the areas
	
	# Calculate approximation of I [Mrongowius and Roessler (2021), p9, (27)]
    Im = broadcast_to(np.eye(m), (N, m, m))
    Ims0 = np.eye(m**2)
    factor3 = broadcast_to(np.dot(Ims0 - Pm0, Km0.T), (N, m**2, M))
    vecI = 0.5*(_kp(dW, dW) - _vec(h*Im)) + _dot(factor3, Atilde)
    I = _unvec(vecI)
    dW = dW.reshape((N, -1)) # change back to shape (N, m)
    return (Atilde, I)
	
	
def Jmr(dW, h, n=5):
    """
    - Created by Philip Schwedler, 05/2021
    - Matrix J approximating repeated Stratonovich integrals for each of N time
    intervals, using the method of Mrongowius and Rössler (2021).
    
    Parameters
    ---------
    dW: array of shape (N, m)
        giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
    h: float
       time step size
    n: int, optional
       number of terms to take in the series expansion before truncation
    
    Returns
    --------
    (Atilde, J): where Atilde is an array of shape (N,m(m-1)//2,1)
                 giving the area integrals used, and
                 I is an array of shape (N, m, m) giving an m x m matrix
                 of repeated Stratonovich integrals values for each of the
                 N time intervals.
    """
    m = dW.shape[1]
    Atilde, I = Imr(dW, h, n)
    J = I + 0.5*h*np.eye(m).reshape((1, m, m))
    return (Atilde, J)
	
	
	
def Ihatkp(N, m, h):
    """
    - Created by Philip Schwedler, 06/2021
    - Vector I representing three-point distribution for weak approximations
      of the iterated stochastic integrals, using [Kloden and Platen (1992), p225]
  
    Parameters
    ----------
    N: int
       number of time steps
    m: int
       number of random variables needed correspondiing to the dimension of the
       Wiener processes
    h: float
       time step size
	
    Returns
    --------
    I: where I is an array of shape (N, m) representing at each time step
       m realizations of the three-point distribution for the weak
       approximation of the iterated stochastic integrals
   
    See also:
       P. Kloeden and E. Platen (1992) Numerical Solution of Stochastic
       Differential Equations, Third Printing
    """
    return np.random.choice(np.array([-np.sqrt(3.0*h), 0.0, np.sqrt(3.0*h)]), size=(N,m), p=np.array([1.0/6.0, 2.0/3.0, 1.0/6.0]))

def Itildekp(N, m, h):
    """
    - Created by Philip Schwedler, 06/2021
    - Vector I representing two-point distribution for weak approximations
      of the iterated stochastic integrals, using [Kloden and Platen (1992), p225]

    Parameters
    ----------
    N: int
       number of time steps
    m: int
       number of random variables needed correspondiing to the dimension of the
       Wiener processes
    h: float
       time step size

    Returns
    --------
    I: where I is an array of shape (N, m) representing at each time step
       m realizations of the two-point distribution for the weak
       approximation of the iterated stochastic integrals

    See also:
       P. Kloeden and E. Platen (1992) Numerical Solution of Stochastic
       Differential Equations, Third Printing
    """
    return np.random.choice(np.array([-np.sqrt(h), np.sqrt(h)]), size=(N,m), p=np.array([0.5, 0.5]))

def Iweakkp(N, m, h, Ihat=None, Itilde=None):
    """
    - Created by Philip Schwedler, 06/2021
    - Vector I representing weak approximations of the iterated stochastic integrals,
      using [Kloden and Platen (1992), p225]

    Parameters
    ----------
    N: int
       number of time steps
    m: int
       number of random variables needed correspondiing to the dimension of the
       Wiener processes
    h: float
       time step size
    Ihat: array of shape (N, m)
       three-point distribution needed for the weak approximation
    Itilde: array of shape (N, m-1)
       two-point distribution needed for the weak approximaton

    Returns
    --------
    I: where I is an array of shape (N, m, m) giving an m x m matrix
       of repeated weak Ito integrals values for each of the
       N time intervals.

    See also:
        P. Kloeden and E. Platen (1992) Numerical Solution of Stochastic
        Differential Equations, Third Printing
	"""
    if Ihat is None:
        # pre-generate samples Ihat
        Ihat = Ihatkp(N,m,h) # shape (N,m)
    if Itilde is None:
        # pre-generate samples Itilde
        Itilde = Itildekp(N,m-1,h) # shape (N,m-1)
    sqrth = np.sqrt(h)
    M = m*(m-1)//2
    Iweak = np.einsum('ij,ik->ijk', Ihat, Ihat)
    for k in range(0,m):
        Iweak[:,k,k] = Iweak[:,k,k] - h
        for l in range(0,m):
            if(k<l):
                Iweak[:,k,l] = Iweak[:,k,l] - sqrth*Itilde[:,k]
            elif(k>l):
                Iweak[:,k,l] = Iweak[:,k,l] + sqrth*Itilde[:,l]			
    return 0.5*Iweak
