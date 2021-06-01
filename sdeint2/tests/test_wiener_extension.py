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

import pytest
import numpy as np
from sdeint.wiener import (deltaW, _t, _dot, _unvec, _P, _K)
from sdeint2.wiener_extension import (Imr, Jmr)

numpy_version = list(map(int, np.version.short_version.split('.')))
if numpy_version >= [1,10,0]:
    broadcast_to = np.broadcast_to
else:
    from sdeint._broadcast import broadcast_to


s = np.random.randint(2**16)
print('Testing using random seed %d' % s)
np.random.seed(s)

N = 10000
h = 0.002
m = 8


def test_Imr_Jmr_identities():
    """
	- Created by Philip Schwedler, 05/2021
	- Test the relations given in Mrongowius and Roessler (2021) equations (4)-(8)
	"""
    dW = deltaW(N, m, h).reshape((N, m, 1))
    Atilde, I = Imr(dW, h)
    M = m*(m-1)//2
	
	# Check if output dimensions
    assert(Atilde.shape == (N, M, 1) and I.shape == (N, m, m))
    Im = broadcast_to(np.eye(m), (N, m, m))
	
	# Check identity in [Mrongowius and Roessler (2021), p4, (4)]
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*Im))
	
    # Can get A from Atilde: [Mrongowius and Roessler (2021), p7, equation between (27) and (28)]
    Ims = broadcast_to(np.eye(m*m), (N, m*m, m*m))
    Pm = broadcast_to(_P(m), (N, m*m, m*m))
    Km = broadcast_to(_K(m), (N, M, m*m))
    A = _unvec(_dot(_dot((Ims - Pm), _t(Km)), Atilde))
	
    # Now can test this A against the identities of [Mrongowius and Roessler (2021), p4, (5)-(7)]
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2.0*(I - A), _dot(dW, _t(dW)) - h*Im))
	
    # Tests for Stratonovich case
    Atilde, J = Jmr(dW, h)
    assert(Atilde.shape == (N, M, 1) and J.shape == (N, m, m))
    assert(np.allclose(J + _t(J), _dot(dW, _t(dW))))
    A = _unvec(_dot(_dot((Ims - Pm), _t(Km)), Atilde))
    assert(np.allclose(2.0*(J - A), _dot(dW, _t(dW))))
	
#test_Imr_Jmr_identities()