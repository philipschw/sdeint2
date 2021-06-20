# sdeint2

An extension of the [sdeint](https://github.com/mattja/sdeint) package.

## Overview
sdeint2 is a collection of numerical algorithms for integrating Ito and Stratonovich stochastic ordinary differential equations (SODEs).

## Specific Algorithms:

`itoMilstein(f, G, dxG, y0, tspan)`: the Milstein algorithm for Ito equations.
``itoTamedEuler(f, G, y0, tspan)``: the drift-tamed Euler-Maruyama order 0.5 strong algorithm for Ito equations with globally one-sided Lipschitz condition on the drift
``itoSRIC2(f, G, y0, tspan)``: the Rößler2010 order 1.0 strong Stochastic Runge-Kutta algorithm SRIC2 for Ito equations with commutative noise.
``itoSRIC2(f, [g1,...,gm], y0, tspan)``: as above, with G matrix given as a separate function for each column (gives speedup for large m or complicated G).
``itoSRID2(f, G, y0, tspan)``: the Rößler2010 order 1.5 strong Stochastic Runge-Kutta algorithm SRID2 for Ito equations with diagonal noise.
``itoSRA3(f, G, y0, tspan)``: the Rößler2010 order 1.5 strong Stochastic Runge-Kutta algorithm SRA3 for Ito equations with additive noise.
``itoSRI2W1(f, G, y0, tspan)``: the Rößler2010 order 1.5 strong Stochastic Runge-Kutta algorithm SRI2W1 for Ito equations with scalar noise.
``itoRI5(f, G, y0, tspan)``: the Rößler2010 order 2.0 weak Stochastic Runge-Kutta algorithm RI5 for Ito equations.
``itoRI5(f, [g1,...,gm], y0, tspan)``: as above, with G matrix given as a separate function for each column (gives speedup for large m or complicated G).
``stratSRA3(f, G, y0, tspan)``: the Rößler2010 order 1.5 strong Stochastic Runge-Kutta algorithm SRS2 for Stratonovich equations with additive noise.
For more information and advanced options see the documentation for each function.

##utility functions:

Repeated integrals by the method of Mrongowius and Roessler (2021):
``Imr(dW, h, n=5)``: Approximate repeated Ito integrals.
``Jmr(dW, h, n=5)``: Approximate repeated Stratonovich integrals.


## References for these algorithms:

``itoMilstein``: 
G. N. Milstein (1974), Approximate integration of stochastic differential equations
``itoSRIC2, itoSRID2, itoSRA3, stratSRA3, itoSRI2W1``: 
A. Rößler (2010) Runge-Kutta Methods for the Strong Approximation of Solutions of Stochastic Differential Equations
``itoRI5``:
A. Roessler (2009) Second Order Runge-Kutta Methods For Ito Stochastic Differential Equations
``itoTamedEuler``:
Hutzenthaler, Jentzen and Kloeden (2012) Strong Convergence of an explicit numerical method for SDEs with nonglobally Lipschitz continuous coefficients
``Imr, Jmr``:
J. Mrongowius and A. Rössler (2021) On the Approximation and Simulation of Iterated Stochastic Integrals and the Corresponding Levy Areas in Terms of a Multidimensional Brownian Motion

## LICENCSE
Copyright 2021 Philip Schwedler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

This code is partially based on the source code of the Python Package
[sdeint](https://github.com/mattja/sdeint) written by Matthew J. Aburn & Yoav Ram.
