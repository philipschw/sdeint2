# Copyright 2015 Matthew J. Aburn
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


"""
Run this script to plot the true solution against an approximated solution
from each integration algorithm, for several exactly solvable test systems
"""

from test_integrate_extension import *

t = Test_KP4446()
t.plot()

t = Test_KP4459()
t.plot()

t = Test_KPS445()
t.plot()

t = Test_R74()
t.plot()