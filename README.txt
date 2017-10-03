Python Neural Genetic Hybrids -- Release 0.3
=================================================

This package provides the Python "pyneurgen" module, which contains several
classes for implementing grammatical evolution, a form of genetic programming,
and classes for neural networks.  These classes enable the creation of hybrid
models that can embody the strengths.

While neural networks can be adept at solving non-linear problems, some
problems remain beyond reach.  For example, a difficult search space can cause
suboptimal solutions to be reached.  Also, multiobjective problems become
extremely difficult, if not impossible.  With genetic algorithms, a more
thorough search can be made.

This latest version has additional features added to enable the construction
of recurrent neural networks easier.  Recurrent neural networks are used
with time series data, because the structure of the network enables a
memory of past events to be included in the process.

There is an additional helper class that configures such recurrent network
types as Elman Simple Recurrent Network, Jordan style recurrent networks, and
NARX (Non-Linear AutoRegressive with eXogenous inputs) recurrent networks. And,
there are instructions on making your own structures for your unique
requirements.


How to install this package
===========================

python setup.py install


Author and license information
==============================

Copyright (C) 2012 Don Smiley <ds@sidorof.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

See the LICENSE file included in this archive

