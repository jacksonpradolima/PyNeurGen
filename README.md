# Python Neural Genetic Algorithm Hybrids (PyNeurGen)

This is a fork of the PyNeurGen based on release 0.3.1. The original site is http://pyneurgen.sourceforge.net/ with the source code at https://sourceforge.net/projects/pyneurgen/

# About

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


# How to install this package

```shell
python setup.py install
```

or

```shell
sudo pip install git+https://github.com/jacksonpradolima/pyneurgen.git@master
```

# Author

Copyright (C) 2012 Don Smiley <ds@sidorof.com>

