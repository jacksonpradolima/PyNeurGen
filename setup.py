#!/usr/bin/env python
from setuptools import setup
from setuptools import setup, find_packages

setup(
    name='pyneurgen',
    version='0.3.1',
    description='Python Neural Genetic Hybrids',
    author='Don Smiley',
    author_email='ds@sidorof.com',
    url='http://pyneurgen/sourceforge.net',
    packages=['pyneurgen'],
    package_dir={'pyneurgen': 'pyneurgen'},
    long_description="""
    This package provides the Python "pyneurgen" module, which contains several
    classes for implementing grammatical evolution, a form of genetic
    programming, and classes for neural networks.  These classes enable the
    creation of hybrid models that can embody the strengths of grammatical
    evolution and neural networks.

    While neural networks can be adept at solving non-linear problems, some
    problems remain beyond reach.  For example, a difficult search space can
    cause suboptimal solutions to be reached.  Also, multiobjective problems
    become extremely difficult, if not impossible.  With genetic algorithms, a
    more thorough search can be made.""",

    keywords    = 'grammatical evolution programming neural networks genetic algorithms',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Adaptive Technologies',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ],
    license='GPL',
      )
