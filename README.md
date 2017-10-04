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

# Documentation

https://jacksonpradolima.github.io/PyNeurGen/


# How to install this package

```shell
python setup.py install
```

or

```shell
sudo pip install git+https://github.com/jacksonpradolima/pyneurgen.git@master
```

# Change Log

pyneurgen-0.3.1 Minor change made to the MANIFEST file that interfered with
                installation on Windows machines.

pyneurgen-0.3   Extensive testing has been implemented, although there is
                room for more. Unit testing is now somewhere over 90%.

                An attempt is made to formalize constants to clean up the
                code.

                Validation testing was added to neural nets.

                The demo programs were updated slightly.

                A bug when loading saved neural nets with recurrent features
                was fixed.

                Some bugs in grammatical evolution were fixed.
                Queuing/threading and garbage collection was removed.  Queing
                never worked very well, and it is not clear that garbage collection
                was much help either, so it is gone.

                In grammatical evolution, the completion process has been
                streamlined. In addition, there is a new feature for building
                families of solutions and using your custom built fitness
                landscape function to evaluate stopping.

                There was a bug in the cross-over function spotted by Franco in
                Argentina (blamaeda@gmail.com) whom I wish to thank.

                There were some bugs corrected in the fitness functions in the
                min and max functions in fitness lists.  Unit testing coverage
                is essentially 100% on fitness functions.  FitnessLinearRanking
                was streamlined for inputs.

                There is a first pass at logging, but unwieldy.

                With python 2.6, there is math.isnan, which replaces a
                homemade function.  Other utility functions were removed
                as useless abstractions.


pyneurgen-0.2   Updates primarily related to recurrent networks.

                In attempt to enable an easier use of recurrent features, a
                new set of classes has been implemented.

                NeuralNet.init_layers no longer takes copy_levels as a
                parameter. Rather, it accepts along with the number of nodes
                for input, hidden and output layers, classes derived from a
                class called RecurrentConfig, which modifies the network
                structure to embody recurrence. Using the feature, a class
                such as JordanRecurrent, ElmanSimpleRecurrent, or NARXRecurrent
                can be passed in and the appropriate structure configured.

                The RecurrentConfig class can easily be subclassed for other
                custom types.  As before, the copy nodes can be easily be added
                manually to network structures to achieve other kinds of
                structures.

                CopyNode now has some additional features to enable the above
                classes.  In the prior version, the Copy node value was always
                replaced by the source node value.  Now, a weighting system of
                existing_weight and incoming weight is used to modify the
                incoming values and/or retain part of the existing value.

                For flexibility, the copy node can take the source node's
                value, or activation, and have its activation type, independent
                of the source node's activation.

                Test modules are slightly more complete.

pyneurgen-0.1   Initial version

# To Do

Immediate / Near Future

Samples:
    More samples would be helpful to express different applications of the
    modules.  Also, additional samples could highlight various features
    currently available.

Testing:
    Unit tests is much more complete than it was (90%?), but more can be done.

Flesh out logging 

# Author

Don Smiley <ds@sidorof.com>

# Author's message

I have benefitted from a number of great resources.  I should particularly note the book Biologically Inspired Algorithms for Financial Modelling by Anthony Brabazon and Michael O'Neill where I read about the technique 'grammatical evolution'.  I particularly like the simplicity for mapping genotypes to programming applications, coupled with ease of controlling the
level of complexity necessary accomplish the task.  There are a number of other applications for grammatical evolution that would great to implement as well.

I should also note the book Adaptive Learning of Polynomial Networks by Nikolay Y. Nikolaev and Hitoshi Iba.  While I would like to support polynomial networks in this package, it was not possible in the initial version.  This book has a very lucid discussion of fitness landscapes.  My fitness objects benefitted greatly from their discussions of fitness functions.

Any mistakes in implementation are my own.

Don Smiley
May, 2008


