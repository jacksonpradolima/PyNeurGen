#!/usr/bin/env python
#
#   Copyright (C) 2012  Don Smiley  ds@sidorof.com

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

#   See the LICENSE file included in this archive
#

"""
This module implements the components for grammatical evolution.

Logging has been added, but is currently poorly integrated.
Hoped for course for implementation:
    Turn on/off logging from user program
    Granularity of logging choices
        ex. be able to specify logging for certain reasons why a genotype fails
            such as timeout, or syntax errors

    Generally speaking, the intent will be to be able to turn on debugging
    for certain types of problems without being flooded in other areas.


"""

from datetime import datetime
from copy import deepcopy
import logging
from random import randint

from pyneurgen.genotypes import Genotype, MUT_TYPE_M, MUT_TYPE_S
from pyneurgen.fitness import FitnessList, Fitness, Replacement
from pyneurgen.fitness import CENTER, MAX, MIN


#   Constants
STATEMENT_FORMAT = '<S'
STOPPING_MAX_GEN = 'max_generations'
STOPPING_FITNESS_LANDSCAPE = 'fitness_landscape'

#   Default values
DEFAULT_CROSSOVER_RATE = 0.2
DEFAULT_CHILDEREN_PER_CROSSOVER = 2
DEFAULT_MUTATION_TYPE = 's'
DEFAULT_MUTATION_RATE = 0.02
DEFAULT_MAX_FITNESS_RATE = .5
DEFAULT_WRAP = True
DEFAULT_EXTEND_GENOTYPE = True
DEFAULT_START_GENE_LENGTH = None
DEFAULT_MAX_GENE_LENGTH = None
DEFAULT_MAX_PROGRAM_LENGTH = None
DEFAULT_FITNESS_FAIL = -1000.0
DEFAULT_MAINTAIN_HISTORY = True
DEFAULT_TIMEOUTS = [20, 3600]

DEFAULT_LOG_FILE = 'pyneurgen.log'

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=DEFAULT_LOG_FILE,
                    level=logging.INFO)


class GrammaticalEvolution(object):
    """
    This class comprises the overall process of generating genotypes,
    expressing the genes as programs using grammer and evalating the fitness of
    those members.
    """

    def __init__(self):
        """
        Because there are a number of parameters to specify, there are no
        specific variables that are initialized within __init__.

        There is a formidable number of default variables specified in this
        function however.

        """

        #   Parameters for changing generations
        self.stopping_criteria = {
                STOPPING_MAX_GEN: None,
                STOPPING_FITNESS_LANDSCAPE: None}
        self._crossover_rate = DEFAULT_CROSSOVER_RATE
        self._children_per_crossover = DEFAULT_CHILDEREN_PER_CROSSOVER
        self._mutation_type = DEFAULT_MUTATION_TYPE
        self._mutation_rate = DEFAULT_MUTATION_RATE
        self._max_fitness_rate = DEFAULT_MAX_FITNESS_RATE

        #   Parameters for phenotype creation
        self._wrap = DEFAULT_WRAP
        self._extend_genotype = DEFAULT_EXTEND_GENOTYPE
        self._start_gene_length = DEFAULT_START_GENE_LENGTH
        self._max_gene_length = DEFAULT_MAX_PROGRAM_LENGTH
        self._max_program_length = DEFAULT_MAX_PROGRAM_LENGTH

        #   Parameters for overall process
        self._generation = 0
        self.fitness_list = FitnessList(CENTER)
        self._fitness_fail = DEFAULT_FITNESS_FAIL
        self._maintain_history = DEFAULT_MAINTAIN_HISTORY
        self._timeouts = DEFAULT_TIMEOUTS

        #   Parameters used during runtime
        self.current_g = None
        self._fitness_selections = []
        self._replacement_selections = []

        self.bnf = {}
        self._population_size = 0
        self.population = []

        self._history = []

    def set_population_size(self, size):
        """
        This function sets the total number of genotypes in the population.
        This program uses a fixed size population.

        """

        size = long(size)
        if isinstance(size, long) and size > 0:
            self._population_size = size
            i = len(self.fitness_list)
            while i < size:
                self.fitness_list.append([0.0, i])
                i += 1
        else:
            raise ValueError("""
                population size, %s, must be a long above 0""" % (size))

    def get_population_size(self):
        """
        This function returns total number of genotypes in the
        population.

        """

        return self._population_size

    def set_genotype_length(self, start_gene_length,
                                    max_gene_length=None):
        """
        This function sets the initial size of the binary genotype.  An
        optional max_gene_length can be entered as well.  This permits the
        genotype to grow during the mapping process of the genotype to a
        program.  The lengths are the length of the decimal genotypes, which
        are therefor 8 times longer the binary genotypes created.

        """

        if max_gene_length is None:
            max_gene_length = start_gene_length

        start_gene_length = long(start_gene_length)
        max_gene_length = long(max_gene_length)
        if not isinstance(start_gene_length, long):
            raise ValueError("start_gene_length, %s, must be a long" % (
                                                    start_gene_length))
        if start_gene_length < 0:
            raise ValueError("start_gene_length, %s, must be above 0" % (
                                                    start_gene_length))
        if not isinstance(max_gene_length, long):
            raise ValueError("max_gene_length, %s, must be a long" % (
                                                    max_gene_length))
        if max_gene_length < 0:
            raise ValueError("max_gene_length, %s, must be above 0" % (
                                                    max_gene_length))
        if start_gene_length > max_gene_length:
            raise ValueError("""max_gene_length, %s, cannot be smaller
                than start_gene_length%s""" % (
                    max_gene_length, start_gene_length))

        self._start_gene_length = start_gene_length
        self._max_gene_length = max_gene_length

    def get_genotype_length(self):
        """
        This function returns a tuple with the length the initial genotype and
        the maximum genotype length permitted.

        """

        return (self._start_gene_length, self._max_gene_length)

    def set_extend_genotype(self, true_false):
        """
        This function sets whether the genotype is extended during the gene
        mapping process.

        """

        if isinstance(true_false, bool):
            self._extend_genotype = true_false
        else:
            raise ValueError("Extend genotype must be True or False")

    def get_extend_genotype(self):
        """
        This function returns whether the genotype is extended during the gene
        mapping process.

        """

        return self._extend_genotype

    def set_wrap(self, true_false):
        """
        This function sets whether the genotype is wrapped during the gene
        mapping process.  Wrapping would occur in the iterative process of
        getting the next codon is the basis for the variable selection process.
        If wrapped, when all the codons in the genotype are exhausted, the
        position marker is wrapped around to the first codon in the sequence
        and goes on.

        """

        if isinstance(true_false, bool):
            self._wrap = true_false
        else:
            raise ValueError("Wrap must be True or False")

    def get_wrap(self):
        """
        This function returns whether the genotype is wrapped during the gene
        mapping process.  Wrapping would occur in the iterative process of
        getting the next codon is the basis for the variable selection process.
        If wrapped, when all the codons in the genotype are exhausted, the
        position marker is wrapped around to the first codon in the sequence
        and goes on.

        """

        return self._wrap

    def set_bnf(self, bnf):
        """
        This function parses up a bnf and builds a dictionary. The incoming
        format is designed to follow a format of:  <key> ::= value1 | value2
        \n. The following lines can also hold additional values to accommodate
        longer choices.

        In addition, a set of statements are marked with a key
        starting with "<S".  These are treated differently in that spaces are
        not automatically stripped from the front.  This enables python
        oriented white space to be honored.

        """

        def strip_spaces(key, values):
            """
            This removes white space unless it is a statement.
            """
            if key.startswith(STATEMENT_FORMAT):
                values = [value.rstrip()
                    for value in values.split('|') if value]
            else:
                values = [value.strip()
                    for value in values.split('|') if value]

            return values

        bnf_dict = {}
        for item in bnf.split('\n'):
            if item.find('::=') >= 0:
                key, values = item.split('::=')
                key = key.strip()
                bnf_dict[key] = strip_spaces(key, values)
            elif item:
                values = bnf_dict[key]
                values.extend(strip_spaces(key, item))
                if key.startswith(STATEMENT_FORMAT):
                    #   Convert statements back to string
                    values = ['\n'.join(values)]
                bnf_dict[key] = values
            else:
                #   blank line
                pass
        self.bnf = bnf_dict

    def get_bnf(self):
        """
        This function returns the Backus Naur form of variables that are used
        to map the genotypes to the generated programs.

        """

        return self.bnf

    def set_maintain_history(self, true_false):
        """
        This function sets a flag to maintain a history of fitness_lists.

        """
        if isinstance(true_false, bool):
            self._maintain_history = true_false
        else:
            raise ValueError("Maintain history must be True or False")

    def get_maintain_history(self):
        """
        This function returns a flag indicating whether the fitness list is
        retained for each generation.

        """

        return self._maintain_history

    def set_max_program_length(self, max_program_length):
        """
        This function sets the maximum length that a program can attain before
        the genotype is declared a failure.

        """

        errmsg1 = """The maximum program length, %s must be an long value
                    """ % (max_program_length)
        errmsg2 = """The maximum program length, %s must be greater than 0
                    """ % (max_program_length)
        max_program_length = long(max_program_length)
        if not isinstance(max_program_length, long):
            raise ValueError(errmsg1)
        if max_program_length < 0:
            raise ValueError(errmsg2)

        self._max_program_length = max_program_length

    def get_max_program_length(self):
        """
        This function gets the maximum length that a program can attain before
        the genotype is declared a failure.

        """

        return self._max_program_length

    def set_fitness_fail(self, fitness_fail):
        """
        This function sets the fitness fail value that will be applied to
        fitness functions that are deemed failure.  Failure would be programs
        that fail due to overflows, or programs that grow to greater than
        maximum program length, syntax failures, or other reasons.

        """

        errmsg = """The fitness_fail, %s must be a float value
                    """ % (fitness_fail)
        #   coerce if possible
        fitness_fail = float(fitness_fail)
        if not isinstance(fitness_fail, float):
            raise ValueError(errmsg)

        self._fitness_fail = fitness_fail

    def get_fitness_fail(self):
        """
        This function returns the value of fitness if the program is a failure.

        """

        return self._fitness_fail

    def set_mutation_type(self, mutation_type):
        """
        This function sets the mutation type.  The choices are s(ingle),
        m(ultiple).  If the choice is "s", then the mutation rate is applied
        as a choice of whether to alter 1 bit on a gene or not.  If the choice
        is "m", then the process applies the rate as the probability that a bit
        will be changed as it walks the gene.  In short, "s", means that if the
        gene is mutated, it will take place once.  Otherwise, the gene could be
        mutated multiple times.

        """

        errmsg = "The mutation type must be either '%s' or '%s'." % (
                                                    MUT_TYPE_S, MUT_TYPE_M)
        if mutation_type not in [MUT_TYPE_M, MUT_TYPE_S]:
            raise ValueError(errmsg)

        self._mutation_type = mutation_type

    def get_mutation_type(self):
        """
        This function returns the mutation type.  See set_mutation_type for a
        more complete explanation.

        """

        return self._mutation_type

    def set_mutation_rate(self, mutation_rate):
        """
        This function sets the mutation rate that will be applied to members
        selected into the fitness pool and to newly generated children.  Note
        that the mutation rate should be vastly different depending upon the
        mutation type that you have selected.  If the mutation type is 's',
        then the rate is the probability that the genotype will be mutated.  If
        the mutation type is 'm', then the rate is the probability that the any
        given bit in the genotype will be altered.  Because of that, the
        mutation rate should be significantly lower than the rate used with a
        mutation type of 's'.

        """

        errmsg = """The mutation rate, %s must be a float value
                    from 0.0 to 1.0""" % (mutation_rate)
        if not isinstance(mutation_rate, float):
            raise ValueError(errmsg)
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(errmsg)

        self._mutation_rate = mutation_rate

    def get_mutation_rate(self):
        """
        This function gets the mutation rate that will be applied to members
        selected into the fitness pool and to newly generated children.  Note
        that the mutation rate should be vastly different depending upon the
        mutation type that you have selected.  If the mutation type is 's',
        then the rate is the probability that the genotype will be mutated.  If
        the mutation type is 'm', then the rate is the probability that the any
        given bit in the genotype will be altered.  Because of that, the
        mutation rate should be significantly lower than the rate used with a
        mutation type of 's'.

        """

        return self._mutation_rate

    def set_crossover_rate(self, crossover_rate):
        """
        This function sets the probablity that will be
        applied to members selected into the fitness pool.

        """

        errmsg = """The crossover rate, %s must be a float value
                    from 0.0 to 1.0""" % (crossover_rate)
        if not isinstance(crossover_rate, float):
            raise ValueError(errmsg)
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError(errmsg)

        self._crossover_rate = crossover_rate

    def get_crossover_rate(self):
        """
        This function gets the probablity that will be applied to members
        selected into the fitness pool.

        """

        return self._crossover_rate

    def set_children_per_crossover(self, children_per_crossover):
        """
        This function sets the number of children that will generated from two
        parents.  The choice is one or two.

        """

        if children_per_crossover not in [1, 2]:
            raise ValueError(
                "The children per crossovermust be either 1 or 2.")
        self._children_per_crossover = children_per_crossover

    def get_children_per_crossover(self):
        """
        This function gets the number of children that will generated from two
        parents.

        """

        return self._children_per_crossover

    def set_max_generations(self, generations):
        """
        This function sets the maximum number of generations that will be run.

        """

        if isinstance(generations, int) and generations >= 0:
            self.stopping_criteria[STOPPING_MAX_GEN] = generations
        else:
            raise ValueError("""
                generations, %s, must be an int 0 or greater""" % (
                                                                generations))

    def get_max_generations(self):
        """
        This function gets the maximum number of generations that will be run.

        """

        return self.stopping_criteria[STOPPING_MAX_GEN]

    def set_fitness_type(self, fitness_type, target_value=0.0):
        """
        This function sets whether the objective is to achieve as large a
        fitness value possible, small, or hit a target_value.  Therefor the
        choices are 'max', 'min', or 'center'.  If center is used, then a
        target value should be entered as well.  For example, suppose that you
        wanted to hit a target somewhere near zero.  Setting the target_value
        at .001 would cause the process to complete if a fitness value achieved
        and absolute value of .001 or less.

        """

        self.fitness_list.set_fitness_type(fitness_type)
        self.fitness_list.set_target_value(target_value)

    def get_fitness_type(self):
        """
        This function gets whether the objective is to achieve as large a
        fitness value possible, small, or hit a target_value.  Therefor the
        choices are 'max', 'min', or 'center'.  If center is used, then a
        target value should be entered as well.  For example, suppose that you
        wanted to hit a target somewhere near zero.  Setting the target_value
        at .001 would cause the process to complete if a fitness value achieved
        .001 or less.

        """

        return self.fitness_list.get_fitness_type()

    def set_max_fitness_rate(self, max_fitness_rate):
        """
        This function sets a maximum for the number of genotypes that can be
        put in the fitness pool.  Since some fitness selection approaches can
        have a varying number selected, and since multiple selection approaches
        can be applied consequentially, there needs to be an ultimate limit on
        the total number.  The max fitness rate must be a value greater than
        zero and less than 1.0.

        """

        errmsg = """The max fitness rate, %s must be a float value
                    from 0.0 to 1.0""" % (max_fitness_rate)
        if not isinstance(max_fitness_rate, float):
            raise ValueError(errmsg)
        if not (0.0 <= max_fitness_rate <= 1.0):
            raise ValueError(errmsg)

        self._max_fitness_rate = max_fitness_rate

    def get_max_fitness_rate(self):
        """
        This function gets a maximum for the number of genotypes that can be
        in the fitness pool.  Since some fitness selection approaches can have
        a varying number selected, and since multiple selection approaches can
        be applied consequentially, there needs to be an ultimate limit on the
        total number.  The max fitness rate must be a value greater than zero
        and less than 1.0.

        """

        return self._max_fitness_rate

    def set_fitness_selections(self, *params):
        """
        This function loads the fitness selections that are to be used to
        determine genotypes worthy of continuing to the next generation.  There
        can be multiple selections, such as elites and tournaments.  See the
        section Fitness Selection for further information.

        """

        for fitness_selection in params:
            if isinstance(fitness_selection, Fitness):
                self._fitness_selections.append(fitness_selection)
            else:
                raise ValueError("Invalid fitness selection")

    def set_replacement_selections(self, *params):
        """
        This function loads the replacement selections that are used to
        determine genotypes are to be replaced.  Basically, it is the grim
        reaper. Multiple replacement types can be loaded to meet the criteria.
        The number replaced is governed by the fitness selection functions to
        ensure that the population number stays constant.

        """

        for replacement_selection in params:
            if isinstance(replacement_selection, Replacement):
                self._replacement_selections.append(replacement_selection)
            else:
                raise ValueError("Invalid replacement selection")

    def get_fitness_history(self, statistic='best_value'):
        """
        This funcion returns a list of values that represent historical values
        from the fitness history.  While there is a default value of
        'best_value', other values are 'mean', 'min_value', 'max_value',
        'worst_value', 'min_member', 'max_member', 'best_member', and
        'worst_member'. The order is from oldest to newest.

        """

        hist_list = []
        for fitness_list in self._history:
            hist_list.append(fitness_list.__getattribute__(statistic)())
        return hist_list

    def get_best_member(self):
        """
        This function returns the member that it is most fit according to the
        fitness list.  Accordingly, it is only functional after at least one
        generation has been completed.

        """

        return self.population[self.fitness_list.best_member()]

    def get_worst_member(self):
        """
        This function returns the member that it is least fit according to the
        fitness list.  Accordingly, it is only functional after at least one
        generation has been completed.

        """

        return self.population[self.fitness_list.worst_member()]

    def set_timeouts(self, preprogram, program):
        """
        This function sets the number of seconds that the program waits until
        declaring that the process is a runaway and cuts it off.  During the
        mapping process against the preprogram, due to random chance a function
        can be calling another function, which calls another, until the process
        becomes so convoluted that the resulting program will be completely
        useless. While the total length of a program can be guide to its
        uselessnes as well, this is another way to handle it. Since variables
        can also be generated during the running of the program there is a
        second variable for the running program. Clearly, the second value must
        be in harmony with the nature of the program that you are actually
        running. Otherwise, you will be cutting of your program prematurely.
        Note that the second timeout is checked only if the running program
        requests an additional variable.  Otherwise, it will not be triggered.

        """

        if isinstance(preprogram, int) and preprogram >= 0:
            self._timeouts[0] = preprogram
        else:
            raise ValueError("""
                timeout, %s, must be an int 0 or above""" % (preprogram))

        if isinstance(program, int) and program >= 0:
            self._timeouts[1] = program
        else:
            raise ValueError("""
                timeout, %s, must be an int 0 or above""" % (program))

    def get_timeouts(self):
        """
        This function returns the number of seconds that must elapse before
        the mapping process cuts off the process and declares that the genotype
        is a failure.  It returns a tuple for the number of seconds for the
        preprogram and the program itself.

        """

        return self._timeouts

    def _compute_fitness(self):
        """
        This function runs the process of computing fitness functions for each
        genotype and calculating the fitness function.

        """

        for gene in self.population:
            starttime = datetime.now()
            gene._generation = self._generation
            logging.debug("Starting member G %s: %s at %s" % (
                self._generation, gene.member_no,
                starttime.strftime('%m/%d/%y %H:%M')))

            #print "Starting member G %s: %s at %s" % (
                #self._generation, gene.member_no,
                #starttime.strftime('%m/%d/%y %H:%M'))
            gene.starttime = starttime
            self.current_g = gene
            gene.compute_fitness()

            logging.debug("fitness=%s" % (gene.get_fitness()))
            self.fitness_list[gene.member_no][0] = gene.get_fitness()

    def run(self, starting_generation=0):
        """
        Once the parameters have all been set governing the course of the
        evolutionary processing, this function starts the process running.  It
        will continue until it the completion criteria have been set.

        """

        logging.info("started run")
        self._generation = starting_generation
        while True:
            self._compute_fitness()
            if self._maintain_history:
                self._history.append(deepcopy(self.fitness_list))

            if self._continue_processing():
                self._perform_endcycle()

                logging.info("Finished generation: %s Max generations: %s" % (
                            self._generation,
                            self.get_max_generations()))
                logging.info(' '.join(
                            ["best_value: %s" % (
                                self.fitness_list.best_value()),
                            "median: %s" % (self.fitness_list.median()),
                            "mean: %s" % (self.fitness_list.mean())]))
                #temp -- remove this
                gene = self.population[self.fitness_list.best_member()]
                program = gene.get_program()
                logging.info(program)

                #logging.debug("stddev= %s" % self.fitness_list.stddev())
                self._generation += 1
            else:
                break

        logging.info(
            "completed run: generations: %s, best member:%s fitness: %s" % (
                    self._generation,
                    self.fitness_list.best_member(),
                    self.fitness_list.best_value()))

        return self.fitness_list.best_member()

    def create_genotypes(self):
        """
        This function creates a genotype using the input parameters for each
        member of the population, and transfers operating parameters to the
        genotype for running the fitness functions.

        """

        member_no = 0
        while member_no < self._population_size:
            gene = Genotype(self._start_gene_length,
                        self._max_gene_length,
                        member_no)
            #   a local copy is made because variables
            #   can be saved within the local_bnf
            gene.local_bnf = deepcopy(self.bnf)
            gene.local_bnf['<member_no>'] = [gene.member_no]
            gene._max_program_length = self._max_program_length
            gene._fitness = self._fitness_fail
            gene._fitness_fail = self._fitness_fail
            gene._extend_genotype = self._extend_genotype
            gene._timeouts = self._timeouts
            gene._wrap = self._wrap
            self.population.append(gene)
            member_no += 1

    def _perform_endcycle(self):
        """
        This function runs after each member of the population has computed
        their fitness function.  Then, the fitness selection objects will
        evaluate those members according to their respective criteria and
        develop a pool of members that will potentially survive to the next
        generation. Crossovers will take place from that pool and each member
        will be subject to the possibility of mutatuting.  Finally, a
        replacement process will find which members should be replaced. The
        fitness pool will then replace those members.

        """

        fitness_pool = self._evaluate_fitness()
        child_list = self._perform_crossovers(fitness_pool)

        fitness_pool.extend(child_list)
        self._perform_mutations(fitness_pool)
        self._perform_replacements(fitness_pool)

    def _evaluate_fitness(self):
        """
        This function evaluates the fitness of the members in the light of the
        fitness criteria functions.  It returns a list of members that will be
        used for crossovers and mutations.

        """

        flist = []
        total = int(round(
            self._max_fitness_rate * float(self._population_size)))
        count = 0
        for fsel in self._fitness_selections:
            fsel.set_fitness_list(self.fitness_list)
            for i in fsel.select():
                flist.append(i)
                count += 1
                if count == total:
                    #   Done
                    break

        flist1 = []
        for member_no in flist:
            flist1.append(deepcopy(self.population[member_no]))

        return flist1

    def _perform_crossovers(self, flist):
        """
        This function accepts a list of genotypes that are to be crossed.  The
        list is processed two at a time, and a child list holding the offspring
        is returned.  The _children_per_crossover indicator governs whether two
        children are produced or one.

        """

        child_list = []
        length = len(flist)
        if length % 2 == 1:
            length -= 1
        if length >= 2:
            for i in xrange(0, length, 2):
                parent1 = flist[i]
                parent2 = flist[i + 1]

                child1, child2 = self._crossover(parent1, parent2)
                if self._children_per_crossover == 2:
                    child_list.append(child1)
                    child_list.append(child2)
                else:
                    child_list.append(child1)

        return child_list

    def _crossover(self, parent1, parent2):
        """
        This function accepts two parents, randomly selects which is parent1
        and which is parent2.  Then, executes the crossover, and returns two
        children.

        """

        if not isinstance(parent1, Genotype):
            raise ValueError("Parent1 is not a genotype")
        if not isinstance(parent2, Genotype):
            raise ValueError("Parent2 is not a genotype")

        if randint(0, 1):
            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)
        else:
            child1 = deepcopy(parent2)
            child2 = deepcopy(parent1)

        child1_binary = child1.binary_gene
        child2_binary = child2.binary_gene

        minlength = min(len(child1_binary), len(child2_binary))
        crosspoint = randint(2, minlength - 2)

        child1_binary, child2_binary = self._crossover_function(
            child1.binary_gene, child2.binary_gene, crosspoint)

        child1.set_binary_gene(child1_binary)
        child1.generate_decimal_gene()
        child2.set_binary_gene(child2_binary)
        child2.generate_decimal_gene()

        return (child1, child2)

    @staticmethod
    def _crossover_function(child1_binary, child2_binary, crosspoint):
        """
        This function performs the actual crossover of material at a random
        point.

        I gratefully acknowlege Franco from Argentina (blamaeda@gmail.com) for
        the fix to my previous version of this code.

        """

        child1_binary, child2_binary = child1_binary[0:crosspoint] + \
                                    child2_binary[crosspoint:], \
                                    child2_binary[0:crosspoint] + \
                                    child1_binary[crosspoint:]

        return (child1_binary, child2_binary)

    def _perform_mutations(self, mlist):
        """
        This functions accepts a list of genotypes that are subject to
        mutation.  Each genotype is then put at risk for mutation and may or
        may not be mutated.

        """

        for gene in mlist:
            gene.mutate(self._mutation_rate, self._mutation_type)

    def _perform_replacements(self, fitness_pool):
        """
        This function accepts a list of members that will replace lesser
        performers.  The replacement process then applies the fitness pool to
        the population.

        """

        position = 0
        for rsel in self._replacement_selections:
            rsel.set_fitness_list(self.fitness_list)

            for replaced_no in rsel.select():
                replaced_g = self.population[replaced_no]
                if position < len(fitness_pool):
                    new_g = fitness_pool[position]
                    new_g.member_no = replaced_g.member_no
                    new_g._generation = self._generation + 1

                    #   update local bnf
                    new_g.local_bnf['<member_no>'] = [new_g.member_no]

                    self.population[new_g.member_no] = new_g
                    position += 1
                else:
                    break

    def _continue_processing(self):
        """
        This function, using the criteria for ending the evolutionary process
        after each generation, returns a flag of whether to continue or not.

        """

        status = True
        fitl = self.fitness_list

        #   check max generations first
        if self.stopping_criteria[STOPPING_MAX_GEN] is not None:
            if self.stopping_criteria[STOPPING_MAX_GEN] <= self._generation:
                logging.info("stopping processing due to max generation")
                return False

        #   check target value
        if fitl.get_target_value() is not None:
            if fitl.get_fitness_type() == MAX:
                if fitl.best_value() >= fitl.get_target_value():
                    logging.info(' '.join([
                    "stopping processing due to",
                    "best value, %s, better than target value, %s" % (
                    fitl.best_value(), fitl.get_target_value())]))
                    return False
            elif fitl.get_fitness_type() == MIN:
                if fitl.best_value() <= fitl.get_target_value():
                    logging.info(' '.join([
                    "stopping processing due to",
                    "best value, %s, better than target value, %s" % (
                    fitl.best_value(), fitl.get_target_value())]))
                    return False
            elif fitl.get_fitness_type() == CENTER:
                if fitl.best_value() <= fitl.get_target_value():
                    logging.info(' '.join([
                    "stopping processing due to",
                    "best value, %s, better than target value, %s" % (
                    fitl.best_value(), fitl.get_target_value())]))
                    return False

        #   Finally check if there is a stopping function
        if self.stopping_criteria[STOPPING_FITNESS_LANDSCAPE] is not None:
            status = self.stopping_criteria[STOPPING_FITNESS_LANDSCAPE](
                                                        self.fitness_list)
            logging.info(' '.join([
                                "stopping processing due to",
                                "fitness landscape being reached."]))

        return status
