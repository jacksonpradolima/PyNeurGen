import unittest

from copy import deepcopy

from pyneurgen.fitness import FitnessList, Fitness, Replacement, ReplacementDeleteWorst
from pyneurgen.fitness import FitnessElites, MAX, MIN, CENTER, FITNESS_TYPES
from pyneurgen.grammatical_evolution import GrammaticalEvolution, STOPPING_MAX_GEN
from pyneurgen.grammatical_evolution import STOPPING_FITNESS_LANDSCAPE

from pyneurgen.grammatical_evolution import DEFAULT_CROSSOVER_RATE
from pyneurgen.grammatical_evolution import DEFAULT_CHILDEREN_PER_CROSSOVER
from pyneurgen.grammatical_evolution import DEFAULT_MUTATION_TYPE
from pyneurgen.grammatical_evolution import DEFAULT_MUTATION_RATE
from pyneurgen.grammatical_evolution import DEFAULT_MAX_FITNESS_RATE
from pyneurgen.grammatical_evolution import DEFAULT_WRAP
from pyneurgen.grammatical_evolution import DEFAULT_EXTEND_GENOTYPE
from pyneurgen.grammatical_evolution import DEFAULT_START_GENE_LENGTH
from pyneurgen.grammatical_evolution import DEFAULT_MAX_GENE_LENGTH
from pyneurgen.grammatical_evolution import DEFAULT_MAX_PROGRAM_LENGTH
from pyneurgen.grammatical_evolution import DEFAULT_FITNESS_FAIL
from pyneurgen.grammatical_evolution import DEFAULT_MAINTAIN_HISTORY
from pyneurgen.grammatical_evolution import DEFAULT_TIMEOUTS

from pyneurgen.genotypes import MUT_TYPE_M, MUT_TYPE_S


class TestGrammaticalEvolution(unittest.TestCase):
    """
    This class tests the GrammaticalEvolution class.

    """
    def setUp(self):

        self.ges = GrammaticalEvolution()
        self.ges.set_genotype_length(10)
        self.ges.set_population_size(5)
        self.ges.set_max_program_length(200)

        self.ges.set_bnf(''.join([
            '<S>        ::=',
            'a = <VALUE1>\n',
            'b = <VALUE2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)\n',
            '<VALUE1>     ::= -1 | 2 | 0 \n',
            '<VALUE2>     ::=  1 | 2 | 3 \n']))

        self.ges.create_genotypes()

        self.ges.set_fitness_type(MAX)
        #   build a fake history
        value = .5
        for generation in range(3):
            #   Pretend that the genotypes have run
            count = 0
            for gene in self.ges.population:
                gene._fitness = float(count) + value
                self.ges.fitness_list[count][0] = float(count) + value
                self.ges.fitness_list[count][0] = float(count) + value
                count += 1

            self.ges._history.append(deepcopy(self.ges.fitness_list))

    def test_class_init__(self):
        """
        This function tests the initialization of the class.

        """

        ges = GrammaticalEvolution()
        self.assertEqual(None, ges.stopping_criteria[STOPPING_MAX_GEN])
        self.assertEqual(None,
                    ges.stopping_criteria[STOPPING_FITNESS_LANDSCAPE])
        self.assertEqual(DEFAULT_CROSSOVER_RATE, ges._crossover_rate)

        self.assertEqual(
            DEFAULT_CHILDEREN_PER_CROSSOVER, ges._children_per_crossover)
        self.assertEqual(DEFAULT_MUTATION_TYPE, ges._mutation_type)
        self.assertEqual(DEFAULT_MUTATION_RATE, ges._mutation_rate)
        self.assertEqual(DEFAULT_MAX_FITNESS_RATE, ges._max_fitness_rate)

        #   Parameters for phenotype creation
        self.assertEqual(DEFAULT_WRAP, ges._wrap)
        self.assertEqual(DEFAULT_EXTEND_GENOTYPE, ges._extend_genotype)
        self.assertEqual(DEFAULT_START_GENE_LENGTH, ges._start_gene_length)
        self.assertEqual(DEFAULT_MAX_PROGRAM_LENGTH, ges._max_gene_length)
        self.assertEqual(DEFAULT_MAX_PROGRAM_LENGTH, ges._max_program_length)

        #   Parameters for overall process
        self.assertEqual(0, ges._generation)
        self.assertEqual(FitnessList(CENTER), ges.fitness_list)
        self.assertEqual(DEFAULT_FITNESS_FAIL, ges._fitness_fail)
        self.assertEqual(DEFAULT_MAINTAIN_HISTORY, ges._maintain_history)
        self.assertEqual(DEFAULT_TIMEOUTS, ges._timeouts)

        #   Parameters used during runtime
        self.assertEqual(None, ges.current_g)
        self.assertEqual([], ges._fitness_selections)
        self.assertEqual([], ges._replacement_selections)

        self.assertEqual({}, ges.bnf)
        self.assertEqual(0, ges._population_size)
        self.assertEqual([], ges.population)

        self.assertEqual([], ges._history)

    def test_set_population_size(self):
        """
        This function tests setting the population size.

        """

        self.ges.set_population_size(1000)
        self.assertEqual(1000L, self.ges._population_size)
        self.assertEqual(1000L, len(self.ges.fitness_list))

        self.assertRaises(ValueError, self.ges.set_population_size, 0)
        self.assertRaises(ValueError, self.ges.set_population_size, -1)

    def test_get_population_size(self):
        """
        This function tests getting the population size.

        """

        self.ges._population_size = 1000
        self.assertEqual(1000L, self.ges.get_population_size())

    def test_set_genotype_length(self):
        """
        This function tests setting the genotype length.

        """

        self.ges.set_genotype_length(10, 1000)
        self.assertEqual(10, self.ges._start_gene_length)
        self.assertEqual(1000, self.ges._max_gene_length)

        self.ges.set_genotype_length(10, None)
        self.assertEqual(10, self.ges._start_gene_length)
        self.assertEqual(10, self.ges._max_gene_length)

        self.assertRaises(ValueError, self.ges.set_genotype_length, 1000, 10)
        self.assertRaises(ValueError, self.ges.set_genotype_length, -1, 10)
        self.assertRaises(ValueError, self.ges.set_genotype_length, 10, -10)

    def test_get_genotype_length(self):
        """
        This function test getting the genotype length.

        """

        self.ges._start_gene_length = 1000
        self.ges._max_gene_length = 1500
        self.assertEqual((1000L, 1500L), self.ges.get_genotype_length())

    def test_set_extend_genotype(self):
        """
        This function tests setting the genotype extend flag.

        """

        self.ges.set_extend_genotype(True)
        self.assertEqual(True, self.ges._extend_genotype)

        self.ges.set_extend_genotype(False)
        self.assertEqual(False, self.ges._extend_genotype)

        self.assertRaises(ValueError, self.ges.set_extend_genotype, 'nottrue')

    def test_get_extend_genotype(self):
        """
        This function tests getting the genotype extend flag.

        """

        self.ges._extend_genotype = True
        self.assertEqual(True, self.ges.get_extend_genotype())

        self.ges._extend_genotype = False
        self.assertEqual(False, self.ges.get_extend_genotype())

    def test_set_wrap(self):
        """
        This function tests setting the wrap flag.

        """
        self.ges.set_wrap(True)
        self.assertEqual(True, self.ges._wrap)

        self.ges.set_wrap(False)
        self.assertEqual(False, self.ges._wrap)

        self.assertRaises(ValueError, self.ges.set_wrap, 'nottrue')

    def test_get_wrap(self):
        """
        This function tests getting the wrap flag.

        """

        self.ges._wrap = True
        self.assertEqual(True, self.ges.get_wrap())

        self.ges._wrap = False
        self.assertEqual(False, self.ges._wrap)

    def test_set_bnf(self):

        sample_bnf = ''.join([
            '<S>        ::=',
            'a = <VALUE1>\n',
            'b = <VALUE2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)\n',
            '<VALUE1>     ::= -1 | 2 | 0 \n',
            '<VALUE2>     ::=  1 | 2 | 3 \n'])

        result_bnf = {'<S>' : [''.join([
            'a = <VALUE1>\n',
            'b = <VALUE2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)'])],
            '<VALUE1>' : ['-1', '2', '0'],
            '<VALUE2>' : ['1', '2', '3']}


        self.ges.set_bnf(sample_bnf)

        self.assertEqual(result_bnf['<VALUE1>'], self.ges.bnf['<VALUE1>'])
        self.assertEqual(result_bnf['<VALUE1>'], self.ges.bnf['<VALUE1>'])

        self.assertEqual(result_bnf['<S>'][0], self.ges.bnf['<S>'][0])
        self.assertEqual(result_bnf['<S>'][0], self.ges.bnf['<S>'][0])
        self.assertEqual(result_bnf, self.ges.bnf)


    def test_get_bnf(self):
        """
        This function test getting a BNF.

        """
        sample_bnf = ''.join([
            '<S>        ::=',
            'a = <VALUE1>\n',
            'b = <VALUE2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)\n',
            '<VALUE1>     ::= -1 | 2 | 0 \n',
            '<VALUE2>     ::=  1 | 2 | 3 \n'])

        result_bnf = {'<S>' : [''.join([
            'a = <VALUE1>\n',
            'b = <VALUE2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)'])],
            '<VALUE1>' : ['-1', '2', '0'],
            '<VALUE2>' : ['1', '2', '3']}

        self.ges.set_bnf(sample_bnf)
        self.assertEqual(result_bnf, self.ges.bnf)

        self.assertEqual(result_bnf, self.ges.get_bnf())

    def test_set_maintain_history(self):
        """
        This function tests setting the maintain fitness list history flag.

        """

        self.ges.set_maintain_history(True)
        self.assertEqual(True, self.ges._maintain_history)

        self.ges.set_maintain_history(False)
        self.assertEqual(False, self.ges._maintain_history)

        self.assertRaises(ValueError, self.ges.set_maintain_history, 'nottrue')

    def test_get_maintain_history(self):
        """
        This function tests getting the maintain fitness list history flag.

        """

        self.ges._maintain_history = True
        self.assertEqual(True, self.ges.get_maintain_history())

        self.ges._maintain_history = False
        self.assertEqual(False, self.ges.get_maintain_history())

    def test_set_max_program_length(self):
        """
        This function tests setting the maximum program length.

        """

        self.ges.set_max_program_length(1000)
        self.assertEqual(1000, self.ges._max_program_length)

        self.assertRaises(ValueError, self.ges.set_max_program_length, -1)

    def test_get_max_program_length(self):
        """
        This function tests getting the maximum program length.

        """

        self.ges._maintain_history = True
        self.assertEqual(True, self.ges.get_maintain_history())

        self.ges._maintain_history = False
        self.assertEqual(False, self.ges.get_maintain_history())

    def test_set_fitness_fail(self):
        """
        This function tests setting the default fitness value for failure.

        """

        self.ges.set_fitness_fail(-9999.0)
        self.assertEqual(-9999.0, self.ges._fitness_fail)

        self.ges.set_fitness_fail(-9999)
        self.assertEqual(-9999.0, self.ges._fitness_fail)

        self.assertRaises(ValueError, self.ges.set_fitness_fail, 'notfloat')

    def test_get_fitness_fail(self):
        """
        This function tests getting the default fitness value for failure.

        """

        self.ges._fitness_fail = -9999.0
        self.assertEqual(-9999.0, self.ges.get_fitness_fail())

    def test_set_mutation_type(self):
        """
        This function tests setting the default fitness value for failure.

        """

        self.ges.set_mutation_type(MUT_TYPE_S)
        self.assertEqual(MUT_TYPE_S, self.ges._mutation_type)

        self.ges.set_mutation_type(MUT_TYPE_M)
        self.assertEqual(MUT_TYPE_M, self.ges._mutation_type)

        self.assertRaises(ValueError, self.ges.set_mutation_type, 'wrongtype')

    def test_get_mutation_type(self):
        """
        This function tests getting the default fitness value for failure.

        """

        self.ges._mutation_type = MUT_TYPE_S
        self.assertEqual(MUT_TYPE_S, self.ges.get_mutation_type())

    def test_set_mut_cross_rate(self):
        """
        This function tests the ability to set mutation and cross-over rates.

        """

        self.ges.set_mutation_rate(.5)
        self.assertEqual(0.5, self.ges._mutation_rate)

        #   boundaries
        self.ges.set_mutation_rate(1.0)
        self.assertEqual(1.0, self.ges._mutation_rate)

        self.ges.set_mutation_rate(0.0)
        self.assertEqual(0.0, self.ges._mutation_rate)

        self.assertRaises(ValueError, self.ges.set_mutation_rate, -.01)
        self.assertRaises(ValueError, self.ges.set_mutation_rate, 1.01)

        #   crossover rate
        self.ges.set_crossover_rate(.5)
        self.assertEqual(0.5, self.ges._crossover_rate)

        #   boundaries
        self.ges.set_crossover_rate(1.0)
        self.assertEqual(1.0, self.ges._crossover_rate)

        self.ges.set_crossover_rate(0.0)
        self.assertEqual(0.0, self.ges._crossover_rate)

        self.assertRaises(ValueError, self.ges.set_crossover_rate, -.01)
        self.assertRaises(ValueError, self.ges.set_crossover_rate, 1.01)

    def test_get_mut_cross_rate(self):
        """
        This function tests the ability to get mutation and cross-over rates.

        """

        self.ges._mutation_rate = 0.5
        self.assertEqual(0.5, self.ges.get_mutation_rate())

        self.ges._crossover = 0.2
        self.assertEqual(0.2, self.ges.get_crossover_rate())

    def test_set_children_per_crossover(self):
        """
        This function tests setting the number of children to be generated
        upon crossover.

        """

        self.ges.set_children_per_crossover(1)
        self.assertEqual(1, self.ges._children_per_crossover)

        self.ges.set_children_per_crossover(2)
        self.assertEqual(2, self.ges._children_per_crossover)

        self.assertRaises(ValueError,
                    self.ges.set_children_per_crossover, 0)
        self.assertRaises(ValueError,
                    self.ges.set_children_per_crossover, 3)

    def test_get_children_per_crossover(self):
        """
        This function tests getting the number of children per crossover.

        """

        self.ges._children_per_crossover = 1
        self.assertEqual(1, self.ges.get_children_per_crossover())

    def test_set_max_generations(self):
        """
        This function tests setting the maximum number of generations.

        """

        self.ges.set_max_generations(500)
        self.assertEqual(500, self.ges.stopping_criteria[STOPPING_MAX_GEN])

    def test_get_max_generations(self):
        """
        This function tests getting the maximum number of generations.

        """

        self.ges.stopping_criteria[STOPPING_MAX_GEN] = 500
        self.assertEqual(500, self.ges.get_max_generations())

    def test_set_fitness_type(self):
        """
        This function tests setting a fitness type.  It also tests the option
        of setting a target value at the same time.

        """

        self.ges.set_fitness_type(CENTER)
        fitl = FitnessList(CENTER)
        self.assertEqual(fitl.get_fitness_type(),
                            self.ges.fitness_list.get_fitness_type())

        fitl = FitnessList(MIN)
        self.ges.set_fitness_type(MIN)
        self.assertEqual(fitl.get_fitness_type(),
                            self.ges.fitness_list.get_fitness_type())

        fitl = FitnessList(MAX)
        self.ges.set_fitness_type(MAX)
        self.assertEqual(fitl.get_fitness_type(),
                            self.ges.fitness_list.get_fitness_type())

        self.assertRaises(ValueError, self.ges.set_fitness_type, "wrong")

        #   Include target value
        fitl = FitnessList(CENTER, 1.0)
        self.ges.set_fitness_type(CENTER, 1.0)
        self.assertEqual(fitl.get_fitness_type(),
                            self.ges.fitness_list.get_fitness_type())
        self.assertEqual(1.0, self.ges.fitness_list._target_value)

        self.assertRaises(ValueError,
                        self.ges.set_fitness_type, CENTER, "test")

    def test_get_fitness_type(self):
        """
        This function tests getting the maximum number of generations.

        """

        self.ges.fitness_list._fitness_type = "min"
        self.assertEqual("min", self.ges.get_fitness_type())

    def test_set_max_fitness_rate(self):
        """
        This function tests the ability to set the maximum fitness rate.

        """

        self.ges.set_max_fitness_rate(.5)
        self.assertEqual(0.5, self.ges._max_fitness_rate)

        #   boundaries
        self.ges.set_max_fitness_rate(1.0)
        self.assertEqual(1.0, self.ges._max_fitness_rate)

        self.ges.set_max_fitness_rate(0.0)
        self.assertEqual(0.0, self.ges._max_fitness_rate)

        self.assertRaises(ValueError, self.ges.set_max_fitness_rate, -.01)
        self.assertRaises(ValueError, self.ges.set_max_fitness_rate, 1.01)

    def test_get_max_fitness_rate(self):
        """
        This function tests the ability to get the maximum fitness rate.

        """

        self.ges._max_fitness_rate = 0.5
        self.assertEqual(0.5, self.ges.get_max_fitness_rate())

    def test_set_fitness_selections(self):
        """
        This function tests setting a fitness selection class.

        """

        self.ges.set_fitness_selections(Fitness(FitnessList('center')))
        self.assertEqual(1, len(self.ges._fitness_selections))

        self.ges.set_fitness_selections(Fitness(FitnessList('center')))
        self.assertEqual(2, len(self.ges._fitness_selections))

        self.ges._fitness_selections = []

        self.ges.set_fitness_selections(Fitness(FitnessList('center')),
                                        Fitness(FitnessList('center')))
        self.assertEqual(2, len(self.ges._fitness_selections))

        self.ges._fitness_selections = []

        self.assertRaises(ValueError, self.ges.set_fitness_selections, "wrong")

    def test_set_replacement_selections(self):
        """
        This function tests setting a replacement selection class.

        """

        self.ges.set_replacement_selections(Replacement(FitnessList('center')))
        self.assertEqual(1, len(self.ges._replacement_selections))

        self.ges.set_replacement_selections(Replacement(FitnessList('center')))
        self.assertEqual(2, len(self.ges._replacement_selections))

        self.ges._replacement_selections = []

        self.ges.set_replacement_selections(Replacement(FitnessList('center')),
                                        Replacement(FitnessList('center')))
        self.assertEqual(2, len(self.ges._replacement_selections))

        self.ges._replacement_selections = []

        self.assertRaises(ValueError,
                            self.ges.set_replacement_selections, "wrong")

    def test_get_fitness_history(self):
        """
        This function tests getting the fitness history.

        """

        self.ges.set_fitness_type('max')
        getfh = self.ges.get_fitness_history
        self.assertEqual([4.5, 4.5, 4.5], getfh('best_value'))
        self.assertEqual([2.5, 2.5, 2.5], getfh('mean'))
        self.assertEqual([0.5, 0.5, 0.5], getfh('min_value'))
        self.assertEqual([4.5, 4.5, 4.5], getfh('max_value'))
        self.assertEqual([0.5, 0.5, 0.5], getfh('worst_value'))
        self.assertEqual([0, 0, 0], getfh('min_member'))
        self.assertEqual([4, 4, 4], getfh('max_member'))
        self.assertEqual([4, 4, 4], getfh('best_member'))
        self.assertEqual([0, 0, 0], getfh('worst_member'))

    def test_get_best_member(self):
        """
        This function tests getting the best member of the fitness list.

        """

        self.assertEqual(self.ges.population[4], self.ges.get_best_member())


    def test_get_worst_member(self):
        """
        This function tests getting the worst member of the fitness list.

        """

        self.assertEqual(self.ges.population[0], self.ges.get_worst_member())

    def test_set_timeouts(self):
        """
        This function tests setting the timeouts.

        """

        self.ges.set_timeouts(5, 10)
        self.assertEqual([5, 10], self.ges._timeouts)
        self.assertRaises(ValueError, self.ges.set_timeouts, -5, 10)
        self.assertRaises(ValueError, self.ges.set_timeouts, 5, -10)
        self.assertRaises(ValueError, self.ges.set_timeouts, 5, 'ten')
        self.assertRaises(ValueError, self.ges.set_timeouts, 'five', 10)

    def test_get_timeouts(self):
        """
        This function tests getting the timeouts.

        """

        self.ges._timeouts = [5, 10]
        self.assertEqual([5, 10], self.ges.get_timeouts())

    def test_compute_fitness(self):
        """
        This function tests the compute process.

        """

        fitness_fail = self.ges._fitness_fail

        self.ges._compute_fitness()

        #   Is each fitness value in the gene
        for gene in self.ges.population:
            self.assertNotEqual(fitness_fail, gene.get_fitness())

        #   Is each fitness value in the fitness list
        for fitness_value, member_no in self.ges.fitness_list:
            self.assertNotEqual(fitness_fail, fitness_value)

    def test_run(self):
        """
        This function tests running grammatical_evolution start to finish.

        """

        self.ges.set_fitness_type(MAX, 20.0)
        self.ges.set_max_generations(5)

        self.ges._history = []
        self.ges.set_maintain_history(True)
        self.ges.run()

        #   number of generations saved.
        self.ges.set_fitness_type('max')
        getfh = self.ges.get_fitness_history

        #   Not really sure how to prove a run.
        print 'best_value', getfh('best_value')
        print 'mean', getfh('mean')
        print 'min_value', getfh('min_value')
        print 'max_value', getfh('max_value')
        print 'worst_value', getfh('worst_value')
        print 'min_member', getfh('min_member')
        print 'max_member', getfh('max_member')
        print 'best_member', getfh('best_member')
        print 'worst_member', getfh('worst_member')

        self.assertEqual(5, self.ges._generation)

        #   this is 6, because the counter starts at 0
        self.assertEqual(6, len(getfh('best_value')))

    def test_create_genotypes(self):
        """
        This function test the building of genotypes and passing along the
        parameters to each individual.

        """

        self.ges.population = []
        self.ges.create_genotypes()

        self.assertEqual(self.ges._population_size, len(self.ges.population))

        for gene in self.ges.population:
            self.assertEqual(gene._gene_length,
                    self.ges._start_gene_length)
            self.assertEqual(gene._max_gene_length,
                    self.ges._max_gene_length)
            self.assertEqual(gene.local_bnf['<member_no>'],
                [gene.member_no])
            self.assertEqual(gene._max_program_length,
                    self.ges._max_program_length)
            self.assertEqual(gene._fitness,
                    self.ges._fitness_fail)
            self.assertEqual(gene._extend_genotype,
                    self.ges._extend_genotype)
            self.assertEqual(gene._timeouts,
                    self.ges._timeouts)
            self.assertEqual(gene._wrap,
                    self.ges._wrap)


    def test_perform_endcycle(self):

        print "perform_endcycle not yet tested"

    def test_evaluate_fitness(self):

        #   test whether select from fitness selections works

        self.assertEqual(.5, self.ges._max_fitness_rate)

        self.ges.set_fitness_selections(
            FitnessElites(self.ges.fitness_list, .1))

        pool = self.ges._evaluate_fitness()
        self.assertEqual(1, len(pool))

        #   test whether max fitness rate works
        self.ges.set_fitness_selections(
            FitnessElites(self.ges.fitness_list, .75))

        pool = self.ges._evaluate_fitness()
        self.assertEqual(3, len(pool))

    def test_perform_crossovers(self):

        #   Make a fitness pool
        flist = self.ges.population[2:4]

        self.assertEqual(2, self.ges._children_per_crossover)
        clist = self.ges._perform_crossovers(flist)
        self.assertEqual(2, len(clist))

        #   Does it round fitness pool properly?
        flist = self.ges.population[1:4]
        clist = self.ges._perform_crossovers(flist)
        self.assertEqual(2, len(clist))

        #   Change children per cross over
        self.ges.set_children_per_crossover(1)
        clist = self.ges._perform_crossovers(flist)
        self.assertEqual(1, len(clist))

    def test_crossover(self):
        """
        This function tests the crossover process from the level of two genes.

        """

        parent1 = self.ges.population[0]
        parent2 = self.ges.population[1]

        (child1, child2) = self.ges._crossover(parent1, parent2)


        pgene1 = parent1.binary_gene
        pgene2 = parent2.binary_gene

        cgene1 = child1.binary_gene
        cgene2 = child2.binary_gene

        self.assertEqual(len(pgene1), len(cgene1))
        self.assertEqual(len(pgene2), len(cgene2))

        #   was there any crossover
        self.assertNotEqual(pgene1, cgene1)
        self.assertNotEqual(pgene2, cgene2)

        self.assertNotEqual(pgene1, cgene2)
        self.assertNotEqual(pgene2, cgene1)

       #   compare cross points
        for crosspoint in range(len(pgene1)):
            if pgene1[crosspoint] != cgene1[crosspoint]:
                break

        self.assertEqual(len(pgene1[:crosspoint]),
                        len(cgene1[:crosspoint]))

        self.assertEqual(len(pgene2[:crosspoint]),
                        len(cgene2[:crosspoint]))

        self.assertEqual(len(pgene1[crosspoint:]),
                        len(cgene2[crosspoint:]))
        self.assertEqual(len(pgene2[crosspoint:]),
                        len(cgene1[crosspoint:]))


    def test_crossover_function(self):
        """
        This function tests the crossover process at the level of two strings
        of binary numbers.777777777777

        """

        crosspoint = 10
        child1_binary = '000000000011111'
        child2_binary = '111111111100000'

        self.assertEqual(('000000000000000', '111111111111111'),
                            self.ges._crossover_function(child1_binary,
                                            child2_binary, crosspoint))
    def test_perform_mutations(self):
        """
        This function tests the process that performs mutations of the genes.

        """

        #   single
        self.ges.population = []
        self.ges.set_population_size(100)
        self.ges.set_mutation_type('s')
        self.ges.set_mutation_rate(.2)
        self.ges.create_genotypes()
        self.ges._compute_fitness()

        pop = deepcopy(self.ges.population)
        new_pop = self.ges.population
        self.ges._perform_mutations(new_pop)

        diffs = 0
        for gene in pop:
            if new_pop[gene.member_no].binary_gene != gene.binary_gene:
                diffs += 1
        self.assertNotEqual(0, diffs)

        #   multiple
        self.ges.population = []
        self.ges.set_population_size(100)
        self.ges.set_mutation_type('m')
        self.ges.set_mutation_rate(.01)
        self.ges.create_genotypes()
        self.ges._compute_fitness()

        pop = deepcopy(self.ges.population)
        new_pop = self.ges.population
        self.ges._perform_mutations(new_pop)

        diffs = 0
        for gene in pop:
            if new_pop[gene.member_no].binary_gene != gene.binary_gene:
                diffs += 1
        self.assertNotEqual(0, diffs)

    def test_perform_replacements(self):

        #   Existing fitness list
        #[[0.5, 0], [1.5, 1], [2.5, 2], [3.5, 3], [4.5, 4]]

        #   make some replacement selections
        #   This should replace the member_no 4
        self.ges.fitness_list.set_fitness_type(MIN)
        self.ges.set_replacement_selections(
                        ReplacementDeleteWorst(self.ges.fitness_list, 1))

        #   make a fitness pool
        new_member = deepcopy(self.ges.population[2])
        new_member.local_bnf['<IAMNEW>'] = ['yes i am']
        orig_member_no = new_member.member_no
        fitness_pool = [new_member]

        #   do the replacements
        self.ges._perform_replacements(fitness_pool)

        #   did replacement take place
        self.assertEqual('yes i am',
                self.ges.population[4].local_bnf['<IAMNEW>'][0])

        #   was the local bnf for each replaced member update for new member no
        self.assertEqual(4, self.ges.population[4].member_no)

        #   was the generation number bumped up by one
        self.assertEqual(1, self.ges.population[4]._generation)


    def test_continue_processing(self):

        #   Max
        self.ges.fitness_list.set_fitness_type(MAX)
        self.ges.fitness_list.set_target_value(5.0)
        self.ges._generation = 0
        self.assertEqual(True, self.ges._continue_processing())

        self.ges.fitness_list.set_target_value(4.0)
        self.assertEqual(False, self.ges._continue_processing())

        #   Min
        self.ges.fitness_list.set_fitness_type(MIN)
        self.assertEqual(False, self.ges._continue_processing())
        self.ges.fitness_list.set_target_value(-4.0)
        self.assertEqual(True, self.ges._continue_processing())

        #   Center
        self.ges.fitness_list.set_fitness_type(CENTER)
        self.ges.fitness_list.set_target_value(-.3)
        self.assertEqual(True, self.ges._continue_processing())

        self.ges.fitness_list.set_target_value(-4.0)
        self.assertEqual(True, self.ges._continue_processing())

        #   Maximum generations
        self.ges.stopping_criteria[STOPPING_MAX_GEN] = 1
        self.assertEqual(True, self.ges._continue_processing())

        self.ges._generation = 1
        self.assertEqual(False, self.ges._continue_processing())

        self.ges._generation = 0

        #   Fitness Landscape function, two pretend functions
        def fit_landscape(fitness_list):
            return True

        def fit_landscape1(fitness_list):
            return False

        self.ges.stopping_criteria[STOPPING_FITNESS_LANDSCAPE] = fit_landscape
        self.assertEqual(True, self.ges._continue_processing())

        self.ges.stopping_criteria[STOPPING_FITNESS_LANDSCAPE] = fit_landscape1
        self.assertEqual(False, self.ges._continue_processing())






if __name__ == '__main__':
    unittest.main()
