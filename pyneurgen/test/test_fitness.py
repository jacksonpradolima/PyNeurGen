import unittest

from pyneurgen.fitness import FitnessList, Fitness, Selection, Tournament
from pyneurgen.fitness import FitnessProportionate, FitnessTournament
from pyneurgen.fitness import FitnessElites, FitnessLinearRanking
from pyneurgen.fitness import FitnessTruncationRanking
from pyneurgen.fitness import Replacement, ReplacementDeleteWorst
from pyneurgen.fitness import ReplacementTournament
from pyneurgen.fitness import MAX, MIN, CENTER, FITNESS_TYPES
from pyneurgen.fitness import SCALING_LINEAR, SCALING_TRUNC, SCALING_EXPONENTIAL
from pyneurgen.fitness import SCALING_LOG, SCALING_TYPES


class TestFitnessList(unittest.TestCase):
    """
    This class tests the base class of fitness.
    """

    def setUp(self):
        self.fitness_list = FitnessList(MAX)

        self.fitness_list.append([3.0, 0])
        self.fitness_list.append([2.0, 1])
        self.fitness_list.append([5.0, 2])
        self.fitness_list.append([4.0, 3])
        self.fitness_list.append([1.0, 4])

    def test_class_init_(self):

        fitness_list = FitnessList(MAX)

        #   Is it a list?
        self.assertEqual(True, isinstance(fitness_list, list))

        #   Does the fitness type get set?
        self.assertEqual(MAX, fitness_list._fitness_type)

        #   Does the target_value get set?
        self.assertEqual(0.0, fitness_list._target_value)

        fitness_list = FitnessList(MAX, .5)
        self.assertAlmostEqual(0.5, fitness_list._target_value)

    def test_set_fitness_type(self):

        self.fitness_list.set_fitness_type(MIN)
        self.assertEqual(MIN, self.fitness_list._fitness_type)

    def test_get_fitness_type(self):

        self.assertEqual(MAX, self.fitness_list.get_fitness_type())

        self.fitness_list._fitness_type = MIN
        self.assertEqual(MIN, self.fitness_list.get_fitness_type())

    def test_set_target_value(self):

        self.fitness_list.set_target_value(0.3)
        self.assertEqual(0.3, self.fitness_list._target_value)

    def test_get_target_value(self):
        self.fitness_list._target_value = .45
        self.assertAlmostEqual(.45, self.fitness_list.get_target_value())

    def test_min_value(self):
        self.assertEqual(1.0, self.fitness_list.min_value())

    def test_max_value(self):
        self.assertEqual(5.0, self.fitness_list.max_value())

    def test_best_value(self):
        self.assertEqual(5.0, self.fitness_list.best_value())

        self.fitness_list.set_fitness_type(MIN)
        self.assertEqual(1.0, self.fitness_list.best_value())

        self.fitness_list.set_fitness_type(CENTER)
        self.fitness_list.set_target_value(3.0)
        self.assertEqual(3.0, self.fitness_list.best_value())

    def test_worst_value(self):
        self.assertEqual(1.0, self.fitness_list.worst_value())

        self.fitness_list.set_fitness_type(MIN)
        self.assertEqual(5.0, self.fitness_list.worst_value())

        self.fitness_list.set_fitness_type(CENTER)
        self.fitness_list.set_target_value(3.0)
        self.assertEqual(1.0, self.fitness_list.worst_value())

    def test_min_member(self):
        self.assertEqual(4, self.fitness_list.min_member())

    def test_max_member(self):
        self.assertEqual(2, self.fitness_list.max_member())

    def test_best_member(self):
        #   max
        self.assertEqual(2, self.fitness_list.best_member())

        #   min
        self.fitness_list.set_fitness_type(MIN)
        self.assertEqual(4, self.fitness_list.best_member())

        #   center
        self.fitness_list.set_fitness_type(CENTER)
        self.assertEqual(4, self.fitness_list.best_member())

        #   check absolute value on center
        self.fitness_list.append([-.5, 5])
        self.assertEqual(5, self.fitness_list.best_member())

    def test_worst_member(self):
        #   max
        self.assertEqual(4, self.fitness_list.worst_member())

        #   min
        self.fitness_list.set_fitness_type(MIN)
        self.assertEqual(2, self.fitness_list.worst_member())

        #   center
        self.fitness_list.set_fitness_type(CENTER)
        self.assertEqual(2, self.fitness_list.worst_member())

        #   check absolute value on center
        self.fitness_list.append([-5.0, 5])
        self.assertEqual(5, self.fitness_list.worst_member())

    def test_mean(self):

        mean = (3.0 + 2.0 + 5.0 + 4.0 + 1.0) / 5
        self.assertAlmostEqual(mean, self.fitness_list.mean())

    def test_median(self):

        self.assertAlmostEqual(3.0, self.fitness_list.median())

    def test_stddev(self):
        self.assertAlmostEqual(1.5811388301, self.fitness_list.stddev())

    def test_sorted(self):

        #   max
        sorted_list = self.fitness_list.sorted()
        self.assertAlmostEqual(5.0, sorted_list[0][0])
        self.assertEqual(2, sorted_list[0][1])
        self.assertAlmostEqual(4.0, sorted_list[1][0])
        self.assertEqual(3, sorted_list[1][1])
        self.assertAlmostEqual(3.0, sorted_list[2][0])
        self.assertEqual(0, sorted_list[2][1])
        self.assertAlmostEqual(2.0, sorted_list[3][0])
        self.assertEqual(1, sorted_list[3][1])
        self.assertAlmostEqual(1.0, sorted_list[4][0])
        self.assertEqual(4, sorted_list[4][1])

        #   min
        self.fitness_list.set_fitness_type(MIN)
        sorted_list = self.fitness_list.sorted()

        self.assertAlmostEqual(1.0, sorted_list[0][0])
        self.assertEqual(4, sorted_list[0][1])
        self.assertAlmostEqual(2.0, sorted_list[1][0])
        self.assertEqual(1, sorted_list[1][1])
        self.assertAlmostEqual(3.0, sorted_list[2][0])
        self.assertEqual(0, sorted_list[2][1])
        self.assertAlmostEqual(4.0, sorted_list[3][0])
        self.assertEqual(3, sorted_list[3][1])
        self.assertAlmostEqual(5.0, sorted_list[4][0])
        self.assertEqual(2, sorted_list[4][1])

        #   center
        self.fitness_list.set_fitness_type(CENTER)
        self.fitness_list.append([-0.5, 5])
        sorted_list = self.fitness_list.sorted()

        self.assertAlmostEqual(-0.5, sorted_list[0][0])
        self.assertEqual(5, sorted_list[0][1])
        self.assertAlmostEqual(1.0, sorted_list[1][0])
        self.assertEqual(4, sorted_list[1][1])
        self.assertAlmostEqual(2.0, sorted_list[2][0])
        self.assertEqual(1, sorted_list[2][1])
        self.assertAlmostEqual(3.0, sorted_list[3][0])
        self.assertEqual(0, sorted_list[3][1])
        self.assertAlmostEqual(4.0, sorted_list[4][0])
        self.assertEqual(3, sorted_list[4][1])
        self.assertAlmostEqual(5.0, sorted_list[5][0])
        self.assertEqual(2, sorted_list[5][1])

class TestSelection(unittest.TestCase):

    def test_class_init_(self):

        sel_type = Selection([1.0, 2.0, 3.0, 4.0, 5.0])

        #   check default selection type
        self.assertEqual(MAX, sel_type._selection_type)

        #   check list
        self.assertEqual(True, isinstance(sel_type._selection_list, list))

        sel_type = Selection()
        self.assertEqual(None, sel_type._selection_list)

    def test_set_selection_list(self):

        sel_type = Selection()
        self.assertEqual(None, sel_type._selection_list)
        sel_type.set_selection_list([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(True, isinstance(sel_type._selection_list, list))

        #   test for fitness_list
        fitl = FitnessList(MAX)
        self.assertRaises(ValueError, sel_type.set_selection_list, fitl)

    def test_set_selection_type(self):

        sel_type = Selection()
        self.assertEqual(None, sel_type._selection_list)
        sel_type.set_selection_list([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(True, isinstance(sel_type._selection_list, list))

    def test_roulette_wheel(self):

        slist = [0.06666666666666667, 0.13333333333333333, 0.2,
                0.26666666666666666, 0.3333333333333333]
        sel_type = Selection(slist)

        for rand_position in sel_type._roulette_wheel(slist):
            self.assertGreaterEqual(5, rand_position)
            self.assertLessEqual(0, rand_position)

    def test_make_sort_list(self):

        slist = [0.06666666666666667, 0.13333333333333333, 0.2,
                0.26666666666666666, 0.3333333333333333]
        sel_type = Selection(slist)

        sorted_list = sel_type._make_sort_list()

        self.assertAlmostEqual(0.06666666666666667, sorted_list[0][0])
        self.assertAlmostEqual(0.13333333333333333, sorted_list[1][0])
        self.assertAlmostEqual(0.2, sorted_list[2][0])
        self.assertAlmostEqual(0.26666666666666666, sorted_list[3][0])
        self.assertAlmostEqual(0.3333333333333333, sorted_list[4][0])
        self.assertAlmostEqual(0, sorted_list[0][1])
        self.assertAlmostEqual(1, sorted_list[1][1])
        self.assertAlmostEqual(2, sorted_list[2][1])
        self.assertAlmostEqual(3, sorted_list[3][1])
        self.assertAlmostEqual(4, sorted_list[4][1])


class TestTournament(unittest.TestCase):

    def setUp(self):
        slist = [0.06666666666666667, 0.13333333333333333, 0.2,
                0.26666666666666666, 0.3333333333333333]

        self.sel_type = Tournament(selection_list=slist, tournament_size=2)

    def test_classinit_(self):

        self.assertEqual(2, self.sel_type._tournament_size)
        self.assertEqual(None, self.sel_type._minmax)

    def test_set_tournament_size(self):
        self.sel_type.set_tournament_size(3)
        self.assertEqual(3, self.sel_type._tournament_size)
        self.assertRaises(ValueError, self.sel_type.set_tournament_size, 10)
        self.assertRaises(ValueError, self.sel_type.set_tournament_size, 6)

    def test_set_minmax(self):

        self.sel_type._set_minmax(MIN)
        self.assertEqual(MIN, self.sel_type._minmax)

        self.sel_type._set_minmax(MAX)
        self.assertEqual(MAX, self.sel_type._minmax)

        self.assertRaises(ValueError, self.sel_type._set_minmax, 'WRONG')

    def test_select(self):
        #   This test should be revisited to prove:
        #       the selection pool was really the tournament_size
        #       the best member was really selected

        #   Min
        self.sel_type._set_minmax(MIN)

        #   Selections are 2 then 1
        count = 0
        for member in self.sel_type.select():
            count += 1

        self.assertEqual(count, len(self.sel_type._selection_list))

class TestFitness(unittest.TestCase):

    def test_classinit_(self):
        """
        Does it set the fitness_list?
        """

        fitness = FitnessList(MAX)
        fit = Fitness(fitness)

        self.assertEqual(fit._fitness_list, fitness)

    def test_set_fitness_list(self):

        #   Do MAX
        fitness = FitnessList(MAX)
        fit = Fitness(fitness)
        fit._fitness_list = None
        self.assertEqual(fit._fitness_list, None)
        fit.set_fitness_list(fitness)
        self.assertEqual(fit._fitness_list, fitness)

        #   Do CENTER selection_list converted to distance from target
        fitness = FitnessList(CENTER, .15)
        fitness.append([.5, 0])
        fitness.append([.25, 1])
        fitness.append([2.5, 2])
        fit = Fitness(fitness)
        fit._fitness_list = None
        self.assertEqual(fit._fitness_list, None)
        fit.set_fitness_list(fitness)
        self.assertEqual(fit._selection_list, [.35, .1, 2.35])

    def test_invert(self):
        fitness = FitnessList(MAX)
        fit = Fitness(fitness)

        self.assertAlmostEqual(.25, fit._invert(4.0))
        self.assertAlmostEqual(-.25, fit._invert(-4.0))
        self.assertAlmostEqual(2.0, fit._invert(0.5))

    def test_scale_list(self):

        #
        #   fitness type MAX, test select type MAX
        #
        fitness = FitnessList(MAX)
        fitness.extend([[.5, 0], [.25, 1], [2.5, 2]])

        fit = Fitness(fitness)
        fit.set_selection_type(MAX)
        fit._scale_list()

        #   not inverted
        self.assertAlmostEqual(.5, fit._selection_list[0])
        self.assertAlmostEqual(.25, fit._selection_list[1])
        self.assertAlmostEqual(2.5, fit._selection_list[2])

        #
        #   fitness type MAX, test select type MIN
        #
        fit.set_fitness_list(fitness)
        fit.set_selection_type(MIN)
        fit._scale_list()

        #   inverted
        self.assertAlmostEqual(2.0, fit._selection_list[0])
        self.assertAlmostEqual(4.0, fit._selection_list[1])
        self.assertAlmostEqual(0.4, fit._selection_list[2])

        #
        #   fitness type MIN, test select type MAX
        #
        fitness.set_fitness_type(MIN)
        fit.set_fitness_list(fitness)
        fit.set_selection_type(MAX)
        fit._scale_list()

        #   inverted
        self.assertAlmostEqual(2.0, fit._selection_list[0])
        self.assertAlmostEqual(4.0, fit._selection_list[1])
        self.assertAlmostEqual(0.4, fit._selection_list[2])

        #
        #   fitness type MIN, test select type MIN
        #
        fit.set_fitness_list(fitness)
        fit.set_selection_type(MIN)
        fit._scale_list()

        #   not inverted
        self.assertAlmostEqual(.5, fit._selection_list[0])
        self.assertAlmostEqual(.25, fit._selection_list[1])
        self.assertAlmostEqual(2.5, fit._selection_list[2])

        #
        #   fitness type CENTER, test select type MAX
        #
        fitness.set_fitness_type(CENTER)
        fitness.set_target_value(.75)
        fit.set_fitness_list(fitness)
        fit.set_selection_type(MAX)
        fit._scale_list()

        #   inverted
        self.assertAlmostEqual(4.0, fit._selection_list[0])
        self.assertAlmostEqual(2.0, fit._selection_list[1])
        self.assertAlmostEqual(0.5714285714, fit._selection_list[2])

        #
        #   fitness type CENTER, test select type MIN
        #
        fit.set_fitness_list(fitness)
        fit.set_selection_type(MIN)
        fit._scale_list()

        #   not inverted
        self.assertAlmostEqual(.25, fit._selection_list[0])
        self.assertAlmostEqual(.5, fit._selection_list[1])
        self.assertAlmostEqual(1.75, fit._selection_list[2])

    def test_make_prob_list(self):

        prob_list = Fitness._make_prob_list([.5, .25, 2.5])

        self.assertAlmostEqual(1.0, sum(prob_list))
        self.assertAlmostEqual(0.153846153846, prob_list[0])
        self.assertAlmostEqual(0.0769230769231, prob_list[1])
        self.assertAlmostEqual(0.769230769231, prob_list[2])

class TestFitnessProportionate(unittest.TestCase):
    def setUp(self):
        #   Check truncation
        self.fitness = FitnessList(MAX)
        self.fitness.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

    def test_class_init_(self):
        #   is the scaling type set
        #   does it check the number range
        fit = FitnessProportionate(self.fitness, SCALING_LINEAR)
        self.assertNotEqual(None, fit._scaling_type)

        #   add some negative numbers
        self.fitness.extend([[-.5, 3], [-.25, 4], [2.5, 2]])
        self.assertRaises(ValueError, FitnessProportionate, self.fitness,
                            SCALING_LINEAR)

    def test_set_scaling_type(self):
        fit = FitnessProportionate(self.fitness, SCALING_LINEAR)
        self.assertEqual(SCALING_LINEAR, fit._scaling_type)

    def test_check_minmax(self):
        self.fitness.extend([[.5, 3], [.25, 1], [-2.5, 2]])
        self.assertRaises(ValueError, FitnessProportionate, self.fitness,
                            SCALING_LINEAR)

    def test_select(self):
        fit = FitnessProportionate(self.fitness, SCALING_LINEAR)

        self.assertEqual(3, len([i for i in fit.select()]))

    def test_apply_prop_scaling(self):

        #   This scales the list according to the scaling type
        param = None
        #   Check linear
        fit = FitnessProportionate(self.fitness, SCALING_LINEAR)
        scaling_list = fit._apply_prop_scaling(param)
        self.assertAlmostEqual(1.0, sum(scaling_list))

        self.assertAlmostEqual(0.09375, scaling_list[0])
        self.assertAlmostEqual(0.15625, scaling_list[1])
        self.assertAlmostEqual(0.75, scaling_list[2])

        #   Check exponential
        fit = FitnessProportionate(self.fitness, SCALING_EXPONENTIAL)
        scaling_list = fit._apply_prop_scaling(param)
        self.assertAlmostEqual(1.0, sum(scaling_list))

        self.assertAlmostEqual(0.014754098360655738, scaling_list[0])
        self.assertAlmostEqual(0.040983606557377046, scaling_list[1])
        self.assertAlmostEqual(0.94426229508196724, scaling_list[2])

        fit = FitnessProportionate(self.fitness, SCALING_EXPONENTIAL)
        scaling_list = fit._apply_prop_scaling(param=1.5)
        self.assertAlmostEqual(1.0, sum(scaling_list))

        self.assertAlmostEqual(0.038791152234464166, scaling_list[0])
        self.assertAlmostEqual(0.083465270324597968, scaling_list[1])
        self.assertAlmostEqual(0.87774357744093778, scaling_list[2])

        #   Check log
        fit = FitnessProportionate(self.fitness, SCALING_LOG)
        scaling_list = fit._apply_prop_scaling(param)
        self.assertAlmostEqual(1.0, sum(scaling_list))

        self.assertAlmostEqual(0.10651459360996007, scaling_list[0])
        self.assertAlmostEqual(0.24070711137026513, scaling_list[1])
        self.assertAlmostEqual(0.6527782950197748, scaling_list[2])

        #   Check truncation
        fit = FitnessProportionate(self.fitness, SCALING_TRUNC)
        scaling_list = fit._apply_prop_scaling(param=2.0)
        self.assertAlmostEqual(1.0, sum(scaling_list))

        self.assertAlmostEqual(0.0, scaling_list[0])
        self.assertAlmostEqual(0.17241379310344829, scaling_list[1])
        self.assertAlmostEqual(0.82758620689655171, scaling_list[2])


class TestFitnessTournament(unittest.TestCase):

    def test_classinit_(self):

        fitness_list = FitnessList(MAX)
        fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

        fit = FitnessTournament(fitness_list)
        self.assertEqual(2, fit._tournament_size)

        fit = FitnessTournament(fitness_list, tournament_size=3)
        self.assertEqual(3, fit._tournament_size)

        self.assertRaises(ValueError, FitnessTournament, fitness_list,
                            tournament_size=4)


class TestFitnessElites(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

    def test_classinit_(self):
        self.assertRaises(ValueError, FitnessElites, self.fitness_list,
                            rate=1.1)
        self.assertRaises(ValueError, FitnessElites, self.fitness_list,
                            rate=-1.1)
        self.assertRaises(ValueError, FitnessElites, self.fitness_list,
                            rate=0.0)

        fit = FitnessElites(self.fitness_list, rate=0.1)
        self.assertAlmostEqual(0.1, fit._rate)
        self.assertAlmostEqual(MIN, fit._selection_type)

    def test_set_rate(self):
        fit = FitnessElites(self.fitness_list, rate=0.1)
        self.assertAlmostEqual(0.1, fit._rate)

    def test_select(self):
        fit = FitnessElites(self.fitness_list, rate=0.3333)
        count = len([i for i in fit.select()])
        self.assertEqual(1, count)

class TestFitnessLinearRanking(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

    def test_classinit_(self):

        fit = FitnessLinearRanking(self.fitness_list, .6)
        self.assertAlmostEqual( 0.6, fit._worstfactor)

    def test_set_worstfactor(self):
        fit = FitnessLinearRanking(self.fitness_list, .6)
        self.assertAlmostEqual( 0.6, fit._worstfactor)
        fit.set_worstfactor(.7)
        self.assertAlmostEqual(.7, fit._worstfactor)

        self.assertRaises(ValueError, fit.set_worstfactor, -.1)
        self.assertRaises(ValueError, fit.set_worstfactor, 2.1)

    def test_select(self):

        fit = FitnessLinearRanking(self.fitness_list, .6)
        count = len([i for i in fit.select()])
        self.assertEqual(3, count)

    def test_linear_ranking(self):
        pass
        fit = FitnessLinearRanking(self.fitness_list, .6)

        prob_list = fit._linear_ranking(3, .6)
        self.assertAlmostEqual(0.199999999, prob_list[0])
        self.assertAlmostEqual(0.333333333, prob_list[1])
        self.assertAlmostEqual(0.466666666, prob_list[2])

        self.assertLessEqual(prob_list[0], prob_list[1])
        self.assertLessEqual(prob_list[1], prob_list[2])

        self.assertAlmostEqual(1.0, sum(prob_list))

        prob_list = fit._linear_ranking(3, .2)
        self.assertAlmostEqual(0.06666667, prob_list[0])
        self.assertAlmostEqual(0.33333333, prob_list[1])
        self.assertAlmostEqual(0.6, prob_list[2])

        self.assertLessEqual(prob_list[0], prob_list[1])
        self.assertLessEqual(prob_list[1], prob_list[2])

        self.assertAlmostEqual(1.0, sum(prob_list))

class TestFitnessTruncationRanking(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

    def test_classinit_(self):

        fit = FitnessTruncationRanking(self.fitness_list, .5)
        self.assertAlmostEqual(.5, fit._trunc_rate)

    def test_set_trunc_rate(self):

        fit = FitnessTruncationRanking(self.fitness_list, .5)
        self.assertRaises(ValueError, fit.set_trunc_rate, 5)
        self.assertRaises(ValueError, fit.set_trunc_rate, 5.0)
        self.assertRaises(ValueError, fit.set_trunc_rate, -5)
        self.assertRaises(ValueError, fit.set_trunc_rate, -5.0)
        self.assertRaises(ValueError, fit.set_trunc_rate, 1.0)

        fit.set_trunc_rate(.4)
        self.assertAlmostEqual(.4, fit._trunc_rate)

    def test_calc_prob(self):
        fit = FitnessTruncationRanking(self.fitness_list, .5)
        self.assertAlmostEqual(.125, fit._calc_prob(10, 2))

    def test_select(self):

        fit = FitnessTruncationRanking(self.fitness_list, .5)
        count = len([i for i in fit.select()])
        self.assertEqual(3, count)


class TestReplacement(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])
        self.repl = Replacement(self.fitness_list)

    def test_classinit_(self):
        self.assertEqual(0, self.repl._replacement_count)


class TestReplacementDeleteWorst(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

        self.repl = ReplacementDeleteWorst(self.fitness_list,
                                            replacement_count=1)

    def test_classinit_(self):

        self.assertEqual(1, self.repl._replacement_count)
        self.assertEqual(MIN, self.repl._selection_type)

    def test_set_replacement_count(self):
        self.repl.set_replacement_count(2)
        self.assertEqual(2, self.repl._replacement_count)

        self.repl.set_replacement_count(3)
        self.assertEqual(3, self.repl._replacement_count)

        self.assertRaises(ValueError, self.repl.set_replacement_count, 4)

    def test_select(self):

        #   attempting to maximize remove worst 3
        self.repl.set_replacement_count(3)
        values = [i for i in self.repl.select()]

        self.assertLessEqual(self.repl._fitness_list[values[0]][0],
                            self.repl._fitness_list[values[1]][0],)
        self.assertLessEqual(self.repl._fitness_list[values[1]][0],
                            self.repl._fitness_list[values[2]][0],)

        #   attempting to minimize remove worst 3
        self.fitness_list = FitnessList(MIN)
        values = [i for i in self.repl.select()]

        self.assertGreaterEqual(self.repl._fitness_list[values[0]][0],
                            self.repl._fitness_list[values[1]][0],)
        self.assertGreaterEqual(self.repl._fitness_list[values[1]][0],
                            self.repl._fitness_list[values[2]][0],)

class TestReplacementTournament(unittest.TestCase):

    def setUp(self):
        self.fitness_list = FitnessList(MAX)
        self.fitness_list.extend([[1.5, 0], [2.5, 1], [12.0, 2]])

        self.repl = ReplacementTournament(self.fitness_list,
                                            tournament_size=1)

    def test_classinit_(self):

        self.assertEqual(1, self.repl._tournament_size)

if __name__ == '__main__':
    unittest.main()
