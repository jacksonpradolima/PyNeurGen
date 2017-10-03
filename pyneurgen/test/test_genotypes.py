import unittest

from datetime import datetime, timedelta

from pyneurgen.genotypes import Genotype, MUT_TYPE_M, MUT_TYPE_S, conv_int


class TestGenotype(unittest.TestCase):
    """
    The Genotype class holds the genetic material.  It has the ability to run
    fitness functions and mutate.  It is an internal object, and so few aspects
    of it would be regarded as public.

    """

    def setUp(self):

        start_gene_length = 10
        max_gene_length = 20
        member_no = 1

        self.g = Genotype(start_gene_length, max_gene_length, member_no)

    def test_class_init__(self):
        """Testing class init"""

        self.assertEqual(1, self.g.member_no)
        self.assertEqual(10, self.g._gene_length)
        self.assertEqual(20, self.g._max_gene_length)

        self.assertEqual({}, self.g.local_bnf)
        self.assertEqual(None, self.g._max_program_length)
        self.assertEqual(None, self.g._fitness)
        self.assertEqual(None, self.g._fitness_fail)
        self.assertEqual(True, self.g._wrap)
        self.assertEqual(True, self.g._extend_genotype)
        self.assertEqual(None, self.g.starttime)
        self.assertEqual((0, 0), self.g._timeouts)
        self.assertEqual((0, 0), self.g._position)
        self.assertEqual(None, self.g._max_program_length)
        self.assertEqual(None, self.g._max_program_length)
        self.assertEqual([], self.g.errors)

        #   Not tested here
        #   self.binary_gene = None
        #   self.decimal_gene = None
        #   self._generate_binary_gene(self._gene_length)
        #   self.generate_decimal_gene()

    def test__generate_binary_gene(self):
        """
        This function tests the generation process for a binary gene.

        Basically, all that is being tested on this is proper length, and
        whether it consists of ones or zeros.  No test is made of rand_int.

        """

        self.g._generate_binary_gene(20)
        self.assertEqual(20 * 8, len(self.g.binary_gene))

        gene = self.g.binary_gene
        self.assertEqual(20 * 8, gene.count("0") + gene.count("1"))

    def test_set_binary_gene(self):
        """
        This function tests setting a binary gene.

        """

        #   should truncate automatically if too long
        binary_gene = "0110100101"
        gene_result = "01101001"
        self.g.set_binary_gene(binary_gene)
        self.assertEqual(gene_result, self.g.binary_gene)

        #   should set the gene length correctly
        self.assertEqual(1, self.g._gene_length)

    def test_generate_decimal_gene(self):
        """
        This function tests the generation of the decimal gene from a binary
            gene

        Tested:
            There must be decimal gene greater than length 0.
            The length of generated decimal gene will be 1/8 the size of the
            binary gene.
            A specific example of binary gene/decimal gene is tested.
            The position pointer for the next gene is reset back to (0, 0))

        """

        #    There must be decimal gene greater than length 0.
        self.g.binary_gene = []
        self.assertRaises(ValueError, self.g.generate_decimal_gene)

        #    The length of generated decimal gene will be 1/8 the size of the
        dec_gene = [2, 1, 3, 4]

        #    The length of generated decimal gene will be 1/8 the size of the
        #       binary gene.
        #   Uses set binary gene to force the recalculation fo the gene length
        self.g.set_binary_gene('00000010000000010000001100000100')
        length = len(self.g.binary_gene)
        self.g.generate_decimal_gene()
        self.assertEqual(length / 8, len(self.g.decimal_gene))


        #    A specific example of binary gene/decimal gene is tested.
        self.assertEqual(dec_gene, self.g.decimal_gene)

        #    The position pointer for the next gene is reset back to (0, 0))
        self.assertEqual((0, 0), self.g._position)

    def test__dec2bin_gene(self):
        """
        This function tests the process of computing a binary gene from the
        decimal gene.

        """

        dec_gene = [2, 1, 3, 4]
        binary_gene = '00000010000000010000001100000100'

        self.assertEqual(binary_gene, self.g._dec2bin_gene(dec_gene))

    def test__place_material(self):
        """
        This function tests whether the process of a string replacement takes
        place properly.

        Is the start position consistent with main string?
        Is the end position consistent with the main string?
        Does the replacement actually work?

        """

        main_string = "this is a test"
        item = " very big"

        #   Valid start position- too big?
        start_pos = 100
        end_pos = 10
        self.assertRaises(ValueError, self.g._place_material,
            main_string, item, start_pos, end_pos)

        #   Valid start position- too small?
        start_pos = -1
        self.assertRaises(ValueError, self.g._place_material,
            main_string, item, start_pos, end_pos)

        #   Valid end position - too big
        start_pos = 10
        end_pos = 100
        self.assertRaises(ValueError, self.g._place_material,
            main_string, item, start_pos, end_pos)

        #   Valid end position
        start_pos = 10
        end_pos = 5
        self.assertRaises(ValueError, self.g._place_material,
            main_string, item, start_pos, end_pos)

        #   Valid insertion
        start_pos = 9
        end_pos = 9
        self.assertEqual("this is a very big test",
            self.g._place_material(main_string, item,
                start_pos, end_pos))

        #   Valid replacement
        start_pos = 10
        end_pos = 13
        item = "drill"
        self.assertEqual("this is a drill",
            self.g._place_material(main_string, item,
                start_pos, end_pos))


        #   testing the gene bit use
        binary_gene = "11010111"
        bit = "3" # to make it really stand out
        start_pos = 5
        end_pos = 6
        self.assertEqual("11010311", self.g._place_material(
                binary_gene, bit, start_pos, end_pos))

    def test_runtime_resolve(self):
        """
        This function tests the process that resolves a <variable> to value
        that can be used in a program as it executes.

        Note that starttime must be initialized.  This is to control for a
        runaway process during runtime.

        """

        self.g.set_bnf_variable("<test>", ["this", "is", "test"])
        self.g.set_bnf_variable("<test_value>", [1, "is", "test"])
        self.g.decimal_gene = [3, 2, 5, 6]
        self.g._max_gene_length = 4
        self.g._position = (0, 0)
        self.g.starttime = datetime.now()
        self.g._max_program_length = 10000

        self.assertEqual("this", self.g.runtime_resolve("<test>", 'str'))
        self.assertEqual("test", self.g.runtime_resolve("<test>", 'str'))

    def test__fmt_resolved_vars(self):
        """
        This function tests the process of converting various kinds of
        variables into specific types for use in program lines.

        """

        # int tests
        self.assertEqual(10, self.g._fmt_resolved_vars(10.0, 'int'))

        #self.g._fmt_resolved_vars("ten", 'int')
        self.assertRaises(NameError, self.g._fmt_resolved_vars, "ten", 'int')
        self.assertEqual(10, self.g._fmt_resolved_vars("10", 'int'))
        self.assertEqual(8, self.g._fmt_resolved_vars("3 + 5", 'int'))

        # float tests
        self.assertEqual(10.0, self.g._fmt_resolved_vars(10, 'float'))
        self.assertRaises(NameError, self.g._fmt_resolved_vars, "ten", 'float')
        self.assertEqual(10.0, self.g._fmt_resolved_vars("10", 'float'))
        self.assertEqual(8.0, self.g._fmt_resolved_vars("3 + 5", 'float'))

        # bool tests
        self.assertEqual(True, self.g._fmt_resolved_vars('True', 'bool'))
        self.assertEqual(False, self.g._fmt_resolved_vars("False", 'bool'))
        self.assertRaises(ValueError, self.g._fmt_resolved_vars, 'not true',
                                                            'bool')

    def test_set_bnf_variable(self):
        """
        Test setting bnf variables
        """

        #   new variable name
        self.g.set_bnf_variable("test_variable", "test_value")
        self.assertEqual(True, self.g.local_bnf.has_key("test_variable"))
        self.assertEqual(["test_value"], self.g.local_bnf["test_variable"])

        #   new value
        self.g.set_bnf_variable("test_variable", "new_value")
        self.assertEqual(["new_value"], self.g.local_bnf["test_variable"])

        #   Does it modify a list?
        self.g.set_bnf_variable("test_variable", ["is", "a", "list"])
        self.assertEqual(["is", "a", "list"],
                                self.g.local_bnf["test_variable"])

        #   Does it convert a value to a string?
        self.g.set_bnf_variable("test_variable", 523.45)
        self.assertEqual(["523.45"], self.g.local_bnf["test_variable"])


    def test_resolve_variable(self):
        """
        This function tests the process of converting a variable received to a
        to a list of possible variables as found in the bnf.

        """

        self.g.set_bnf_variable("<test>", ["this", "is", "test"])
        self.g.decimal_gene = [3, 2, 5, 6]
        self.g._max_gene_length = 4
        self.g._position = (0, 0)

        self.assertEqual("this", self.g.resolve_variable("<test>"))
        self.assertEqual("test", self.g.resolve_variable("<test>"))

    def test__map_variables(self):
        """
        This function test the process of mapping variables\

        Test:
            mapping of variables with check_stoplist
            mapping of variables without check_stoplist

        """

        self.g._position = (0, 0)
        self.g.starttime = datetime.now()
        self.g._max_program_length = 10000
        self.g.set_bnf_variable("<value1>", [-1, 2, 0])
        self.g.set_bnf_variable("<value2>", [1, 2, 3])

        program = ''.join([
            'a = <value1>\n',
            'b = <value2>\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)'])

        completed_program = ''.join([
            'a = -1\n',
            'b = 2\n',
            'fitness = a + b\n',
            'self.set_bnf_variable("<fitness>", fitness)'])

        self.g.decimal_gene = [0, 1, 5, 6]
        self.g._max_gene_length = 4

        self.assertEqual(completed_program, self.g._map_variables(
                                                        program, True))

    def test_conv_int(self):
        """
        This function tests the process of converting a string value to an int.

        """

        self.assertEqual(23, conv_int("23"))
        self.assertEqual(23, conv_int(" 23 "))
        self.assertRaises(NameError, conv_int, "test")
        self.assertEqual(5, conv_int("3 + 2"))

    def test__get_codon(self):
        """
        This function tests getting a codon.

        Test:
            Testable conditions:
                Wrap  True/False
                Extend Genotype  True/False
                Beyond max gene length

            Both position and sequence_no are less than the length of the
                decimal gene.
            Gene is set to not wrap, and sequence_no is set beyond the length
                of the gene.
            Gene is set to wrap and sequence_no is greater than gene length.


        """

        self.g.decimal_gene = [3, 34, 5, 6]
        self.g._max_gene_length = 4
        self.g._wrap = False
        self.g._extend_genotype = False
        self.g._position = (0, 0)

        #   no wrap -- gets to end of sequence, raises exception
        self.assertEqual(3, self.g._get_codon())
        self.assertEqual(34, self.g._get_codon())
        self.assertEqual(5, self.g._get_codon())
        self.assertEqual(6, self.g._get_codon())
        self.assertEqual((4, 4), self.g._position)
        self.assertRaises(ValueError, self.g._get_codon)

        #   wrap -- no extend -- _max_gene_length = length of gene
        #       gets to end of sequence, wraps around to start,
        #           length not increased, raises error
        self.g._wrap = True
        self.g._max_gene_length = 5

        self.g._position = (0, 0)
        self.assertEqual(3, self.g._get_codon())
        self.assertEqual(34, self.g._get_codon())
        self.assertEqual(5, self.g._get_codon())
        self.assertEqual(6, self.g._get_codon())
        self.assertEqual((0, 4), self.g._position)
        self.assertEqual(3, self.g._get_codon())
        self.assertEqual((1, 5), self.g._position)
        self.assertEqual(4, len(self.g.decimal_gene))

        #   wrap -- extend -- _max_gene_length > length of gene
        #       gets to end of sequence, wraps around to start,
        self.g._wrap = True
        self.g._extend_genotype = True
        self.g._position = (0, 0)
        self.assertEqual(3, self.g._get_codon())
        self.assertEqual(34, self.g._get_codon())
        self.assertEqual(5, self.g._get_codon())
        self.assertEqual(6, self.g._get_codon())
        self.assertEqual((0, 4), self.g._position)
        self.assertEqual(3, self.g._get_codon())
        self.assertEqual(5, len(self.g.decimal_gene))
        self.assertEqual((1, 5), self.g._position)

    def test__reset_gene_position(self):
        """
        This function tests whether the starting position is reset back to 0.

        """

        self.g._position = "something else"
        self.g._reset_gene_position()
        self.assertEqual((0, 0), self.g._position)

    def test__update_genotype(self):
        """
        This function tests whether the binary gene is properly updated with a
        new decimal gene.

        """

        self.g.binary_gene = '0000001000000001'

        #   new dec_gene
        self.g.decimal_gene = [2, 1, 3, 4]

        self.g._update_genotype()
        self.assertEqual('00000010000000010000001100000100',
                        self.g.binary_gene)

    def test_compute_fitness(self):
        """
        This function tests the process of computing fitness.

        Because this process is an amalgamation of other processes, which are
        already tested, what will be tested here, is setting up a sample
        template program with variables, mapping the gene

        Test:
            Program that matches fitness
            Program that fails due to program error
            Program that fails due to time out.

        """

        self.g._fitness_fail = "-999999"
        self.g.set_bnf_variable('<S>', ''.join([
                                            'a = <value1>\n',
                                            'b = <value2>\n',
                                            'fitness = a + b\n',
                    'self.set_bnf_variable("<fitness>", fitness)']))


        self.g.set_bnf_variable('<fitness>', 0)

        self.g.set_bnf_variable("<value1>", [-1, 2, 0])
        self.g.set_bnf_variable("<value2>", [1, 2, 3])
        self.g.decimal_gene = [0, 1, 5, 6]
        self.g._max_gene_length = 4

        #   intentionally incorrect position set to test reset
        self.g._position = (3, 3)

        self.g.starttime = datetime.now()
        self.g._max_program_length = 10000

        self.assertEqual(1, self.g.compute_fitness())
        self.assertEqual(1, self.g._fitness)

        #   Faulty program -- incorrect variable
        self.g._fitness = "test"
        self.g.set_bnf_variable('<S>', ''.join([
                    'logging.debug("Executing Example of a Faulty Program")\n',
                    'a = <value1>\n',
                    'b1 = <value2>\n',
                    'fitness = a + b\n',
                    'self.set_bnf_variable("<fitness>", fitness)']))

        self.assertEqual(-999999, self.g.compute_fitness())

        #   Long running program
        self.g._fitness = -999999
        self.g.starttime = datetime.now() - timedelta(seconds=1000)
        self.g._timeouts = (1, 1)

        self.g.set_bnf_variable('<S>', ''.join([
                    'logging.debug("Executing Example of a Long Program")\n',
                    'a = <value1>\n',
                    'b = <value2>\n',
                    'fitness = a + b\n',
                    'self.set_bnf_variable("<fitness>", fitness)']))

        self.assertEqual(-999999, self.g.compute_fitness())

        ## Long Program creation time
        #   not yet implemented

        #   Program size
        #   not yet implemented

    def test__map_gene(self):
        """
        This function tests the production and execution of a program by
        mapping the gene to the template program.

        This is not implemented.  test_compute_fitness would not function if
        map_gene() did not work.  Nonetheless, at some point, a separate
        test for this should be written.

        """

        pass

    def test__execute_code(self):
        """
        This function tests whether code can be executed.
        In addition, prior to executing, the program is put into the local bnf.

        """

        test_program = "a = 3\nb=2\nc = a + b"

        self.g._execute_code(test_program)

        self.g.local_bnf['program'] = test_program
        self.assertEqual("a = 3\nb=2\nc = a + b", self.g.local_bnf['program'])

    def test__mutate(self):
        """
        This function tests whether _mutate changes a bit in the appropriate
            spot.
        Note that it does not check for appropriateness of position value,
        because it has already been cleared by the calling routines.

        In these test functions, test__mutate and test_mutate are ambiguous.
        test__mutate tests self._mutate for altering a gene at a particular
            spot.
        test_mutate tests self.mutate which is an umbrella function for both
        single and multiple mutations.  It's possible that self._mutate should
        be remained.

        """

        gene = '1110101'

        position = 0
        self.assertEqual('0110101', self.g._mutate(gene, position))

        position = 4
        self.assertEqual('1110001', self.g._mutate(gene, position))

        position = 6
        self.assertEqual('1110100', self.g._mutate(gene, position))

    def test_mutate(self):
        """
        This function tests the routing to mutation type functions.
        Tested:
            Is an error generated if the mutation type is not 's' or 'm'.
            Is an error generated if the mutation rate outside of (0, 1).
        """

        mutation_rate = .05
        mutation_type = 'wrong'
        self.assertRaises(ValueError, self.g.mutate, mutation_rate,
            mutation_type)

        #   Invalid mutation rates
        mutation_rate = -0.5
        mutation_type = MUT_TYPE_M
        self.assertRaises(ValueError, self.g.mutate, mutation_rate,
                                                                mutation_type)

        mutation_rate = -1.5
        self.assertRaises(ValueError, self.g.mutate, mutation_rate,
                                                                mutation_type)

        #   Edge values - Failure if ValueError raised in function
        mutation_rate = 0.0
        mutation_type = MUT_TYPE_M
        self.g.mutate(mutation_rate, mutation_type)

        mutation_rate = 1.0
        mutation_type = MUT_TYPE_M
        self.g.mutate(mutation_rate, mutation_type)

    def test__multiple_mutate(self):
        """
        This function tests multiple mutations.  Because the rate value is
        already tested in the upstream test, this will simply test to see if
        any mutations are made.  The looping process is currently not being
        tested until a good way of isolating without unnecessary complication
        can be found.

        """

        mutation_rate = 1.0

        gene = self.g.binary_gene

        self.g._multiple_mutate(mutation_rate)

        self.assertEqual(len(gene), len(self.g.binary_gene))
        self.assertNotEqual(gene, self.g.binary_gene)

    def test_get_binary_gene_length(self):
        """
        This function tests getting the length of binary gene.

        """

        length = self.g._gene_length
        self.assertEqual(length * 8, self.g.get_binary_gene_length())

    def test_single_mutate(self):
        """
        Tests a single mutation.

        """

        length = len(self.g.binary_gene)

        original_gene = self.g.binary_gene
        self.g._single_mutate()

        changes = 0
        for i in range(length):
            if original_gene[i] != self.g.binary_gene[i]:
                changes += 1
        self.assertEqual(1, changes)

    def test__select_choice(self):
        """
        This function tests the process of selecting an item from a list based
        on the codon.

        """

        self.assertRaises(ValueError, self.g._select_choice,
                1, "no list")
        self.assertRaises(ValueError, self.g._select_choice,
                1, 3)

        codon = 0
        selection = ["this", "is", "test"]
        self.assertEqual("this", self.g._select_choice(
                                                codon, selection))
        codon = 40
        selection = ["this", "is", "test"]
        self.assertEqual("is", self.g._select_choice(
                                                codon, selection))
        codon = 65
        selection = ["this", "is", "test"]
        self.assertEqual("test", self.g._select_choice(
                                                codon, selection))
    def test_get_program(self):
        """Test Get program"""
        self.g.local_bnf["program"] = "program here"
        self.assertEqual("program here", self.g.get_program())

    def test_get_preprogram(self):
        """Test Get preprogram"""
        self.g.local_bnf["<S>"] = "preprogram"
        self.assertEqual("preprogram", self.g.get_preprogram())

    def test_get_fitness(self):
        """Test Get Fitness Fail"""
        self.g._fitness = "Fitness"
        self.assertEqual("Fitness", self.g.get_fitness())

    def test_get_fitness_fail(self):
        """Test Get Fitness Fail"""
        self.g._fitness_fail = "Fitness Fail"
        self.assertEqual("Fitness Fail", self.g.get_fitness_fail())


if __name__ == '__main__':
    unittest.main()
