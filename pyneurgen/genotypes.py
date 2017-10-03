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
This module implements genotypes for grammatical evolution.

"""
from datetime import datetime
import logging
import random
import re
import traceback

from pyneurgen.utilities import base10tobase2, base2tobase10

STOPLIST = ['runtime_resolve', 'set_bnf_variable']
VARIABLE_FORMAT = '(\<([^\>|^\s]+)\>)'
MUT_TYPE_M = 'm'
MUT_TYPE_S = 's'
BNF_PROGRAM = 'program'

#   Positions in _timeouts
TIMEOUT_PROG_BUILD = 0
TIMEOUT_PROG_EXECUTE = 1

DEFAULT_LOG_FILE = 'pyneurgen.log'
DEFAULT_LOG_LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=DEFAULT_LOG_FILE,
                    level=DEFAULT_LOG_LEVEL)


class Genotype(object):
    """
    The Genotype class holds the genetic material.  It has the ability to run
    fitness functions and mutate.  It is an internal object, and so few aspects
    of it would be regarded as public.

    The class takes many properties from the grammatical evolution class, and
    so in some ways it might seem to be unnecessarily duplicative.  The reason
    for doing it this way is to make each genotype relatively complete on its
    own.  That way, if the genotype is packed up and marshalled off to a remote
    device for processing, everything is there to handle the tasks.

    """

    def __init__(self, start_gene_length,
                        max_gene_length,
                        member_no):
        """
        This function initiates the genotype.  It must open with the starting
        gene length and the maximum gene length.  These lengths are the decimal
        lengths not the binary lengths.  In addition, the member number is
        needed, since the genotype creation process is controlled by the
        grammatic evolution class.

        """

        self.member_no = member_no
        self.local_bnf = {}
        self._max_program_length = None
        self._fitness = None
        self._fitness_fail = None
        self._wrap = True
        self._extend_genotype = True
        self.starttime = None
        self._timeouts = (0, 0)

        self._gene_length = start_gene_length
        self._max_gene_length = max_gene_length

        self.binary_gene = None
        self.decimal_gene = None
        self._generate_binary_gene(self._gene_length)
        self.generate_decimal_gene()

        self._position = (0, 0)

        self.errors = []

    def _generate_binary_gene(self, length):
        """
        This function creates a random set of bits.

        """

        geno = []
        count = 0
        while count < length * 8:
            geno.append(str(random.randint(0, 1)))
            count += 1
        self.binary_gene = ''.join(geno)

    def set_binary_gene(self, binary_gene):
        """
        This function sets the value of the binary gene directly.  This is
        used in the crossover and mutation functions.  There is an automatic
        adjustment to trim the length to a multiple of 8.

        """

        length = len(binary_gene)
        trunc_binary_gene = binary_gene[:length - (length % 8)]
        self.binary_gene = trunc_binary_gene
        self._gene_length = len(self.binary_gene) / 8

    def generate_decimal_gene(self):
        """
        This function converts the binary gene to a usable decimal gene.

        """

        if self._gene_length == 0:
            raise ValueError("Invalid gene length")
        dec_geno = []

        for i in range(0, self._gene_length * 8, 8):
            item = self.binary_gene[i:i + 8]
            str_trans = base2tobase10(item)
            dec_geno.append(int(str_trans))

        self.decimal_gene = dec_geno
        self._position = (0, 0)

    @staticmethod
    def _dec2bin_gene(dec_gene):
        """
        This is a utility function that converts a decimal list to binary
        string.

        """

        bin_gene = []
        for item in dec_gene:
            bin_gene.append(base10tobase2(item, zfill=8))
        return ''.join(bin_gene)

    @staticmethod
    def _place_material(program, item, start_pos, end_pos):
        """
        This is a utility function that replaces a part of a string in a
        specific location.

        """

        if end_pos > len(program) - 1:
            raise ValueError("end_pos greater than len(program)")
        if start_pos < 0:
            raise ValueError("starting position cannot be less than 0")
        if start_pos > end_pos:
            raise ValueError("starting position > end postion")
        if start_pos == 0:
            if end_pos == len(program) - 1:
                program = item
            else:
                program = item + program[end_pos + 1:]
        else:
            if end_pos == len(program) - 1:
                program = program[:start_pos] + item
            else:
                program = program[:start_pos] + item + \
                            program[end_pos:]
        return program

    def runtime_resolve(self, item, return_type):
        """
        This function is callable by the generated program to enable
        additional values be pulled from genotype and BNF as the need arises
        during execution of the program.

        Usage is self.runtime_resolve('<myvariable>', return_type);

        The return type casts the result back to the format needed.  Supported
        return types are: 'int', 'float', 'str', and 'bool'.

        """

        value = self._map_variables(item, False)
        value = self._fmt_resolved_vars(value, return_type)
        return value

    @staticmethod
    def _fmt_resolved_vars(value, return_type):
        """
        This method formats the result for a resolved variable for use
        during runtime so that the information can fit into the context of what
        is running.

        Note that if the execute code was to be subclassed to a parser to avoid
        the use of exec, then this funtion should also be done as well, since
        it uses eval.

        """

        return_types = ['int', 'float', 'str', 'bool']

        if return_type == 'str':
            return value
        elif return_type == 'int':
            return conv_int(value)
        elif return_type == 'float':
            try:
                value = float(value)
            except:
                #   allow normal error message to bubble up
                value = eval(value)
        elif return_type == 'bool':
            if value in 'True':
                value = True
            elif value == 'False':
                value = False
            else:
                msg = "return_type must be either True or False: %s"
                raise ValueError(msg, value)
        else:
            msg = "return_type, %s must be in %s" % (value, return_types)
            raise ValueError(msg)

        return value

    def set_bnf_variable(self, variable_name, value):
        """
        This function adds a variable to the bnf.  The format is the name,
        typically bounded by <>, such as "<variable_name>", and the parameters
        are in the form of a list. The items in the list will be converted to
        strings, if not already.

        """

        if isinstance(value, list):
            self.local_bnf[variable_name] = value
        else:
            self.local_bnf[variable_name] = [str(value)]

    def resolve_variable(self, variable):
        """
        This function receives a variable and using the variable as a key
        looks it up in the local_bnf.  The list of possible values available
        are then used by the genotype via a codon to select a final value that
        would be used.

        """

        values = self.local_bnf[variable]
        #try:
        value = self._select_choice(self._get_codon(), values)
        #except:
            #raise ValueError("""
                #Failure to resolve variable: %s values: %s
                #""" % (variable, values))

        return str(value)

    def _map_variables(self, program, check_stoplist):
        """
        This function looks for a variable in the form of <variable>.  If
        check_stoplist is True, then there will be a check to determine if it
        is a run-time variable, and therefor will be resolved later.

        This process runs until all of the variables have been satisfied, or a
        time limit on the process has been reached.

        """

        def on_stoplist(item):
            """
            Checks the stop list for runtime variables

            """

            status = False
            for stopitem in STOPLIST:
                if item.find(stopitem) > -1:
                    status = True

            return status

        self.errors = []
        incomplete = True
        prg_list = re.split(VARIABLE_FORMAT, program)
        while incomplete:
            position = 0
            continue_map = False
            while position < len(prg_list):
                item = prg_list[position]
                if item.strip() == '':
                    del(prg_list[position])
                else:
                    if item[0] == "<" and item[-1] == ">":
                        #   check stop list
                        status = True
                        if check_stoplist and position > 0:
                            if on_stoplist(prg_list[position - 1]):
                                status = False
                        if status:
                            prg_list[position] = self.resolve_variable(item)
                            continue_map = True

                        del(prg_list[position + 1])
                    position += 1

            program = ''.join(prg_list)
            prg_list = re.split(VARIABLE_FORMAT, program)
            elapsed = datetime.now() - self.starttime

            #   Reasons to fail the process
            if check_stoplist:
                #   Program already running
                if elapsed.seconds > self._timeouts[TIMEOUT_PROG_EXECUTE]:
                    msg = "elapsed time greater than program timeout"
                    logging.debug(msg)
                    self.errors.append(msg)
                    raise StandardError(msg)
                    #continue_map = False
            else:
                #   Preprogram
                if elapsed.seconds > self._timeouts[TIMEOUT_PROG_BUILD]:
                    msg = "elapsed time greater than preprogram timeout"
                    logging.debug(msg)
                    self.errors.append(msg)
                    raise StandardError(msg)
                    #continue_map = False

            if len(program) > self._max_program_length:
                #   Runaway process
                msg = "program length, %s is beyond max program length: %s" % (
                            len(program), self._max_program_length)
                logging.debug(msg)
                logging.debug("program follows:")
                #logging.debug(program)
                self.errors.append(msg)
                raise StandardError(msg)
                #continue_map = False

            if continue_map is False:
                return program

    def _get_codon(self):
        """
        This function gets the next decimal codon from the genotype.

        There are two counters for this function. One pointer is used to
        indicate the next location of the decimal code that is to be returned.
        The other pointer is the index of the codon that has been drawn
        regardless if process has wrapped around.

        If the end of the genotype is reached, and the wrap flag is True, then
        the position for the next codon is taken from the front again.
        Additionally, if wrapping has taken place and the extend_genotype flag
        is set, then the genotype will continue to grow in length until the
        max_gene_length is reached.

        If the wrap flag is not set, when the end of the genotype is
        reached, an error is raised.

        At the start of this function, the position has been already
        incremented to get the codon.  Therefore, the position has to be
        checked to determine whether it is pointing past the end of of the
        maximum length of the gene. If it is, then the position is just
        reset back to the starting position.

        """

        #   position is the location on the gene, sequence_no is the number of
        #   codons used since the start
        position, sequence_no = self._position
        length = len(self.decimal_gene)
        wrap = self._wrap

        status = True
        while status:
            if not wrap:
                if sequence_no == self._max_gene_length:
                    raise ValueError("Max length of genotype reached.")
            codon = self.decimal_gene[position]
            if self._extend_genotype:
                if sequence_no == length:
                    #   modify var directly
                    self.decimal_gene.append(codon)
                    self._gene_length = len(self.decimal_gene)

            position += 1
            sequence_no += 1
            if position == length:
                if wrap:
                    position = 0

            self._position = (position, sequence_no)
            return codon

    def _reset_gene_position(self):
        """
        This function resets the next decimal gene that will be selected back
        to 0.  The point of this is when reusing a gene that is already
        instantiated, you can regenerate your program using exactly the same
        characteristics as before.

        """

        self._position = (0, 0)

    def _update_genotype(self):
        """
        This function updates the binary genotype from the decimal gene if the
        genotype is extended.

        """

        self.set_binary_gene(self._dec2bin_gene(self.decimal_gene))

    def compute_fitness(self):
        """
        This function computes the fitness function.  The process consists
        mapping the codon to the program variables and running the resulting
        program and computing the fitness value.  In addition, the binary gene
        is updated if the decimal gene has been extended.

        """

        self._reset_gene_position()
        self._map_gene()
        if self._extend_genotype:
            logging.debug("updating genotype...")
            self._update_genotype()
            logging.debug("Finished updating genotype...")

        return self._fitness

    def _map_gene(self):
        """
        This function applies the genotype information to build a program.
        Mapping the variables into the search space is an initial load, and can
        also iteratively accept values as the program that has been created
        executes via the runtime_resolve function.

        If for any reason the mapping process fails to create a viable
        program, or it takes too long, then the process is aborted and the
        fitness_fail value is applied instead.

        This function uses the print command to show the program that has been
        generated as well as print the fitness value.  It is expected that this
        will be converted to a more useful logging system at some point.

        """

        self.local_bnf['<fitness>'] = [str(self._fitness_fail)]
        try:
            logging.debug("==================================================")
            logging.debug("mapping variables to program...")
            self.local_bnf[BNF_PROGRAM] = [
                    'mapping variables into program failed']
            program = self._map_variables(self.local_bnf['<S>'][0], True)
            logging.debug("finished mapping variables to program...")
            self.local_bnf[BNF_PROGRAM] = [program]
            #print program[program.find('def'):]
            logging.debug(program)
            self._execute_code(program)
            logging.debug("==================================================")
        except:
            #traceback.print_exc()
            #a = raw_input("waiting")
            logging.debug("program failed")
            program = self.local_bnf['program'][0]
            logging.debug("errors: %s", (self.errors))
            logging.debug(program)
            #logging.debug(traceback.print_exc())
            logging.debug(traceback.format_exc())
            logging.debug("end of failure report")
            #a = raw_input("Program failed")
            #if a == "stop":
                #raise ValueError("Program halted")

        self._fitness = float(self.local_bnf['<fitness>'][0])

    def _execute_code(self, program):
        """
        This function executes code that has been generated. This function
        would be subclassed if executing the code on a remote server, or
        swapping in a custom parser.

        """

        self.local_bnf['program'] = program

        #   I'll revisit this again sometime.
        #print "compiling code..."
        #program_comp = compile(program, '<program>', 'exec')
        #print "executing code..."
        #exec program_comp
        ns = locals()
        exec(program) in ns

    def mutate(self, mutation_rate, mutation_type):
        """
        This is function randomly mutates a binary genotype by changing 1 to 0
        and vice versa.  It is not context-perserving at this level.

        """

        if mutation_type == MUT_TYPE_S:
            if random.random() < mutation_rate:
                self._single_mutate()
        elif mutation_type == MUT_TYPE_M:
            self._multiple_mutate(mutation_rate)
        else:
            raise ValueError("The mutation type must be either '%s' or '%s'",
                MUT_TYPE_S, MUT_TYPE_M)

    def _multiple_mutate(self, mutation_rate):
        """
        This function walks the gene and based upon the mutation rate will
        alter a bit.

        """

        if mutation_rate < 0.0:
            raise ValueError("The mutation rate must be >= 0.0")
        elif mutation_rate > 1.0:
            raise ValueError("The mutation rate must be <= 1.0")
        else:
            pass

        gene = self.binary_gene
        length = len(gene)
        for i in range(length):
            if random.random() < mutation_rate:
                gene = self._mutate(gene, i)

        self.set_binary_gene(''.join(gene))
        self.generate_decimal_gene()

    def get_binary_gene_length(self):
        """
        This function returns the length of the binary gene.  Which is
        8 times the length of the decimal gene.

        """

        return self._gene_length * 8

    def _single_mutate(self):
        """
        This function with a randomly selects a mutation point within the gene
        and changes a 1 to 0, or vice versa.

        """

        position = random.randint(0, self._gene_length * 8 - 1)
        gene = self.binary_gene

        self.binary_gene = self._mutate(gene, position)
        self.generate_decimal_gene()

    @staticmethod
    def _mutate(gene, position):
        """
        This function does the actual mutation of the gene at a specific
        position.

        """

        if gene[position] == '0':
            gene = ''.join([gene[:position], "1", gene[position + 1:]])
        else:
            gene = ''.join([gene[:position], "0", gene[position + 1:]])

        return gene

    @staticmethod
    def _select_choice(codon, selection):
        """
        This function, based upon the codon, makes a choice from the list.
        The determination is based upon the module of the codon to the length
        of the list of choices.  For example, if the codon is 10 and the list
        is 7 choices long, then the selection would be from selection[3].  This
        ensures that for every codon for every selection there is some choice.

        """

        if isinstance(selection, list):
            return selection[codon % len(selection)]
        else:
            msg = "selection. %s, must be a list" % (selection)
            raise ValueError(msg)

    def get_program(self):
        """
        This function returns the program that has been generated.  It is only
        valid after the gene has been mapped.

        """

        return self.local_bnf['program']

    def get_preprogram(self):
        """
        This function returns the prototype program to which the variables
        will be applied.

        """

        return self.local_bnf['<S>']

    def get_fitness(self):
        """
        This function returns the fitness value that has been created as a
        result of running the fitness function.

        """

        return self._fitness

    def get_fitness_fail(self):
        """
        This function returns the fitness value that constitutes failure as
        assigned by the parent grammatical evolution.

        """

        return self._fitness_fail


def conv_int(str_value):
    """
    This method attempts to convert string value to an int.  This function
    used to live inside self._fmt_resolved_vars, but was taken out because
    it is easier to do unit testing this way.

    """

    try:
        value = int(str_value)
    except:
        #   Try the expensive eval -- if fails, then let error bubble up.
        value = eval(str_value)

    return value
