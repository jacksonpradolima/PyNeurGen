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
This module implements the components for an artficial neural network.

"""
import math
from random import random
import ConfigParser

from pyneurgen.layers import Layer
from pyneurgen.nodes import Node, CopyNode, BiasNode, Connection
from pyneurgen.nodes import NODE_OUTPUT, NODE_HIDDEN, NODE_INPUT, NODE_COPY
from pyneurgen.nodes import NODE_BIAS


class NeuralNet(object):
    """
    This class implements a standard multi-layered perceptron (MLP).

    """

    def __init__(self):
        """
        Because there are a number of parameters to specify, there are
        no specific variables that are initialized within __init__.
        """

        self._learnrate = .1
        self._random_constraint = 1.0
        self._epochs = 1
        self.layers = []
        self._data_range = {'learning': [None, None],
                            'validation': [None, None],
                            'test': [None, None]}

        self._allinputs = []
        self._alltargets = []
        self._time_delay = 0

        #   This is where the test results are stored.
        self._allresults = []
        #   This holds the mean squared error for the test.
        self.mse = []

        #   This holds the accumulated mean squared errors for each epoch
        #       tested
        self.accum_mse = []
        self.validation_targets_activations = []
        self.test_targets_activations = []

        self._halt_on_extremes = False

        self.input_layer = None
        self.output_layer = None

    def init_layers(self, input_nodes, total_hidden_nodes_list,
            output_nodes, *recurrent_mods):
        """
        This function initializes the layers.
        The variables:

        * input_nodes: the number of nodes in the input layer
        * total_hidden_nodes_list:  a list of numbers of nodes in the
            hidden layer.  For example, [5, 3]
        * output_nodes: the number of nodes in the output layer

        The initial network is created, and then a series of modifications can
        be made to enable recurrent features.  recurrent_mods are
        configurations for modifications to the neural network that is created
        within init_layers.

        For example, if
            init_layers(input_nodes, total_hidden_nodes_list, output_nodes,
                            ElmanSimpleRecurrent())
        was used, then the initial network structure of input, hidden, and
        output nodes would be created.  After that, the additional copy or
        context nodes that would automatically transfer values from the lowest
        hidden layer would be added to the input layer.

        More than one recurrent scheme can be applied, each one adding to the
        existing network.

        """

        self.layers = []

        #   Input layer
        layer = Layer(len(self.layers), NODE_INPUT)
        layer.add_nodes(input_nodes, NODE_INPUT)

        layer.add_node(BiasNode())

        self.layers.append(layer)
        self.input_layer = layer

        for hid in total_hidden_nodes_list:
            layer = Layer(len(self.layers), NODE_HIDDEN)
            layer.add_nodes(hid, NODE_HIDDEN)

            layer.add_node(BiasNode())

            self.layers.append(layer)

        layer = Layer(len(self.layers), NODE_OUTPUT)
        layer.add_nodes(output_nodes, NODE_OUTPUT)

        self.layers.append(layer)
        self.output_layer = layer

        self._init_connections()

        for recurrent_mod in recurrent_mods:
            recurrent_mod.apply_config(self)

    def set_halt_on_extremes(self, halt):
        """
        This function sets the flag as to whether the program should halt when
        experiencing extremely positive or negative numbers.  This can happen
        when using linear functions and data that may not be normalized.  Such
        things as nan and inf can be experienced otherwise.  Not halting
        instead, simply scales back the values to LARGEVALUE_LIMIT and
        NEGVALUE_LIMIT. A satisfactory output from the network may be in doubt,
        but at least it gives it the possibility.

        """

        err_msg = "Halt: %s -- Must be True or False" % (halt)
        if not isinstance(halt, bool):
            raise ValueError(err_msg)
        else:
            self._halt_on_extremes = halt

    def get_halt_on_extremes(self):
        """
        This function returns the True/False flag for halting on extremes.

        """

        return self._halt_on_extremes

    def set_random_constraint(self, constraint):
        """
        This fuction sets a value between 0 and 1 for limiting the random
        weights upon initialization.  For example, .8 would limit weights to
        -.8 through .8.

        """

        err_msg = """The constraint, %s, must be a float between 0.0 and 1.0
                  """ % (constraint)
        if not isinstance(constraint, float):
            raise ValueError(err_msg)
        elif 0.0 < constraint <= 1.0:
            self._random_constraint = constraint
        else:
            raise ValueError(err_msg)

    def get_random_constraint(self):
        """
        This function gets the random constraint used in weights
        initialization.s

        """

        return self._random_constraint

    def set_epochs(self, epochs):
        """
        This function sets the number of epochs or cycles through the learning
        data.

        """
        err_msg = """The epochs, %s, must be an int """ % (epochs)
        if not isinstance(epochs, int):
            raise ValueError(err_msg)
        elif epochs <= 0:
            raise ValueError(err_msg)
        else:
            self._epochs = epochs

    def get_epochs(self):
        """
        This function gets the number of epochs that will run during learning.

        """

        return self._epochs

    def set_time_delay(self, time_delay):
        """
        This function sets a value for time delayed data.  For example, is the
        time delay was 5, then input values would be taken 5 at a time.  Upon
        the next increment the next input values would be 5, with 4 of the
        previous values included, and one new value.

        """

        err_msg = """The time_delay , %s, must be an int greater than or equal
                  to zero.""" % (time_delay)
        if not isinstance(time_delay, int):
            raise ValueError(err_msg)
        elif time_delay < 0:
            raise ValueError(err_msg)
        else:
            self._time_delay = time_delay

    def get_time_delay(self):
        """
        This function gets the time delay to be used with timeseries data.

        """

        return self._time_delay

    def set_all_inputs(self, allinputs):
        """
        This function sets the inputs.  Inputs are basically treated as a
        list.

        """

        self._allinputs = allinputs

    def set_all_targets(self, alltargets):
        """
        This function sets the targets.

        """

        self._alltargets = alltargets

    def set_learnrate(self, learnrate):
        """
        This function sets the learn rate for the modeling.  It is used to
        determine how much weight to associate with an error when learning.

        """

        err_msg = """The learnrate, %s, must be a float between 0.0 and 1.0
                  """ % (learnrate)

        if not isinstance(learnrate, float):
            raise ValueError(err_msg)
        elif 0.0 < learnrate <= 1.0:
            self._learnrate = learnrate
        else:
            raise ValueError(err_msg)

    def get_learnrate(self):
        """
        This function gets the learn rate for the modeling.  It is used to
        determine how much weight to associate with an error when learning.

        """

        return self._learnrate

    def _set_data_range(self, data_type, start_position, end_position):
        """
        This function sets the data positions by type

        """

        err_msg = "The %s, %s, must be an int value"
        if not isinstance(start_position, int):
            raise ValueError(err_msg % ('start_position', start_position))

        if not isinstance(end_position, int):
            raise ValueError(err_msg % ('end_position', end_position))

        self._data_range[data_type] = (start_position, end_position)

    def set_learn_range(self, start_position, end_position):
        """
        This function sets the range within the data that is to used for
        learning.

        """

        self._set_data_range('learning', start_position, end_position)

    def get_learn_range(self):
        """
        This function gets the range within the data that is to used for
        learning.

        """
        return self._data_range['learning']

    def _check_time_delay(self, position):
        """
        This function checks the position or index of the data and determines
        whether the position is consistent with the time delay that has been
        set.

        """

        if position - self._time_delay < 0:
            raise ValueError("Invalid start position with time delayed data")

    def get_learn_data(self, random_testing=None):
        """
        This function is a generator for learning data.  It is assumed that in
        many cases, this function will be over-written with a situation
        specific function.

        """

        start_position, end_position = self._data_range['learning']
        self._check_positions(start_position, end_position)

        if random_testing is None:
            random_testing = False

        if random_testing:
            for i in self._get_randomized_position(start_position,
                                                    end_position):
                tdly = self.get_time_delay()
                if tdly > 0:
                    start = i - tdly
                    inputs = self._allinputs[start:i + 1]
                else:
                    inputs = self._allinputs[i]

                targets = self._alltargets[i]

                yield (inputs, targets)
        else:
            for inputs, targets in self._get_data(start_position,
                                                    end_position):
                yield (inputs, targets)

    def get_validation_data(self):
        """
        This function is a generator for validation data.  It is assumed that
        in many cases, this function will be over-written with a situation
        specific function.

        """

        start_position, end_position = self._data_range['validation']
        self._check_positions(start_position, end_position)

        for inputs, targets in self._get_data(start_position,
                                                end_position):
            yield (inputs, targets)

    def get_test_data(self):
        """
        This function is a generator for testing data.  It is assumed that in
        many cases, this function will be over-written with a situation
        specific function.

        """

        start_position, end_position = self._data_range['test']
        self._check_positions(start_position, end_position)

        for inputs, targets in self._get_data(start_position,
                                                end_position):
            yield (inputs, targets)

    def _get_data(self, start_position, end_position):
        """
        This function gets an input from the list of all inputs.

        """

        i = start_position
        if end_position > len(self._allinputs) - 1:
            raise ValueError(
                "end_position %s is past end of ._allinputs, %s" % (
                    end_position, len(self._allinputs)))
        while i < end_position:
            tdly = self.get_time_delay()
            if tdly > 0:
                start = i - tdly
                inputs = [item[0] for item in self._allinputs[start:i + 1]]
            else:
                inputs = self._allinputs[i]

            targets = self._alltargets[i]

            yield (inputs, targets)
            i += 1

    @staticmethod
    def _get_randomized_position(start_position, end_position):
        """
        This function accepts integers representing a starting and ending
        position within a set of data and yields a position number in a random
        fashion until all of the positions have been exhausted.

        """

        for i in xrange(start_position, end_position):
            order = [[random(), i]
                for i in xrange(start_position, end_position)]

        order.sort()
        for item in order:
            yield item[1]

    def _check_positions(self, start_position, end_position):
        """
        This function evaluates validates, somewhat, start and end positions
        for data ranges.

        """

        if start_position is None:
            raise ValueError(
                "Start position is not defined.")
        if end_position is None:
            raise ValueError(
                "End position data is not defined.")
        if start_position > end_position:
            raise ValueError("""
                Start position, %s, cannot be greater than
                     end position %s""" % (start_position, end_position))

        self._check_time_delay(start_position)

    def set_validation_range(self, start_position, end_position):
        """
        This function sets the start position and ending position for the
        validation range.  The first test period is often used to test the
        current weights against data that is not within the learning period
        after each epoch run.

        """

        self._set_data_range('validation', start_position, end_position)

    def get_validation_range(self):
        """
        This function gets the start position and ending position for the
        validation range.  The first test period is often used to test the
        current weights against data that is not within the learning period
        after each epoch run.

        """

        return self._data_range['validation']

    def set_test_range(self, start_position, end_position):
        """
        This function sets the start position and ending position for
        the out-of-sample range.

        """

        self._set_data_range('test', start_position, end_position)

    def get_test_range(self):
        """
        This function gets the start position and ending position for
        the out-of-sample range.

        """

        return self._data_range['test']

    def _init_connections(self):
        """
        Init connections sets up the linkages between layers.

        This function connects all nodes, which is typically desirable
        However, note that by substituting in a different process, a
        sparse network can be achieved.  And, there is no restriction
        to connecting layers in a non-traditional fashion such as skip-layer
        connections.

        """

        for layer in self.layers[1:]:
            self._connect_layer(layer)

    def _connect_layer(self, layer):
        """
        Generates connections to the lower layer.

        If it is the input layer, then it's skipped
        It could raise an error, but it seems pointless.

        """

        lower_layer_no = layer.layer_no - 1
        if lower_layer_no >= 0:
            lower_layer = self.layers[lower_layer_no]
            layer.connect_layer(lower_layer)

    def randomize_network(self):
        """
        This function randomizes the weights in all of the connections.

        """

        for layer in self.layers:
            if layer.layer_type != NODE_INPUT:
                layer.randomize(self._random_constraint)

    def learn(self, epochs=None, show_epoch_results=True,
           random_testing=False, show_sample_interval=0):
        """
        This function performs the process of feeding into the network inputs
        and targets, and computing the feedforward process.  After the
        feedforward process runs, the actual values calculated by the output
        are compared to the target values.  These errors are then used by the
        back propagation process to adjust the weights for the next set of
        inputs. If a recurrent netork structure is used, the stack of copy
        levels is pushed with the latest set of hidden nodes.

        Then, the next set of inputs is input.

        When all of the inputs have been processed, resulting in the
        completion of an epoch, if show_epoch_results=True, then the MSE will
        be printed.

        Finally, if random_testing=True, then the inputs will not be processed
        sequentially.  Rather, the inputs will be sorted into a random order
        and then input.  This is very useful for timeseries data to avoid
        autocorrelations.

        """

        if epochs is not None:
            self.set_epochs(epochs)

        self.accum_mse = []
        for epoch in range(self._epochs):
            summed_errors = 0.0
            count = 0
            for inputs, targets in self.get_learn_data(random_testing):
                self.process_sample(inputs, targets, learn=True)
                summed_errors += self.calc_sample_error()
                count += 1
                if show_sample_interval > 0:
                    if count % show_sample_interval == 0:
                        #   Convert to logging at some point
                        print ("epoch %s of %s, sample: %s errors: %s" % (
                            epoch, self._epochs, count, summed_errors))

            mse = self.calc_mse(summed_errors, count)
            if show_epoch_results:
                #   Convert this over to logging
                print ("epoch: %s MSE: %s" % (epoch, mse))

            self.accum_mse.append(mse)

    def validate(self, show_sample_interval=0):
        """
        This function loads and feedforwards the network with validation data.
        Optionally, it can also store the actuals as well.

        """

        return  self._evaluate('v', show_sample_interval)

    def test(self, show_sample_interval=0):
        """
        This function loads and feedforwards the network with test data.
        Optionally, it can also store the actuals as well.

        """

        return  self._evaluate('t', show_sample_interval)

    def _evaluate(self, eval_type, show_sample_interval):
        """
        This function loads and feedforwards the network with data.
        eval_type is either validation or test data ('t' or 'v')
        Optionally, it can also store the actuals as well.

        """

        if eval_type == 't':
            #   test data
            eval_list = self.test_targets_activations = []
            get_data = self.get_test_data
        elif eval_type == 'v':
            eval_list = self.validation_targets_activations = []
            get_data = self.get_validation_data
        summed_errors = 0.0
        count = 0
        for inputs, targets in get_data():
            self.process_sample(inputs, targets, learn=False)
            summed_errors += self.calc_sample_error()
            count += 1
            eval_list.append([targets,
                        self.output_layer.activations()])
            if show_sample_interval > 0:
                if count % show_sample_interval == 0:
                    #   Convert to logging at some point
                    print "sample: %s errors: %s" % (
                        count, summed_errors)

        self.mse = self.calc_mse(summed_errors, count)
        return self.mse

    @staticmethod
    def calc_mse(total_summed_errors, count):
        """
        This function calculates mean squared errors.
        """

        mse = total_summed_errors / float(count) / 2.0

        if math.isnan(mse):
            raise ValueError("Mean squared error is Nan")

        return mse

    def process_sample(self, inputs, targets, learn=False):
        """
        Accepts inputs and targets, then forward and back propagations.  A
        comparison is then made of the generated output with the target values.

        Note that this is for an incremental learn, not the full set of inputs
        and examples.

        """
        self.input_layer.load_inputs(inputs)

        if targets:
            #   Must be either learn or testing mode
            self.output_layer.load_targets(targets)

        self._feed_forward()

        if targets and learn:
            self._back_propagate()
        elif targets:
            self._update_error(toponly=True)

        self._copy_levels()

    def _feed_forward(self):
        """
        This function starts with the first hidden layer and
        gathers the values from the lower layer, applies the
        connection weightings to those values, and activates the
        nodes.  Then, the next layer up is selected and the process
        is repeated; resulting in output values in the upper-most
        layer.

        """

        for layer in self.layers[1:]:
            layer.feed_forward()

    def _back_propagate(self):
        """
        Backpropagate the error through the network. Aside from the
        initial compution of error at the output layer, the process takes the
        top hidden layer, looks at the output connections reaching up to the
        next layer, and carries the results down through each layer back to the
        input layer.

        """

        self._update_error(toponly=False)
        self._adjust_weights()

    def _update_error(self, toponly):
        """
        This function goes through layers starting with the top hidden layer
        and working its way down to the input layer.

        At each layer, the errors are updated in the nodes from the errors and
        weights in connections to nodes in the upper layer.

        """
        if toponly is False:
            self._zero_errors()

        if toponly:
            self.output_layer.update_error(self.get_halt_on_extremes())
        else:
            for layer_no in range(len(self.layers) - 1, -1, -1):
                self.layers[layer_no].update_error(self.get_halt_on_extremes())

    def _zero_errors(self):
        """
        This function sets the node errors to zero in preparation for back
        propagation.

        """

        for layer in self.layers:
            for node in layer.nodes:
                node.error = 0.0

    def _adjust_weights(self):
        """
        This function goes through layers starting with the top hidden layer
        and working its way down to the input layer.

        At each layer, the weights are adjusted based upon the errors.

        """

        for layer_no in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_no]
            layer.adjust_weights(self._learnrate, self.get_halt_on_extremes())

    def calc_sample_error(self):
        """
        The mean squared error (MSE) is a measure of how well the outputs
        compared to the target values.

        """

        total = 0.0
        for node in self.output_layer.nodes:
            total += pow(node.error, 2.0)

        return total

    def _copy_levels(self):
        """
        This function advances the copy node values, transferring the values
        from the source node to the copy node.  In order to avoid stomping on
        the values that are to be copies, it goes from highest node number to
        lowest.

        No provision is made at this point to exhaustively check precedence.

        """

        for layer in self.layers:
            for i in range(layer.total_nodes() - 1, -1, -1):
                node = layer.nodes[i]
                if isinstance(node, CopyNode):
                    node.load_source_value()

    def _parse_inputfile_layer(self, config, layer_no):
        """
        This function loads a layer and nodes from the input file. Note that
        it does not load the connections for those nodes here, waiting
        until all the nodes are fully instantiated.  This is because
        the connection objects have nodes as part of the object.

        """

        layer_id = 'layer %s' % (layer_no)
        layer_nodes = config.get(layer_id, 'nodes').split(" ")
        layer_type = config.get(layer_id, 'layer_type')
        layer = Layer(layer_no, layer_type)

        for node_id in layer_nodes:
            node = self._parse_inputfile_node(config, node_id)
            layer.add_node(node)

        self.layers.append(layer)

    @staticmethod
    def _parse_inputfile_node(config, node_id):
        """
        This function receives a node id, parses it, and returns the node in
        the network to which it pertains.  It implies that the network
        structure must already be in place for it to be functional.

        """

        activation_type = config.get(node_id, 'activation_type')

        node_type = config.get(node_id, 'node_type')

        if node_type == NODE_BIAS:
            node = BiasNode()
        elif node_type == NODE_COPY:
            node = CopyNode()
            node._source_type = config.get(node_id, 'source_type')
            node._incoming_weight = float(config.get(node_id,
                                            'incoming_weight'))

            node._existing_weight = float(config.get(node_id,
                                    'existing_weight'))
            node.set_activation_type(activation_type)
        else:
            node = Node()
            node.set_activation_type(activation_type)

        node.node_type = node_type

        return node

    def _parse_inputfile_conn(self, conn_strs, node):
        """
        This function instantiates a connection based upon the
        string loaded from the input file.
        Ex.  node-1:0, 0.166366874487
        """

        node_id, weight = conn_strs.split(',')
        weight = float(weight)
        layer_no, node_no = self._parse_node_id(node_id)
        lower_node = self.layers[layer_no].get_node(node_no)

        return Connection(lower_node, node, weight)

    def _parse_inputfile_copy(self, source_str):
        """
        This function instantiates a source node.

        """

        layer_no, node_no = self._parse_node_id(source_str)
        return self.layers[layer_no].get_node(node_no)

    @staticmethod
    def _parse_node_id(node_id):
        """
            This function parses the node_id received from the input file.
            Format of node id: 'node-%s:%s' % (layer_no, node_no)

            Returns layer_no and node_no
        """

        components = [i for i in node_id.split(":")]
        layer_no = int(components[0].split('-')[1])
        node_no = int(components[1])
        return (layer_no, node_no)

    def load(self, filename):
        """
        This function loads a file that has been saved by save funtion.  It is
        designed to be used when implementing a run-time version of the neural
        network.

        """

        config = ConfigParser.ConfigParser()
        config.readfp(open(filename))

        hidden_neurons = config.get('net', 'hidden_neurons').split(",")
        hidden_neurons = [int(item) for item in hidden_neurons]

        #   Load layer/nodes framework
        self.set_learnrate(float(config.get('net', 'learnrate')))
        self.set_epochs(int(config.get('net', 'epochs')))

        self.set_time_delay(int(config.get('net', 'time_delay')))

        flag = config.get('net', 'halt_on_extremes').title()
        if flag == "True":
            self.set_halt_on_extremes(True)
        elif flag == "False":
            self.set_halt_on_extremes(False)
        else:
            raise ValueError("Invalid halt on extremes flag")

        self.set_random_constraint(
            float(config.get('net', 'random_constraint')))

        total_layers = 1 + len(hidden_neurons) + 1
        for layer_no in range(total_layers):
            self._parse_inputfile_layer(config, layer_no)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        #   Attempt load of connections now that all nodes instantiated
        for layer_no in range(total_layers):
            layer = self.layers[layer_no]
            for node in layer.nodes:
                node_id = self._node_id(node)

                connections = config.get(node_id, 'connections').split('\n')
                for conn_str in connections:
                    if conn_str:
                        conn = self._parse_inputfile_conn(conn_str, node)
                        node.add_input_connection(conn)

                if isinstance(node, CopyNode):
                    #   load source node
                    source_str = config.get(node_id, 'source_node')
                    node.set_source_node(self._parse_inputfile_copy(
                        source_str))

        for layer_no in range(total_layers):
            layer = self.layers[layer_no]

    def output_values(self):
        """
        This function outputs the values of the network.  It is meant to be
        sufficiently complete that, once saved to a file, it could be loaded
        back from that file completely to function.

        To accommodate configparser, which is used as the file format, there is
        a form of [category],
                label = value

        Since there are also no sub-categories possible, so the naming
        structure is designed to take that into account. This accommodation
        also leads to a couple design choices: Each layer is given a separate
        category and a list of nodes follows.  Then each node has a separate
        category identifying it by layer number and node number.  This can't be
        inferred from just knowing the number of nodes in the layer and
        sequentially reading, because if a sparse network is used, then the
        node numbers may be out of sync with the position of the node within of
        the layer.
        """

        output = ''
        #   Overall parameters
        output += '[net]\n'

        #   Overall layer structure
        output += 'input_neurons = %s\n' % (
                self.input_layer.total_nodes(NODE_INPUT))
        output += 'hidden_neurons = %s\n' % (', '.join(
                [str(layer.total_nodes()) for layer in self.layers[1:-1]]))
        output += 'output_neurons = %s\n' % (
            self.output_layer.total_nodes(NODE_OUTPUT))

        # Note: this line will change if backpropagation through time (BPTT)
        # is implemented
        if self.layers[1].total_nodes(NODE_HIDDEN) > 0:
            copy_levels = self.input_layer.total_nodes(NODE_COPY) / \
                            self.layers[1].total_nodes(NODE_HIDDEN)
            output += 'copy_levels = %s\n' % (copy_levels)
        output += 'learnrate = %s\n' % (self._learnrate)
        output += 'epochs = %s\n' % (self._epochs)
        output += 'time_delay = %s\n' % (self._time_delay)
        output += 'halt_on_extremes = %s\n' % (self._halt_on_extremes)
        output += 'random_constraint = %s\n' % (self._random_constraint)
        output += '\n\n'

        # Layers
        for layer in self.layers:
            output += '[layer %s]\n' % (layer.layer_no)
            output += 'layer_type = %s\n' % (layer.layer_type)
            # Nodes
            output += 'nodes = '
            for node in layer.nodes:
                output += 'node-%s:%s ' % (layer.layer_no, node.node_no)

            output = output[:-1] + '\n'
            output += '\n\n'

            for node in layer.nodes:
                output += '[node-%s:%s]\n' % (layer.layer_no, node.node_no)
                output += 'node_type = %s\n' % (node.node_type)

                if isinstance(node, CopyNode):
                    snode = node.get_source_node()
                    output += 'source_node = %s\n' % (self._node_id(snode))
                    output += 'source_type = %s\n' % (node.get_source_type())
                    output += 'incoming_weight = %s\n' % (
                                node.get_incoming_weight())
                    output += 'existing_weight = %s\n' % (
                                node.get_existing_weight())
                    output += 'activation_type = %s\n' % (
                            node.get_activation_type())
                else:
                    output += 'activation_type = %s\n' % (
                            node.get_activation_type())

                # Connections
                output += 'connections = \n '
                for conn in node.input_connections:
                    lower_node = conn.lower_node
                    node_id = self._node_id(lower_node)
                    output += '%s, %s\n ' % (node_id, conn.get_weight())
                output += '\n'
            output += '\n'

        return output

    @staticmethod
    def _node_id(node):
        """
        This function receives a node, and returns an text based id that
        uniquely identifies it within the network.
        """

        return 'node-%s:%s' % (node.layer.layer_no, node.node_no)

    def save(self, filename):
        """
        This function saves the network structure to a file.
        """

        try:
            fobj = open(filename, "w")
        except:
            raise ValueError("Invalid filename")

        fobj.write(self.output_values())
        fobj.close()
