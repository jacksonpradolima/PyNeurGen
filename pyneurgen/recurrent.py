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
This module implements various approaches to recurrence.

        Elman Simple Recurrent Network:

        *    Source nodes are hidden
        *    One level of copy nodes
        *    Source Value is activation value
        *    Source value replaces existing copy node value
        *    Copy node activation is linear

        Jordan

        *    Souce nodes are output nodes
        *    One level of copy nodes
        *    Source Value is activation value
        *    Existing copy node value is discounted
                and combined with new source value

        NARX  Non-Linear AutoRegressive with eXogenous inputs

        *    Using the Narendra and Parthasathy variation
        *    Source nodes can come from outputs, inputs
                Outputs -- multple copies or orders
                Inputs -- multple copies
        *    Order == Number of copies
        *    Copy value can be discounted


"""

from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import CopyNode, Connection
from pyneurgen.nodes import NODE_OUTPUT, NODE_HIDDEN, NODE_INPUT
from pyneurgen.nodes import NODE_BIAS, ACTIVATION_LINEAR


class RecurrentConfig(object):
    """
    This is the base class for recurrent modifications.  It is not intended to
    be used directly.

    """

    def __init__(self):
        """
        This function initializes the configuration class.

        """

        self.source_type = 'a'
        self.activation_type = ACTIVATION_LINEAR
        self.incoming_weight = 1.0
        self.existing_weight = 0.0
        self.connection_type = 'm'
        self.copy_levels = 1
        self.copy_nodes_layer = 0
        self.connect_nodes_layer = 1

    def apply_config(self, neural_net):
        """
        This function modifies the neural net that is passed in by taking the
        parameters that have been set in this class.  By having _apply_config,
        subclassed versions of apply_config can take multiple passes with less
        code.

        """

        self._apply_config(neural_net)

    def _apply_config(self, neural_net):
        """
        This function actually does the work.

        """

        if not isinstance(neural_net, NeuralNet):
            raise ValueError("neural_net must be of the NeuralNet class.")
        for snode in self.get_source_nodes(neural_net):
            prev_copy_node = None
            for level in xrange(self.copy_levels):
                copy_node = CopyNode()
                if level == 0:
                    copy_node.set_source_node(snode)
                else:
                    copy_node.set_source_node(prev_copy_node)

                copy_node.source_update_config(
                    self.source_type,
                    self.incoming_weight,
                    self.existing_weight)

                copy_node.set_activation_type(self.activation_type)

                if self.connection_type == 'm':
                    self._fully_connect(
                        copy_node,
                        self.get_upper_nodes(neural_net))
                elif self.connection_type == 's':
                    copy_node.add_input_connection(
                        Connection(copy_node, snode))
                else:
                    raise ValueError("Invalid connection_type")

                neural_net.layers[self.copy_nodes_layer].add_node(copy_node)
                prev_copy_node = copy_node

    @staticmethod
    def _fully_connect(lower_node, upper_nodes):
        """
        This function creates connections to each of the upper nodes.

        This is a separate function from the one in layers, because using this
        version does not require ALL of the nodes on a layer to be used.

        """

        for upper_node in upper_nodes:
            upper_node.add_input_connection(Connection(lower_node, upper_node))

    def get_source_nodes(self, neural_net):
        """
        This function is a stub for getting the appropriate source nodes.

        """

        return neural_net

    def get_upper_nodes(self, neural_net):
        """
        This function is a stub for getting the appropriate nodes to which the
        copy nodes will connect.

        """
        layer = neural_net.layers[self.connect_nodes_layer]
        return [node for node in layer.nodes
                    if node.node_type != NODE_BIAS]


class ElmanSimpleRecurrent(RecurrentConfig):
    """
    This class implements a process for converting a standard neural network
    into an Elman Simple Recurrent Network.  The following is used to define
    such a configuration:
        Source nodes are nodes in the hidden layer.
        One level of copy nodes is used, in this situation referred to as
            context units.
        The source value from the hidden node is the activation value and the
            copy node (context) activation is linear; in other words simply a
            copy of the activation.  The source value replaces any previous
            value.

        In the case of multiple hidden layers, this class will take the lowest
        hidden layer.

        The class defaults to context nodes being fully connected to nodes in
        the hidden layer.

    """

    def __init__(self):
        """
        This function initializes the weights and default connection type
        consistent with an Elman Network.

        """

        RecurrentConfig.__init__(self)
        self.source_type = 'a'
        self.incoming_weight = 1.0
        self.existing_weight = 0.0
        self.connection_type = 'm'
        self.copy_levels = 1
        self.copy_nodes_layer = 0

    def get_source_nodes(self, neural_net):
        """
        This function returns the hidden nodes from layer 1.

        """

        return neural_net.layers[1].get_nodes(NODE_HIDDEN)


class JordanRecurrent(RecurrentConfig):
    """
    This class implements a process for converting a standard neural network
    into an Jordan style recurrent metwork.  The following is used to define
    such a configuration:

    *    Source nodes are nodes in the output layer.
    *    One level of copy nodes is used, in this situation referred to as
            context units.
    *    The source value from the output node is the activation value and the
            copy node (context) activation is linear; in other words simply a
            copy of the activation.

    *        The source value is added to the slightly discounted previous copy
            value.  So, the existing weight is some value less than 1.0 and
            greater than zero.

    *    In the case of multiple hidden layers, this class will take the lowest
        hidden layer.

    *    The class defaults to context nodes being fully connected to nodes in
        the output layer.

    """

    def __init__(self, existing_weight):
        """
        Initialization in this class means passing the weight that will be
        multiplied time the existing value in the copy node.

        """
        RecurrentConfig.__init__(self)
        self.source_type = 'a'
        self.incoming_weight = 1.0
        self.existing_weight = existing_weight
        self.connection_type = 'm'
        self.copy_levels = 1
        self.copy_nodes_layer = 0

    def get_source_nodes(self, neural_net):
        """
        This function returns the output nodes.

        """

        return neural_net.layers[-1].get_nodes(NODE_OUTPUT)


class NARXRecurrent(RecurrentConfig):
    """
    This class implements a process for converting a standard neural network
    into a NARX (Non-Linear AutoRegressive with eXogenous inputs) recurrent
    network.

    It also contains some modifications suggested by Narendra and Parthasathy
    (1990).

    Source nodes can come from outputs and inputs.  There can be multiple
    levels of copies (or order in this nomenclature) from either outputs or
    inputs.

    The source value can be weighted fully, or the incoming weight adjusted
    lower.

    This class applies changes to the neural network by first applying the
    configurations related to the output nodes and then to the input nodes.

    """

    def __init__(self, output_order, incoming_weight_from_output,
                      input_order, incoming_weight_from_input):
        """
        This function takes:
            the output order, or number of copy levels of
                output values,
            the weight to apply to the incoming values from output nodes,
            the input order, or number of copy levels of input values,
            the weight to apply to the incoming values from input nodes

        """
        RecurrentConfig.__init__(self)
        self.existing_weight = 0.0
        self._node_type = None

        self.output_values = [output_order, incoming_weight_from_output]
        self.input_values = [input_order, incoming_weight_from_input]

    def get_source_nodes(self, neural_net):
        """
        This function returns either the output nodes or input nodes depending
        upon self._node_type.

        """

        if self._node_type == NODE_OUTPUT:
            return neural_net.layers[-1].get_nodes(self._node_type)
        elif self._node_type == NODE_INPUT:
            return neural_net.layers[0].get_nodes(self._node_type)

    def apply_config(self, neural_net):
        """
        This function first applies any parameters related to the output nodes
        and then any with the input nodes.

        """

        if self.output_values[0] > 0:
            self._node_type = NODE_OUTPUT
            self.copy_levels = self.output_values[0]
            self.incoming_weight = self.output_values[1]

            self._apply_config(neural_net)

        if self.input_values[0] > 0:
            self._node_type = NODE_INPUT
            self.copy_levels = self.input_values[0]
            self.incoming_weight = self.input_values[1]

            self._apply_config(neural_net)
