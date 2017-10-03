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
This module implements a layer class for an artficial neural network.

"""

from pyneurgen.nodes import Node, CopyNode, BiasNode, Connection

LAYER_TYPE_INPUT = 'input'
LAYER_TYPE_HIDDEN = 'hidden'
LAYER_TYPE_OUTPUT = 'output'


class Layer(object):
    """
    A layer comprises a list of nodes and behaviors appropriate for their
    place in the hierarchy.  A layer_type can be either 'input', 'hidden',
    or 'output'.

    """

    def __init__(self, layer_no, layer_type):
        """
        The layer class initializes with the layer number and the type of
        layer.  Lower layer numbers are toward the input end of the network,
        with higher numbers toward the output end.
        """
        self.nodes = []
        self.layer_no = layer_no

        if layer_type in [LAYER_TYPE_INPUT, LAYER_TYPE_HIDDEN,
                            LAYER_TYPE_OUTPUT]:
            self.layer_type = layer_type
        else:
            raise ValueError(
            "Layer type must be 'input', 'hidden', or 'output'")

        if layer_type == LAYER_TYPE_INPUT and layer_no != 0:
            raise ValueError("the input layer must always be layer_no 0")

        if self.layer_type == LAYER_TYPE_INPUT:
            self.default_activation_type = 'linear'
        elif self.layer_type == LAYER_TYPE_OUTPUT:
            self.default_activation_type = 'linear'
        else:
            self.default_activation_type = 'sigmoid'

        self.set_activation_type(self.default_activation_type)

    def total_nodes(self, node_type=None):
        """
        This function returns the total nodes.  It can also return the total
        nodes of a particular type, such as 'copy'.

        """

        count = 0
        if node_type:
            for node in self.nodes:
                if node.node_type == node_type:
                    count += 1
            return count
        else:
            return len(self.nodes)

    def unconnected_nodes(self):
        """
        This function looks for nodes that do not have an input
        connection.

        """

        return [node.node_no for node in self.nodes
            if not node.input_connections]

    def values(self):
        """
        This function returns the values for each node as a list.

        """

        return [node.get_value() for node in self.nodes]

    def activations(self):
        """
        This function returns the activation values for each node as a list.

        """

        return [node.activate() for node in self.nodes]

    def set_activation_type(self, activation_type):
        """
        This function is a mechanism for setting the activation type
        for an entire layer.  If most nodes need to one specific type,
        this function can be used, then set whatever nodes individually
        after this use.

        """

        for node in self.nodes:
            if node.node_type != 'bias':
                node.set_activation_type(activation_type)

    def add_nodes(self, number_nodes, node_type, activation_type=None):
        """
        This function adds nodes in bulk for initialization.

        If an optional activation type is passed through, that will be set for
        the nodes.  Otherwise, the default activation type for the layer will
        be used.

        """

        count = 0
        while count < number_nodes:
            if node_type == 'copy':
                node = CopyNode()
            else:
                node = Node(node_type)

            if activation_type:
                node.set_activation_type(activation_type)

            self.add_node(node)
            count += 1

    def add_node(self, node):
        """
        This function adds a node that has already been formed.  Since it can
        originate outside of the initialization process, the activation type is
        assumed to be set appropriately already.

        """

        node.node_no = self.total_nodes()
        if node.node_type != 'bias':
            if not node.get_activation_type():
                node.set_activation_type(self.default_activation_type)
        node.layer = self
        self.nodes.append(node)

    def get_node(self, node_no):
        """
        This function returns the node associated with the node_no.
        Although it would seem to be reasonable to look it up by
        position within the node list, because sparse nodes are supported,
        there might be a mis-match between node_no and position within the
        list.

        """

        for node in self.nodes:
            if node.node_no == node_no:
                return node

        return False

    def get_nodes(self, node_type=None):
        """
        This function returns all the nodes of a layer.  Optionally it can
        return all of the nodes of a particular type, such as 'copy'.

        """

        if node_type is None:
            return [node for node in self.nodes]
        else:
            return [node for node in self.nodes if node.node_type == node_type]

    def connect_layer(self, lower_layer):
        """
        This function accepts a lower layer within a network and for each node
        in that layer connects the node to nodes in the current layer.

        An exception is made for bias nodes. There is no reason to
        connect a bias node to a lower layer, since it always produces a 1.0
        for its value and activation.

        """

        for node in self.nodes:
            if node.node_type != 'bias':
                for lower_node in lower_layer.nodes:
                    conn = Connection(lower_node, node)
                    node.add_input_connection(conn)

    def load_inputs(self, inputs):
        """
        This takes a list of inputs that applied sequentially to
        each node in the input_layer

        """

        if self.layer_type != LAYER_TYPE_INPUT:
            raise ValueError("inputs are only entered into the input layer")

        for i in range(len(inputs)):
            node = self.nodes[i]
            if node.node_type != LAYER_TYPE_INPUT:
                raise ValueError(
                    "Attempting to load an input value into a non-input node")
            if isinstance(inputs[i], float):
                node.set_value(inputs[i])
            else:
                raise ValueError(
                    "Invalid value, most be float: %s" % (inputs[i]))

    def load_targets(self, targets):
        """
        This takes a list of targets that applied sequentially to
        each node in the output_layer

        """

        if self.layer_type != LAYER_TYPE_OUTPUT:
            raise ValueError(
                "target values are only loaded to the output layer")

        if len(targets) != len(self.nodes):
            raise ValueError(
                "Number of targets: %s, Number of nodes: %s""" % (
                    (len(targets), len(self.nodes))))
        for i in range(self.total_nodes()):
            node = self.nodes[i]
            if isinstance(targets[i], float):
                node.set_value(targets[i])
            else:
                raise ValueError(
                    "Invalid value, most be float: %s" % (targets[i]))
            node = self.nodes[i]
            node.target = targets[i]

    def randomize(self, random_constraint):
        """
        This function builds random weights for all the input connections in
        the layer.

        """

        for node in self.nodes:
            node.randomize(random_constraint)

    def feed_forward(self):
        """
        This function loops through the nodes on the layer and causes each
        node to feedforward values from nodes below that node.

        """

        for node in self.nodes:
            if not isinstance(node, BiasNode):
                node.feed_forward()

    def update_error(self, halt_on_extremes):
        """
        This function loops through the nodes on the layer and causes each
        node to update errors as part of the back propagation process.

        """

        for node in self.nodes:
            node.update_error(halt_on_extremes)

    def adjust_weights(self, learnrate, halt_on_extremes):
        """
        This function loops through the nodes causing each node to adjust the
        weights as a result of errors and the learning rate.

        """

        for node in self.nodes:
            if node.node_type != 'bias':
                node.adjust_weights(learnrate, halt_on_extremes)

    def get_errors(self):
        """
        This function returns a list of the error with each node.

        """

        return [node.error for node in self.nodes]

    def get_weights(self):
        """
        This function returns a list of the weights of input connections into
        each node in the layer.

        """
        weights = []

        for node in self.nodes:
            for conn in node.input_connections:
                weights.append(conn.get_weight())

        return weights
