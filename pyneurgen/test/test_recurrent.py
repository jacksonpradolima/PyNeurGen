import unittest
from os import sys
sys.path.append(r'../')

from pyneurgen.recurrent import RecurrentConfig, ElmanSimpleRecurrent
from pyneurgen.recurrent import JordanRecurrent, NARXRecurrent

from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import Node, CopyNode, Connection
from pyneurgen.nodes import NODE_OUTPUT, NODE_HIDDEN, NODE_INPUT, NODE_COPY, NODE_BIAS


class RecurrentConfigTest(unittest.TestCase):
    """
    Tests RecurrentConfig

    """

    def setUp(self):

        self.net = NeuralNet()
        self.net.init_layers(2, [1], 1)

        self.rec_config = RecurrentConfig()

    def test_apply_config(self):

        self.assertRaises(
                ValueError,
                self.rec_config.apply_config, 'not neural net')

    def test__apply_config(self):

        print 'test__apply_config not yet implemented'

    def test_fully_connect(self):

        node = Node()
        unode1 = Node()
        unode2 = Node()

        self.rec_config._fully_connect(node, [unode1, unode2])

        conn = unode1.input_connections[0]
        self.assertEqual(node, conn.lower_node)
        self.assertEqual(unode1, conn.upper_node)
        conn = unode2.input_connections[0]
        self.assertEqual(node, conn.lower_node)
        self.assertEqual(unode2, conn.upper_node)

    def test_get_source_nodes(self):

        self.assertEqual(True, isinstance(
                                    self.rec_config.get_source_nodes(self.net),
                                    NeuralNet))

    def test_get_upper_nodes(self):

        self.assertEqual(1, len(self.rec_config.get_upper_nodes(self.net)))


class ElmanSimpleRecurrentTest(unittest.TestCase):
    """
    Tests ElmanSimpleRecurrent

    """

    def setUp(self):

        self.net = NeuralNet()
        self.net.init_layers(2, [1], 1)

        self.rec_config = ElmanSimpleRecurrent()

    def test_class_init_(self):

        self.assertEqual('a', self.rec_config.source_type)
        self.assertEqual(1.0, self.rec_config.incoming_weight)
        self.assertEqual(0.0, self.rec_config.existing_weight)
        self.assertEqual('m', self.rec_config.connection_type)
        self.assertEqual(1, self.rec_config.copy_levels)
        self.assertEqual(0, self.rec_config.copy_nodes_layer)

    def test_get_source_nodes(self):

        nodes1 = self.net.layers[1].get_nodes(NODE_HIDDEN)
        nodes2 = self.rec_config.get_source_nodes(self.net)

        #   Should be the same
        self.assertEqual(len(nodes1), len(nodes2))
        self.assertEqual(
            self.net.layers[1].get_nodes(NODE_HIDDEN),
            self.rec_config.get_source_nodes(self.net))


class JordanRecurrentTest(unittest.TestCase):
    """
    Tests JordanRecurrent

    """

    def setUp(self):

        self.net = NeuralNet()
        self.net.init_layers(2, [1], 1)

        self.rec_config = JordanRecurrent(existing_weight=.8)

    def test_class_init_(self):

        self.assertEqual('a', self.rec_config.source_type)
        self.assertEqual(1.0, self.rec_config.incoming_weight)
        self.assertEqual(0.8, self.rec_config.existing_weight)
        self.assertEqual('m', self.rec_config.connection_type)
        self.assertEqual(1, self.rec_config.copy_levels)
        self.assertEqual(0, self.rec_config.copy_nodes_layer)

    def test_get_source_nodes(self):

        self.assertEqual(
            self.net.layers[2].nodes,
            self.rec_config.get_source_nodes(self.net))


class NARXRecurrentTest(unittest.TestCase):
    """
    Tests NARXRecurrent

    """

    def setUp(self):

        self.net = NeuralNet()
        self.net.init_layers(2, [1], 1)

        self.rec_config = NARXRecurrent(
                                        output_order=1,
                                        incoming_weight_from_output=.9,
                                        input_order=1,
                                        incoming_weight_from_input=.7)

    def test_class_init_(self):

        self.assertEqual(0, self.rec_config.existing_weight)
        self.assertEqual(None, self.rec_config._node_type)
        self.assertEqual([1, .9], self.rec_config.output_values)
        self.assertEqual([1, .7], self.rec_config.input_values)

    def test_get_source_nodes(self):

        self.rec_config._node_type = NODE_OUTPUT
        self.assertEqual(
            self.net.layers[-1].get_nodes(NODE_OUTPUT),
            self.rec_config.get_source_nodes(self.net))

        self.rec_config._node_type = NODE_INPUT
        self.assertEqual(
            self.net.layers[0].get_nodes(NODE_INPUT),
            self.rec_config.get_source_nodes(self.net))


if __name__ == '__main__':
    unittest.main()
