import unittest

from copy import deepcopy

from pyneurgen.layers import Layer
from pyneurgen.nodes import Node, CopyNode, BiasNode, Connection
from pyneurgen.nodes import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from pyneurgen.nodes import linear, linear_derivative


class TestLayer(unittest.TestCase):
    """
    Tests Layer

    """

    def test__init__(self):

        self.assertEqual('input', Layer(0, 'input').layer_type)
        self.assertEqual('hidden', Layer(1, 'hidden').layer_type)
        self.assertEqual('output', Layer(2, 'output').layer_type)

        layer = Layer(0, 'input')
        self.assertEqual('linear', layer.default_activation_type)
        layer = Layer(0, 'hidden')
        self.assertEqual('sigmoid', layer.default_activation_type)
        layer = Layer(0, 'output')
        self.assertEqual('linear', layer.default_activation_type)

        self.failUnlessRaises(ValueError, Layer, 0, 'test')
        self.failUnlessRaises(ValueError, Layer, 1, 'input')

        layer = Layer(0, 'input')

    def test_total_nodes(self):

        layer = Layer(0, 'input')
        layer.add_nodes(2, 'input')
        layer.add_nodes(2, 'copy')
        layer.add_node(BiasNode())

        self.assertEqual(5, layer.total_nodes())
        self.assertEqual(2, layer.total_nodes('input'))
        self.assertEqual(2, layer.total_nodes('copy'))
        self.assertEqual(0, layer.total_nodes('hidden'))


    def test_unconnected_nodes(self):

        layer = Layer(1, 'hidden')
        conn = Connection(Node(), Node())

        layer.add_nodes(2, 'hidden')

        layer.nodes[0].add_input_connection(
            Connection(Node(), layer.nodes[0]))
        input_side = layer.unconnected_nodes()

        self.assertEqual(1, input_side[0])
        self.assertNotEqual(0, input_side[0])

    def test_values(self):

        layer = Layer(1, 'hidden')
        layer.add_nodes(2, 'hidden')

        layer.nodes[0].set_value(.2)
        layer.nodes[1].set_value(.3)

        values = layer.values()

        self.assertEqual(True, isinstance(values, list))
        self.assertEqual(.2, values[0])
        self.assertEqual(.3, values[1])

    def test_activations(self):

        layer = Layer(1, 'hidden')
        layer.add_nodes(2, 'hidden')
        layer.set_activation_type('linear')

        layer.nodes[0].set_value(.2)
        layer.nodes[1].set_value(.3)

        activations = layer.activations()

        self.assertEqual(True, isinstance(activations, list))
        self.assertEqual(.2, activations[0])
        self.assertEqual(.3, activations[1])

    def test_set_activation_type(self):

        layer = Layer(1, 'hidden')
        layer.add_nodes(1, 'hidden')

        self.assertEqual('sigmoid', layer.nodes[0].get_activation_type())

        layer.set_activation_type('linear')

        self.assertEqual('linear', layer.nodes[0].get_activation_type())

        self.failUnlessRaises(
            ValueError,
            layer.set_activation_type, 'fail')

    def test_add_nodes(self):

        layer = Layer(0, 'input')

        layer.add_nodes(1, 'input')
        layer.add_nodes(1, 'copy')

        self.assertEqual(2, len(layer.nodes))
        self.assertEqual('copy', layer.nodes[1].node_type)
        self.assertNotEqual('copy', layer.nodes[0].node_type)

    def test_add_node(self):

        layer = Layer(0, 'input')
        layer.default_activation_type = 'linear'
        node = Node()
        layer.add_node(node)

        self.assertEqual(1, layer.total_nodes())
        self.assertEqual(0, layer.nodes[0].node_no)
        self.assertEqual('linear', layer.nodes[0].get_activation_type())

        layer.default_activation_type = 'sigmoid'
        node = Node()
        layer.add_node(node)

        self.assertEqual(2, layer.total_nodes())
        self.assertEqual(1, layer.nodes[1].node_no)
        self.assertEqual('sigmoid', layer.nodes[1].get_activation_type())

        node = BiasNode()
        layer.add_node(node)

        self.assertEqual(3, layer.total_nodes())
        self.assertEqual(2, layer.nodes[2].node_no)

        node = Node()
        node.set_activation_type('tanh')
        layer.add_node(node)

        self.assertEqual('tanh', layer.nodes[3].get_activation_type())

    def test_get_node(self):

        layer = Layer(0, 'input')
        layer.add_nodes(6, 'input')

        del(layer.nodes[3])

        node = layer.get_node(4)
        self.assertEqual(node, layer.nodes[3])

    def test_get_nodes(self):

        pass

    def test_connect_layer(self):

        pass

    def test_load_inputs(self):

        pass

    def test_load_targets(self):

        pass

    def test_randomize(self):

        pass

    def test_feed_forward(self) :

        layer0 = Layer(0, 'input')
        layer0.add_nodes(2, 'input')
        layer0.set_activation_type('sigmoid')

        layer1 = Layer(1, 'hidden')
        layer1.add_nodes(1, 'hidden')

        inode1 = layer0.nodes[0]
        inode2 = layer0.nodes[1]

        inode1.set_value(.25)
        inode2.set_value(.5)

        node = layer1.nodes[0]
        node.add_input_connection(
            Connection(inode1, node, .25))
        node.add_input_connection(
            Connection(inode2, node, .5))

        layer1.feed_forward()

        self.assertAlmostEqual(
            sigmoid(.25) * .25 + sigmoid(.5) * .5, node.get_value())


    def test_update_error(self):

        pass
       ##    At this level test to see if went through all nodes
        #layer = Layer(0, 'input')
        #layer.add_nodes(2, 'input')

        #def update_error(node):
            #node.error = 1

        #for node in layer.nodes:
            #node.error = 1000
            #node.update_error = update_error

        #layer.update_error(True)

        #self.assertEqual(1, layer.nodes[0].error)
        #self.assertEqual(1, layer.nodes[1].error)

    def test_adjust_weights(self):

       pass

    def test_get_errors(self):

       pass

    def test_get_weights(self):

       pass

if __name__ == '__main__':
    unittest.main()
