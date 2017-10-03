import unittest

from pyneurgen.nodes import ProtoNode, Node, CopyNode, BiasNode, Connection
from pyneurgen.nodes import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from pyneurgen.nodes import linear, linear_derivative
from pyneurgen.nodes import NODE_OUTPUT, NODE_HIDDEN, NODE_INPUT, NODE_COPY, NODE_BIAS
from pyneurgen.nodes import ACTIVATION_SIGMOID, ACTIVATION_TANH, ACTIVATION_LINEAR


class ProtoNodeTest(unittest.TestCase):
    """
    Tests ProtoNode

    """

    def setUp(self):

        self.node = ProtoNode()
        self.node._activation_type = ACTIVATION_SIGMOID
        self.node._error_func = sigmoid_derivative

    def test_get_value(self):

        self.node._value = .4
        self.assertEqual(.4, self.node.get_value())

    def test_randomize(self):

        pass

    def test_get_activation_type(self):
        """
        This function returns the activation type of the node.

        """
        self.assertEqual(ACTIVATION_SIGMOID, self.node.get_activation_type())

class NodeTest(unittest.TestCase):
    """
    Tests Node

    """

    def setUp(self):

        self.node = Node()

        node1 = Node()
        node2 = Node()

        node1._value = .2
        node2._value = .1
        node1.error = .8
        node2.error = .4

        node1.set_activation_type(ACTIVATION_SIGMOID)
        node2.set_activation_type(ACTIVATION_SIGMOID)

        self.node.add_input_connection(
            Connection(node1, self.node, .3))
        self.node.input_connections.append(
            Connection(node2, self.node, .7))

    def test_set_activation_type(self):

        self.node._activate = 'error'
        self.node._error_func = 'error'
        self.node._activation_type = 'error'

        self.node.set_activation_type(ACTIVATION_SIGMOID)

        self.assertEqual(sigmoid, self.node._activate)
        self.assertEqual(sigmoid_derivative, self.node._error_func)
        self.assertEqual(ACTIVATION_SIGMOID, self.node._activation_type)

        self.node._activate = 'error'
        self.node._error_func = 'error'
        self.node._activation_type = 'error'

        self.node.set_activation_type(ACTIVATION_TANH)

        self.assertEqual(tanh, self.node._activate)
        self.assertEqual(tanh_derivative, self.node._error_func)
        self.assertEqual(ACTIVATION_TANH, self.node._activation_type)

        self.node._activate = 'error'
        self.node._error_func = 'error'
        self.node._activation_type = 'error'

        self.node.set_activation_type(ACTIVATION_LINEAR)

        self.assertEqual(linear, self.node._activate)
        self.assertEqual(linear_derivative, self.node._error_func)
        self.assertEqual(ACTIVATION_LINEAR, self.node._activation_type)

        self.failUnlessRaises(
                ValueError,
                self.node.set_activation_type, 'error')

    def test_set_error_func(self):

        self.node._error_func = 'error'
        self.node._set_error_func(ACTIVATION_SIGMOID)
        self.assertEqual(sigmoid_derivative, self.node._error_func)

        self.node._error_func = 'error'
        self.node._set_error_func(ACTIVATION_TANH)
        self.assertEqual(tanh_derivative, self.node._error_func)

        self.node._error_func = 'error'
        self.node._set_error_func(ACTIVATION_LINEAR)
        self.assertEqual(linear_derivative, self.node._error_func)

    def test_set_value(self):

        self.node._value = .2
        self.node.set_value(.3)
        self.assertAlmostEqual(.3, self.node._value)

    def test_get_value(self):

        self.node._value = .2
        self.assertAlmostEqual(.2, self.node.get_value())

    def test_error_func(self):

        self.node.set_activation_type(ACTIVATION_SIGMOID)
        self.assertAlmostEqual(
            sigmoid_derivative(.2),
            self.node.error_func(.2))

    def test_feed_forward(self):

        self.node_value = 1000.0

        self.node.feed_forward()

        total = sigmoid(.2) * .3 + sigmoid(.1) * .7

        self.assertAlmostEqual(total, self.node._value)

    def test__init__(self):

        self.node = Node('test')
        self.assertEqual('test', self.node.node_type)

    def test_add_input_connection(self):

        connections = len(self.node.input_connections)
        self.node.add_input_connection(
            Connection(ProtoNode(), self.node))

        self.assertEqual(connections + 1, len(self.node.input_connections))


        self.failUnlessRaises(
            ValueError,
            self.node.add_input_connection, Connection(Node(), Node()))

    def test_update_error(self):

        # upper_node1.error = .8
        # upper_node2.error = .4
        # conn1 weight = .3
        # conn2 weight = .7

        self.node.node_type = NODE_OUTPUT
        self.node.set_activation_type(ACTIVATION_SIGMOID)
        halt_on_extremes = True
        self.node._value = .4
        self.node.target = .55
        self.node.error = 0.0

        self.node.update_error(halt_on_extremes)

        self.assertAlmostEqual(.55 - sigmoid(.4), self.node.error)

        #
        self.node.node_type = NODE_HIDDEN
        self.node.set_activation_type(ACTIVATION_SIGMOID)
        halt_on_extremes = True
        self.node._value = .4
        self.node.error = .55

        self.node.update_error(halt_on_extremes)

        self.assertAlmostEqual(
            .55  * sigmoid_derivative(sigmoid(.4)),
            self.node.error)


    def test__update_lower_node_errors(self):

        self.node.error = .55
        halt_on_extremes = True

        node1 = self.node.input_connections[0].lower_node
        node2 = self.node.input_connections[1].lower_node

        node1.error = 0.0
        node2.error = 0.0

        self.node._update_lower_node_errors(halt_on_extremes)

        self.assertAlmostEqual(
            .3 * .55,
            self.node.input_connections[0].lower_node.error)

        self.assertAlmostEqual(
            .7 * .55,
            self.node.input_connections[1].lower_node.error)

    def test_adjust_weights(self):

        learnrate = .35
        halt_on_extremes = True
        self.node.error = .9
        self.node.set_activation_type(ACTIVATION_SIGMOID)

        #   adjusts incoming values
        conn1 = .3 + .35 * sigmoid(.2) * .9
        conn2 = .7 + .35 * sigmoid(.1) * .9

        self.node.adjust_weights(learnrate, halt_on_extremes)

        self.assertAlmostEqual(
            conn1,
            self.node.input_connections[0]._weight)

        self.assertAlmostEqual(
            conn2,
            self.node.input_connections[1]._weight)

    def test__adjust_weight(self):

        # learnrate = .20
        # activate_value = .25
        # error = .10

        self.assertAlmostEqual(
            .20 * .25 * .10,
            self.node._adjust_weight(.20, .25, .10))


class CopyNodeTest(unittest.TestCase):
    """
    Tests CopyNode

    """

    def setUp(self):
        self.node = CopyNode()

    def test__init__(self):
        self.assertEqual(NODE_COPY, self.node.node_type)

    def test_set_source_node(self):

        source_node = Node()
        self.node.set_source_node(source_node)

        self.assertEqual(source_node, self.node._source_node)

    def test_get_source_node(self):

        self.node._source_node = Node()
        self.assertEqual(self.node._source_node, self.node.get_source_node())

    def test_load_source_value(self):

        self.node._value = .25
        self.node._existing_weight = .25
        self.node._incoming_weight = .5

        source_node = Node()
        source_node.set_value(.3)
        source_node.set_activation_type(ACTIVATION_SIGMOID)
        self.node.set_source_node(source_node)

        #   activate
        self.node._source_type = 'a'
        self.node.load_source_value()
        self.assertAlmostEqual(sigmoid(.3) * .5 + .25 * .25, self.node._value)

        #   value
        self.node._value = .25
        self.node._source_type = 'v'
        self.node.load_source_value()
        self.assertAlmostEqual(.3 * .5 + .25 * .25, self.node._value)

        #   invalid source type
        self.node._source_type = 'f'
        self.failUnlessRaises(ValueError, self.node.load_source_value)

    def test_get_source_type(self):
        self.node._source_type = 'a'
        self.assertEqual('a', self.node.get_source_type())

    def test_get_incoming_weight(self):
        self.node._incoming_weight = .3
        self.assertAlmostEqual(.3, self.node.get_incoming_weight())

    def test_get_existing_weight(self):
        self.node._existing_weight = .3
        self.assertAlmostEqual(.3, self.node.get_existing_weight())

    def test_source_update_config(self):

        self.node.source_update_config('a', .3, .2)
        self.assertEqual('a', self.node._source_type)
        self.assertAlmostEqual(.3, self.node._incoming_weight)
        self.assertAlmostEqual(.2, self.node._existing_weight)

        self.failUnlessRaises(
                ValueError, self.node.source_update_config, 'e', .3, .2)
        self.failUnlessRaises(
                ValueError, self.node.source_update_config, 'a', 1.3, .2)
        self.failUnlessRaises(
                ValueError, self.node.source_update_config, 'a', .3, 1.2)


class BiasNodeTest(unittest.TestCase):
    """
    Tests BiasNode

    """

    def setUp(self):

        self.node = BiasNode()

    def test__init__(self):

        self.assertEqual(NODE_BIAS, self.node.node_type)
        self.assertEqual(1.0, self.node._value)
        self.assertEqual(1.0, self.node._activated)

    def test_activate(self):

        self.assertEqual(1.0, self.node.activate())

    def test_error_func(self):

        #   should always be 1.0
        self.assertEqual(1.0, self.node.error_func(.3))


class ConnectionTest(unittest.TestCase):
    """
    This class tests the Connection class

    """

    def setUp(self):

        self.upper_node = Node()
        self.lower_node = Node()

        self.upper_node._value = .2
        self.lower_node._value = .1
        self.upper_node.error = .8

        self.conn = Connection(self.lower_node, self.upper_node)

    def test_set_weight(self):
        self.conn.set_weight(.3)
        self.assertAlmostEqual(.3, self.conn._weight)

    def test_add_weight(self):
        self.conn.set_weight(.3)
        self.conn.add_weight(.3)
        self.assertAlmostEqual(.6, self.conn._weight)

    def test_get_weight(self):
        self.conn.set_weight(.3)
        self.assertAlmostEqual(.3, self.conn.get_weight())


##   remaining tests
#def test sigmoid(value):

    #pass

#def sigmoid_derivative(value):

    #pass

#def tanh(value):

    #pass

#def tanh_derivative(value):

    #pass

#def linear(value):

    #pass

#def linear_derivative(value):

    #pass

#nodesTestSuite = unittest.TestSuite()
#nodesTestSuite.addTest(ProtoNodeTest('proto_node_test'))
#nodesTestSuite.addTest(NodeTest('node_test'))
#nodesTestSuite.addTest(BiasNodeTest('bias_node_test'))
#nodesTestSuite.addTest(ConnectionTest('connection_test'))




if __name__ == '__main__':
    unittest.main()
