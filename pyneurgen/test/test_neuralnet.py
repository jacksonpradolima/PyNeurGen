import unittest

from pyneurgen.neuralnet import NeuralNet
from pyneurgen.layers import Layer
from pyneurgen.nodes import Node, CopyNode, BiasNode, Connection
from pyneurgen.nodes import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from pyneurgen.nodes import linear, linear_derivative


class TestNeuralNet(unittest.TestCase):
    """
    Tests NeuralNet

    """

    def setUp(self):

        self.net = NeuralNet()

        layer = Layer(0, 'input')
        layer.add_nodes(1, 'input')
        self.net.layers.append(layer)

        layer = Layer(1, 'hidden')
        layer.add_nodes(1, 'hidden')
        self.net.layers.append(layer)

        layer = Layer(2, 'output')
        layer.add_nodes(1, 'output')
        self.net.layers.append(layer)

        #   Specify connections
        self.net.layers[1].nodes[0].add_input_connection(
            Connection(
                self.net.layers[0].nodes[0],
                self.net.layers[1].nodes[0],
                1.00))

        self.net.layers[2].nodes[0].add_input_connection(
            Connection(
                self.net.layers[1].nodes[0],
                self.net.layers[2].nodes[0],
                .75))

        self.net._epochs = 1
        self.net.copy_levels = 0
        self.net._allinputs = [[.1], [.2], [.3], [.4], [.5]]
        self.net._alltargets = [[.2], [.4], [.6], [.8], [1.0]]

        self.net.input_layer = self.net.layers[0]
        self.net.output_layer = self.net.layers[-1]

    def test_set_halt_on_extremes(self):

        self.net._halt_on_extremes = 'fail'
        self.net.set_halt_on_extremes(True)
        self.assertEqual(True, self.net._halt_on_extremes)

        self.net._halt_on_extremes = 'fail'
        self.net.set_halt_on_extremes(False)
        self.assertEqual(False, self.net._halt_on_extremes)

        self.net._halt_on_extremes = 'fail'
        self.failUnlessRaises(ValueError, self.net.set_halt_on_extremes, 'a')

        self.net._halt_on_extremes = 'fail'
        self.failUnlessRaises(ValueError, self.net.set_halt_on_extremes, 3)

    def test_get_halt_on_extremes(self):

        self.net.set_halt_on_extremes(True)
        self.assertEqual(True, self.net.get_halt_on_extremes())

        self.net.set_halt_on_extremes(False)
        self.assertEqual(False, self.net.get_halt_on_extremes())

    def test_set_random_constraint(self):

        self.net._random_constraint = 'fail'
        self.net.set_random_constraint(.1)
        self.assertEqual(.1, self.net._random_constraint)

        self.failUnlessRaises(ValueError, self.net.set_random_constraint, 3)
        self.failUnlessRaises(ValueError, self.net.set_random_constraint, 1)
        self.failUnlessRaises(ValueError, self.net.set_random_constraint, 0.0)
        self.failUnlessRaises(ValueError, self.net.set_random_constraint, -.2)
        self.failUnlessRaises(ValueError, self.net.set_random_constraint, 'a')

    def test_get_random_constraint(self):

        self.net.set_random_constraint(.2)
        self.assertEqual(.2, self.net.get_random_constraint())

        self.net.set_random_constraint(.8)
        self.assertEqual(.8, self.net.get_random_constraint())

    def test_set_epochs(self):

        self.net._epochs = 'fail'
        self.net.set_epochs(3)
        self.assertEqual(3, self.net._epochs)

        self.failUnlessRaises(ValueError, self.net.set_epochs, .3)
        self.failUnlessRaises(ValueError, self.net.set_epochs, 0)
        self.failUnlessRaises(ValueError, self.net.set_epochs, -3)
        self.failUnlessRaises(ValueError, self.net.set_epochs, -.2)
        self.failUnlessRaises(ValueError, self.net.set_epochs, 'a')

    def test_get_epochs(self):

        self.net.set_epochs(3)
        self.assertEqual(3, self.net.get_epochs())

    def test_set_time_delay(self):

        self.net._time_delay = 'fail'
        self.net.set_time_delay(3)
        self.assertEqual(3, self.net._time_delay)

        self.failUnlessRaises(ValueError, self.net.set_time_delay, .3)
        self.failUnlessRaises(ValueError, self.net.set_time_delay, -3)
        self.failUnlessRaises(ValueError, self.net.set_time_delay, -.2)
        self.failUnlessRaises(ValueError, self.net.set_time_delay, 'a')

    def test_get_time_delay(self):

        self.net.set_time_delay(3)
        self.assertEqual(3, self.net.get_time_delay())

    def test_set_all_inputs(self):

        pass

    def test_set_all_targets(self):

        pass

    def test_set_learnrate(self):

        pass

    def test_get_learnrate(self):

        pass

    def test__set_data_range(self):

        pass

    def test_set_learn_range(self):

        pass

    def test_get_learn_range(self):

        pass

    def test__check_time_delay(self):

        pass

    def test_get_learn_data(self):

        pass

    def test_get_validation_data(self):

        pass

    def test_get_test_data(self):

        pass

    def test__get_data(self):

        pass

    def test__get_randomized_position(self):

        pass

    def test__check_positions(self):

        pass

    def test_set_validation_range(self):

        pass

    def test_get_validation_range(self):

        pass

    def test_set_test_range(self):

        pass

    def test_get_test_range(self):

        pass

    def test_init_layers(self):

        pass

    def test__init_connections(self):

        pass

    def test__connect_layer(self):

        pass

    def test__build_output_conn(self):

        pass

    def test_randomize_network(self):

        pass

    def test_learn(self):

        pass

    def test_test(self):

        pass

    def test_calc_mse(self):

        self.assertAlmostEqual(10.0 / 2.0, self.net.calc_mse(100.0, 10))

    def test_process_sample(self):

        pass

    def test__feed_forward(self):

        #   simplify activations
        self.net.layers[0].set_activation_type('sigmoid')
        self.net.layers[1].set_activation_type('sigmoid')
        self.net.layers[2].set_activation_type('sigmoid')

        #   These values should be replaced
        self.net.layers[1].nodes[0].set_value(1000.0)
        self.net.layers[2].nodes[0].set_value(1000.0)

        self.assertEqual(1000.0, self.net.layers[1].nodes[0].get_value())
        self.assertEqual(1000.0, self.net.layers[2].nodes[0].get_value())

        self.net.layers[0].load_inputs([.2])

        self.net._feed_forward()

        self.assertEqual(.2, self.net.layers[0].nodes[0].get_value())
        self.assertEqual(
                sigmoid(.2) * 1.0,
                self.net.layers[1].nodes[0].get_value())

        self.assertEqual(
                sigmoid(sigmoid(.2) * 1.0) * .75,
                self.net.layers[2].nodes[0].get_value())

    def test__back_propagate(self):

        pass

    def test__update_error(self):

        pass

    def test__adjust_weights(self):
        """
        This function goes through layers starting with the top hidden layer
        and working its way down to the input layer.

        At each layer, the weights are adjusted based upon the errors.

        """
        halt_on_extremes = True

        for layer_no in range(len(self.net.layers) - 2, 0, -1):
            layer = self.net.layers[layer_no + 1]
            layer.adjust_weights(self.net._learnrate, halt_on_extremes)

    def test__zero_errors(self):

        for layer in self.net.layers[1:]:
            for node in layer.nodes:
                node.error = 1000

        self.net._zero_errors()

        for layer in self.net.layers[1:]:
            for node in layer.nodes:
                self.failIfEqual(1000, node.error)

    def test_calc_output_error(self):

        pass

    def test_calc_sample_error(self):

        pass

    def test__copy_levels(self):

        pass

    def test__parse_inputfile_layer(self):

        pass

    def test__parse_inputfile_node(self):

        pass

    def test__parse_inputfile_conn(self):

        pass

    def test__parse_inputfile_copy(self):

        pass

    def test__parse_node_id(self):

        pass

    def test_load(self):

        pass

    def test_output_values(self):

        pass

    def test__node_id(self):

        pass

    def test_save(self):

        pass


if __name__ == '__main__':
    unittest.main()
