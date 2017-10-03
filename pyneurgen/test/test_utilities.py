import unittest

from pyneurgen.utilities import rand_weight, base10tobase2, base2tobase10


class TestUtilities(unittest.TestCase):


    def test_test_rand_weight(self):

        constraint = 1.0
        sample = [rand_weight() for i in range(1000)]
        self.assertGreaterEqual(constraint, max(sample))
        self.assertLessEqual(-constraint, min(sample))

        constraint = 0.5
        sample = [rand_weight(constraint) for i in range(1000)]
        self.assertGreaterEqual(constraint, max(sample))
        self.assertLessEqual(-constraint, min(sample))

    def test_test_base10tobase2(self):

        base10 = "3"
        self.assertEqual("11", base10tobase2(base10))

        base10 = "34"
        self.assertEqual("100010", base10tobase2(base10))

        base10 = "34"
        zfill = 8
        self.assertEqual("00100010", base10tobase2(base10, zfill))

        base10 = "34"
        zfill = 5
        self.assertRaises(ValueError, base10tobase2, base10, zfill)

        base10 = "34"
        zfill = 5
        self.assertRaises(ValueError, base10tobase2, base10, zfill)

        base10 = "0"

        self.assertEqual("0", base10tobase2(base10))

        base10 = "-34"
        self.assertEqual("-100010", base10tobase2(base10))


    def test_test_base2tobase10(self):

        base2 = "100010"
        self.assertEqual(34, base2tobase10(base2))

        base2 = "1"
        self.assertEqual(1, base2tobase10(base2))

        base2 = "10"
        self.assertEqual(2, base2tobase10(base2))


if __name__ == '__main__':
    unittest.main()
