import unittest
import numpy as np
from pyoculus.irrationals import expandcf, fromcf

# Define a list of test cases, each case is a tuple of (FRAC, CI)
# where FRAC is a tuple of (numerator, denominator) of a fraction
# and CI is a list of coefficients of the continued fraction expansion
test_cases = [
    ((5, 7), [0, 1, 2, 2]),
# Content is available under The OEIS End-User License Agreement: http://oeis.org/LICENSE
    # OEIS A010124: Continued fraction for sqrt(19)
    ((1421, 326), [4, 2, 1, 3, 1, 2, 8]),
    # OEIS A001203: Continued fraction for pi
    ((833719, 265381), [3, 7, 15, 1, 292, 1, 1, 1, 2]),
]

class TestContinuedFractionFunctions(unittest.TestCase):
    def test_expandcf(self):
        for frac, ci in test_cases:
            with self.subTest(ci=ci, frac=frac):
                result = expandcf(frac[0]/frac[1], len(ci))
                expected = ci
                np.testing.assert_array_equal(result, expected, f"Failed to correctly expand frac={frac} into ci={ci}.")

    def test_fromcf(self):
        for frac, ci in test_cases:
            with self.subTest(ci=ci, frac=frac):
                result = fromcf(ci)
                expected = frac
                self.assertEqual(result, expected, f"Failed to correctly convert ci={ci} back to fraction frac={frac}.")

if __name__ == '__main__':
    unittest.main()