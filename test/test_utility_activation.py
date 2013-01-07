
import unittest

from yaplf.utility.activation import *

class Test(unittest.TestCase):	
    """Unit tests for utility module of yaplf."""
        
    def test_sigmoid(self):
        """Test yaplf sigmoid()."""
        f = SigmoidActivationFunction()
        # TODO: add test checking type(f) is function
        self.assertEqual(f.compute(0), 0.5)
        self.assertEqual(f.compute(1), 0.7310585786300049)
        f = SigmoidActivationFunction(10)
        self.assertEqual(f.compute(0), 0.5)
        self.assertEqual(f.compute(1), 0.99995460213129761)
        self.assertRaises(ValueError, SigmoidActivationFunction, 0)
    
    def test_heaviside(self):
        """Test yaplf heaviside()."""
        from numpy import random
        f = HeavisideActivationFunction()
        self.assertEqual(f.compute(random.normal()**2), 1)
        self.assertEqual(f.compute(-1*random.normal()**2), 0)
        self.assertEqual(f.compute(0), 1)

if __name__ == "__main__":
    unittest.main()