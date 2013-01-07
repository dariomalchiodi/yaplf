
import unittest

from yaplf.utility.error import *

class Test(unittest.TestCase):	
    """Unit tests for module of yaplf.utility.error."""


    def test_MSE(self):
        """Test yaplf MSE()."""
        from yaplf.models import ConstantModel # Dummy model outputting a constant value
        from yaplf.data import LabeledExample
        model = ConstantModel(0)
        sample = ( LabeledExample( (-1,), (-1,) ), LabeledExample( (0,), (0,) ), LabeledExample( (1,), (1,) ) )
        error_model = MSE()
        self.assertEqual(error_model.compute(sample, model), 2.0/3)
    
    def test_MaxError(self):
        """Test yaplf MaxError()."""
        from yaplf.models import ConstantModel # Dummy model outputting a constant value
        from yaplf.data import LabeledExample
        model = ConstantModel(0)
        sample = ( LabeledExample( (-1,), (-1,) ), LabeledExample( (0,), (0,) ), LabeledExample( (1,), (1,) ) )
        error_model = MaxError()
        self.assertEqual(error_model.compute(sample, model), 1)


if __name__ == "__main__":
    unittest.main()