
import unittest

from yaplf.utility.validation import *

class Test(unittest.TestCase):	
    """Unit tests for module yaplf.utility.validation"""

    def test_train_and_test(self):
        """Test yaplf train_and_test()."""
        from yaplf.data import LabeledExample
        from yaplf.algorithms import IdiotAlgorithm
        train_sample = (LabeledExample((0, 0,), (0,)), 
            LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)), \
            LabeledExample((1, 1), (1,)))
        test_sample = (LabeledExample((1, 0), (1,)),
            LabeledExample((1, 1), (1,)))
        self.assertEqual(train_and_test(IdiotAlgorithm, 
            train_sample, test_sample, {'verbose': False},
            run_parameters={'num_steps': 500}), 1.0)

    def test_cross_validation_step(self):
        """Test yaplf cross_validation_step()."""
        from yaplf.data import LabeledExample
        from yaplf.algorithms import IdiotAlgorithm
        
        split_sample=((LabeledExample((0, 0), (0,)),
            LabeledExample((0, 1), (1,))), (LabeledExample((1, 0), (1,)),
            LabeledExample((1, 1), (1,))))
        parameters={'threshold': True}
        self.assertEqual(cross_validation_step(IdiotAlgorithm,
            parameters, split_sample, fixed_parameters = {'verbose': False},
            run_parameters={'num_steps': 500}), 0.75)

    def test_cross_validation(self):
        from yaplf.data import LabeledExample
        from yaplf.algorithms import IdiotAlgorithm
        from yaplf.models import ConstantModel
        from yaplf.utility.error import MSE

        sample = (LabeledExample((0, 0), (0,)), LabeledExample((1, 1), (1,)),
            LabeledExample((2, 2), (1,)), LabeledExample((3, 3), (1,)),
            LabeledExample((4, 4), (0,)), LabeledExample((5, 5), (1,)),
            LabeledExample((6, 6), (1,)), LabeledExample((7, 7), (1,)),
            LabeledExample((8, 8), (0,)), LabeledExample((9, 9), (1,)))
        parameter_description = {'c': (1, 10), 'sigma': (.1, .01)}
        self.assertEquals(cross_validation(IdiotAlgorithm, sample,
            parameter_description,
            fixed_parameters = {'verbose': False},
            run_parameters = {'num_iters': 100}, error_model = MSE(),
            num_folds = 3, verbose = False), ConstantModel(0))

if __name__ == "__main__":
    unittest.main()
