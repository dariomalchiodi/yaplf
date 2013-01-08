
#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@di.unimi.it>
#
# This file is part of yaplf.
# yaplf is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
# yaplf is distributed in the hope that it will be useful, but without any
# warranty; without even the implied warranty of merchantability or fitness
# for a particular purpose. See the GNU Lesser General Public License for
# more details.
# You should have received a copy of the GNU Lesser General Public License
# along with yaplf; if not, see <http://www.gnu.org/licenses/>.
#
#*****************************************************************************

import unittest

from yaplf.models import *
from yaplf.models.neural import *

class Test(unittest.TestCase):	
    """Unit tests for models module of yaplf."""
    
    def test_Model(self):
        """Test yaplf Model."""
        from yaplf.utility.error import MSE, MaxError
        from yaplf.data import LabeledExample
        sample = (LabeledExample( (-1,), (-1,) ), LabeledExample((0,), (0,)),
            LabeledExample((1,), (1,)))
        model = ConstantModel(0)
        self.assertEqual(model.test(sample, MSE()), 2.0 / 3)
        self.assertEqual(model.test(sample, MaxError()), 1)
        
    def test_ConstandModel(self):
        """Test yalpf ConstantModel."""
        from numpy import random
        model = ConstantModel(0)
        self.assertEqual(model.compute(1), 0)
        self.assertEqual(model.compute((1, 3)), 0)
        self.assertEqual(model.compute("string"), 0)
        self.assertEqual(model.compute(random.normal()), 0)
    
    def test_Perceptron(self):
        """Test yalpf Perceptron."""
        Perceptron(((1, 1),))
        Perceptron(((1, 1), (8, -4)))
        self.assertRaises(ValueError, Perceptron, ((1, 1), (8, -4, 9)))
        Perceptron(((1, 1),), threshold = (-1,))
        Perceptron(((1, 1), (8, -4)), threshold = (-1, 1))
        self.assertRaises(ValueError, Perceptron, ((1, 1), (8, -4)),
            threshold = (-1,))
        from yaplf.utility.activation import SigmoidActivationFunction
        from numpy import array
        Perceptron(((1, 1),), threshold = (-1,),
            activation = SigmoidActivationFunction())
        self.assertEqual(Perceptron(((1, 1),)).compute((0, 2)), 1)
        self.assertEqual(Perceptron(((1, 1),),
            activation=SigmoidActivationFunction()).compute((0, 2)),
            0.88079707797788231)
        self.assertEqual(Perceptron(((1, 1),), threshold=(1,),
            activation=SigmoidActivationFunction()).compute((0, 2)),
            0.7310585786300049)
        self.assertEqual(Perceptron(((1, -1), (-1, 1)),
            threshold = (-1, 1)).compute((0, 1)).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()