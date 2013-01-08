
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

from yaplf.utility.stopping import *

class Test(unittest.TestCase):	
    """Unit tests for utility module of yaplf."""


    def test_FixedIterationsStoppingCriterion(self):
        self.assertRaises(ValueError, FixedIterationsStoppingCriterion, 0)
        self.assertRaises(ValueError, FixedIterationsStoppingCriterion, 2.3)
        f=FixedIterationsStoppingCriterion(10)
        self.assertEquals([f.stop() for i in range(13)],
            [False, False, False, False, False, False, False, False, False,
            False, True, True, True])

    def test_TestErrorStoppingCriterion(self):
        from yaplf.data import LabeledExample
        from yaplf.algorithms import IdiotAlgorithm
        test_set = [LabeledExample((0, 1), (0,))]
        self.assertEqual(TestErrorStoppingCriterion(test_set),
            TestErrorStoppingCriterion([LabeledExample((0, 1), (0,))]))
        self.assertEqual(TestErrorStoppingCriterion(test_set, 0.01),
            TestErrorStoppingCriterion([LabeledExample((0, 1), (0,))],
            0.0100000000000000))
        self.assertRaises(ValueError, TestErrorStoppingCriterion, test_set, -4)
        #self.assertRaises(ValueError, TestErrorStoppingCriterion, test_set, 2.3)
        f = TestErrorStoppingCriterion(test_set, .01)
        xor_sample = [LabeledExample((1, 1), (1,)),
            LabeledExample((0, 0), (0,)), LabeledExample((1, 0), (0,))]
        idiot = IdiotAlgorithm(xor_sample, stopping_criterion = f, verbose = False)
        f.register_learning_algorithm(idiot)
        self.assertEqual([f.stop() for i in range(13)], [True, True, True,
            True, True, True, True, True, True, True, True, True, True])

    def test_TrainErrorStoppingCriterion(self):
        from yaplf.data import LabeledExample
        from yaplf.algorithms import IdiotAlgorithm
        self.assertEqual(TrainErrorStoppingCriterion(),
            TrainErrorStoppingCriterion())
        self.assertEqual(TrainErrorStoppingCriterion(0.01),
            TrainErrorStoppingCriterion(0.0100000000000000))
        self.assertRaises(ValueError, TrainErrorStoppingCriterion, -4)
        #self.assertRaises(ValueError, TrainErrorStoppingCriterion, 2.3)
        f = TrainErrorStoppingCriterion(.01)
        xor_sample = [LabeledExample((1, 1), (1,)),
            LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
            LabeledExample((1, 0), (0,))]
        idiot = IdiotAlgorithm(xor_sample, stopping_criterion = f, verbose = False)
        f.register_learning_algorithm(idiot)
        self.assertEqual([f.stop() for i in range(13)], [False, False, False,
           False, False, False, False, False, False, False, False, False,
           False])

if __name__ == "__main__":
    unittest.main()