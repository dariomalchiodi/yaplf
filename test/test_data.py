
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

from yaplf.data import *

class Test(unittest.TestCase):	
    """Unit tests for data module of yaplf."""
    
    def test_Example(self):
        """Test yaplf Example."""
        ex = Example((1, 2, 1, 4))
        self.assertEqual(ex.__str__(), '<(1, 2, 1, 4)>')
        self.assertEqual(ex.__repr__(), 'Example((1, 2, 1, 4))')
        self.assertEqual(ex.pattern, (1, 2, 1, 4))
        ex2 = Example((1, 2, 1, 4))
        self.assertEqual(ex, ex2)
        ex3 = ((1, 2, -1, 4))
        self.assertNotEqual(ex, ex3)
        ex4 = Example((2, 3, 5, 7, 11), name = 'first five prime numbers')
        self.assertEqual(ex4.__str__(),
            '<(2, 3, 5, 7, 11)> (first five prime numbers)')
    
    def test_LabeledExample(self):
        """Test yaplf LabeledExample."""
        lex = LabeledExample((1,2,1,4), (5,))
        self.assertEqual(lex.__str__(), '<(1, 2, 1, 4), (5,)>')
        self.assertEqual(lex.__repr__(), 'LabeledExample((1, 2, 1, 4), (5,))')
        self.assertEqual(lex.pattern, (1, 2, 1, 4))
        self.assertEqual(lex.label, (5,))
        self.assertEqual(lex, LabeledExample((1,2,1,4), (5,)))
        self.assertNotEqual(lex, LabeledExample((1,-2,1,4), (5,)))
        self.assertNotEqual(lex, LabeledExample((1,2,1,4), (6,)))
        XORexample = LabeledExample((1, 1), 0, name = 'peculiar XOR example')
        self.assertEqual(XORexample.__str__(), \
            '<(1, 1), 0> (peculiar XOR example)')
        self.assertEqual(XORexample.name, 'peculiar XOR example')
    
    def test_AccuracyExample(self):
        """Test yaplf AccuracyExample."""
        ae = AccuracyExample(LabeledExample((1, 2, 5), (-1)), 3)
        self.assertEqual(ae.__str__(), '<(1, 2, 5), -1> (accuracy 3)')
        self.assertEqual(ae.accuracy, 3)
        self.assertEqual(ae,
            AccuracyExample(LabeledExample((1, 2, 5), (-1)), 3))
        self.assertNotEqual(ae, \
            AccuracyExample(LabeledExample((1, 2, 51), (-1)), 3))
        self.assertNotEqual(ae, \
            AccuracyExample(LabeledExample((1, 2, 5), (31)), 3))
        self.assertNotEqual(ae, \
            AccuracyExample(LabeledExample((1, 2, 5), (-1)), -3))


if __name__ == "__main__":
    unittest.main()