
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

from yaplf.utility import *

class Test(unittest.TestCase):	
    """Unit tests for utility module of yaplf."""
    
    def test_has_homogeneous_type(self):
        """Test yaplf has_homogeneous_type()."""
        self.assertEqual(has_homogeneous_type((1, 5, 3, 4, 2)), True)
        self.assertEqual(has_homogeneous_type(("one", "two", "three")), True)
        self.assertEqual(has_homogeneous_type((3.0, 1.0, 6.5)), True)
        self.assertEqual(has_homogeneous_type((1, 2, 3.0, 4)), False)
        self.assertEqual(has_homogeneous_type((1, "a", 4.0)), False)
        self.assertEqual(has_homogeneous_type((1, "a", 4)), False)
        
        self.assertEqual(has_homogeneous_type([1, 5, 3, 4, 2]), True)
        self.assertEqual(has_homogeneous_type(["one", "two", "three"]), True)
        self.assertEqual(has_homogeneous_type([3.0, 1.0, 6.5]), True)
        self.assertEqual(has_homogeneous_type([1, 2, 3.0, 4]), False)
        self.assertEqual(has_homogeneous_type([1, "a", 4.0]), False)
        self.assertEqual(has_homogeneous_type([1, "a", 4]), False)
        
    def test_is_iterable(self):
        self.assertEqual(is_iterable(8), False)
        self.assertEqual(is_iterable("string"), True)
        self.assertEqual(is_iterable((1,4,3)), True)
        self.assertEqual(is_iterable([4, "w"]), True)
        self.assertEqual(is_iterable(()), True)
        self.assertEqual(is_iterable(8.9), False)
    
    def test_chop(self):
        """Test yaplf chop()."""
        self.assertEqual(chop(3), 3)
        self.assertEqual(chop(10 ** -6), 10 ** -6)
        self.assertEqual(chop(10 ** -7), 0)
        self.assertEqual(chop(10 ** -7, tolerance = 0.001), 0)
        self.assertEqual(chop(2.999999999, right = 3), 3)
        self.assertRaises(ValueError, chop, 3, left = 5, right = 2)

    def test_split(self):
        """Test yaplf split()."""
        self.assertEqual(split(range(10), (.1, .5, .2, .2), random=False), [[0], [1, 2, 3, 4, 5], [6, 7], [8, 9]])
        self.assertRaises(ValueError, split, range(5), (.1, .5, .2, .2))
        self.assertRaises(ValueError, split, range(10), (.3, .5, .2, .2))
        self.assertRaises(ValueError, split, range(10), (.4, .7, -.1))

    def test_flatten(self):
        """Test yaplf flatten()."""
        nested_list = [1, [2, 3], [4, [5, 6]], 7, [8, 9, 0]]
        self.assertEqual(flatten(nested_list), [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        nested_tuple = ()
        for i in range(10):
           nested_tuple = (nested_tuple, i)
        self.assertEqual(flatten(nested_tuple), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    
    def test_cartesian_product(self):
        """Test yaplf cartesian_product()."""
        self.assertEqual(tuple(cartesian_product(*( (1,2), ('a','b') ))), ((1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')))

    def test_filter_arguments(self):
        """Test yaplf filter_arguments()."""
        d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        self.assertEqual(filter_arguments(d, ('b', 'c')), {'a': 1, 'd': 4})


if __name__ == "__main__":
    unittest.main()