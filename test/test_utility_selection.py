
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

from yaplf.utility.selection import *

class Test(unittest.TestCase):	
    """Unit tests for module yaplf.utility.selection"""
        
    def test_sequential_selector(self):
        """Test yaplf sequential_selector()."""
        s=sequential_selector([1,2,3])
        # TODO: add test checking type(s) is generator
        self.assertEqual(s.next(), 1)
        self.assertEqual(s.next(), 2)
        self.assertEqual(s.next(), 3)
        self.assertEqual(s.next(), 1)
        self.assertEqual(s.next(), 2)
    
    def test_random_selector(self):
        """Test yaplf random_selector()."""
        pass


if __name__ == "__main__":
    unittest.main()