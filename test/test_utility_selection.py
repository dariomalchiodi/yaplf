
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