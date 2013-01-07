
import unittest

from yaplf.utility.folding import *

class Test(unittest.TestCase):	
    """Unit tests for module yaplf.utility.folding"""

    def test_SampleFolder_partition(self):
        """Test yaplf SampleFolder.partition()."""

        self.assertEqual(SampleFolder.partition(range(10), 2), \
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual(SampleFolder.partition(range(10), 3), \
            [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]])
        self.assertEqual(SampleFolder.partition(range(10), 7), \
            [[0], [1], [2, 3], [4], [5, 6], [7], [8, 9]])

    def test_ProportionalSampleFolder_fold(self):
        """Test yaplf ProportionalSampleFolder.fold()."""

        from yaplf.data import Example
        sample = [Example(p) for p in range(10)]
        prop = ProportionalSampleFolder(sample, (.5, .3, .1, .1))
        self.assertEqual(prop.fold(4), [[Example(0), Example(1), \
            Example(2), Example(3), Example(4)], [Example(5), \
            Example(6), Example(7)], [Example(8)], [Example(9)]])

    def test_ProportionalSampleFolder_fold_exceptions(self):
        """Test that ValueError is thrown when the number
        of required folds is different from the number of
        specified proportions, or when the latter do not
        sum to 1"""

        from yaplf.data import Example
        sample = [Example(p) for p in range(10)]
        prop = ProportionalSampleFolder(sample, (.5, .3, .1, .1))
        self.assertRaises(ValueError, prop.fold, 3)

        self.assertRaises(ValueError, ProportionalSampleFolder, \
        *(sample, (.5, .1, 3.)))


if __name__ == "__main__":
    unittest.main()
