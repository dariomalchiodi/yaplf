
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