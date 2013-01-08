
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

from distutils.core import setup

setup(name = 'yaplf',
    version = '0.7',
    description = 'Yet Another Python Learning Framework',
    author = 'Dario Malchiodi',
    author_email = 'malchiodi@dsi.unimi.it',
    url = 'http://homes.dsi.unimi.it/~malchiod/software/yaplf',
    packages = ['yaplf', 'yaplf.models', 'yaplf.algorithms'],
    requires = ['numpy (>=1.3)', 'cvxopt (>=0.9)', 'PyML (>=0.7)'],
    provides = ['yaplf'],
    license = 'GNU Lesser General Public License')