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