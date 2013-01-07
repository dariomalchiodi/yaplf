

r"""
Module handling example selection in yaplf

Module :mod:`yaplf.utility.selection` contains examples' selectors in yaplf.


AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  :mod:`yaplf.utility`, containing base sequential and random selectors.

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
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


from numpy import random


def sequential_selector(sample):
    r"""
    Return a generator subsequently yielding elements in ``sample`` in
    positional order, starting over when every elements has been selected.

    :param sample: elements to be enumerated cyclically.

    :type sample: list or tuple of :class:`yaplf.data.Example`

    :returns: selectory cyclically yelding the elements of :obj:`sample`.

    :rtype: generator

    EXAMPLES:

    >>> from yaplf.utility.selection import sequential_selector
    >>> sequential_selector([1, 2, 3]) #doctest: +ELLIPSIS
    <generator object sequential_selector at ...>
    >>> s = sequential_selector([1, 2, 3])
    >>> s.next()
    1
    >>> s.next()
    2
    >>> s.next()
    3
    >>> s.next()
    1
    >>> s.next()
    2

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    i = 0
    while(True):
        yield sample[i % len(sample)]
        i = i + 1


def random_selector(sample):
    r"""
    Return a generator yelding elements in sample uniformly chosen
    at random, with replacement.

    :param sample: elements to be selected at random.

    :type sample: list or tuple of :class:`yaplf.data.Example`

    :returns: selectory randomly yelding the elements of :obj:`sample`.

    :rtype: generator

    EXAMPLES:

    The generator returned by this function selects each time one element in
    :obj`sample` uniformly at random. Sampling is done with replacement.

    >>> from yaplf.utility.selection import random_selector
    >>> r = random_selector((1, 2, 3))
    >>> r.next() #doctest: +SKIP
    1
    >>> r.next() #doctest: +SKIP
    2
    >>> r.next() #doctest: +SKIP
    2
    >>> r.next() #doctest: +SKIP
    2

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    random.seed()
    while(True):
        yield sample[random.randint(0, len(sample))]
