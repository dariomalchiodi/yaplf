
r"""
Module handling data in yaplf

Module :mod:`yaplf.data` contains all the classes handling data in yaplf.

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

- Dario Malchiodi (2011-03-28): Added PrecomputedKernelSample

"""

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


class Example(object):
    r"""
    An example in its simplest form.

    The simplest form for an example consists of a sequence of numeric values
    called a *pattern*. Patterns are **always** expressed as lists or tuples,
    even when only one value is given (for instance ``[1]`` or ``(1,)`` -- the
    trailing comma in latter case is mandatory, otherwise the whole expression
    is equivalent to ``1``). Furthermore, a named argument :obj:`name` can be
    specified in order to denote the example through a mnemonic name.

    Examples have a simple string representation in which the pattern is
    enclosed in angle brackets, possibly followed by the example name in
    parentheses, and a :obj:`pattern` field containing the example's pattern.

    :param pattern: numeric values describing a pattern.

    :type pattern: sequence

    :param name: mnemonic name describing the example

    :type name: string, default: ``None``


    EXAMPLES:

    A simple example and its string representation:

    >>> from yaplf.data import Example
    >>> ex = Example((1, 2, 1, 4))
    >>> print ex
    <(1, 2, 1, 4)>

    Another example with indication of a symbolic name:

    >>> ex2 = Example((2, 3, 5, 7, 11), name = 'first five prime numbers')

    Full support for python introspection is provided:

    >>> ex.__repr__()
    'Example((1, 2, 1, 4))'
    >>> ex
    Example((1, 2, 1, 4))
    >>> ex2
    Example((2, 3, 5, 7, 11), name = 'first five prime numbers')
    >>> print ex2
    <(2, 3, 5, 7, 11)> (first five prime numbers)

    :class:`Example` objects can be accessed using the :obj:`pattern` and
    :obj:`name` fields:

    >>> ex.pattern
    (1, 2, 1, 4)
    >>> ex.name is None
    True
    >>> ex2.pattern
    (2, 3, 5, 7, 11)
    >>> ex2.name
    'first five prime numbers'

    :class:`Example` objects are compared only w.r.t. their patterns, that is
    regardless of their name, if any:

    >>> ex2 == Example((2, 3, 5, 7, 11))
    True

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, pattern, **kwargs):
        r"""
        See ``Example`` for full documentation.

        """

        self.pattern = pattern

        try:
            self.name = kwargs['name']
        except KeyError:
            self.name = None

    def __str__(self):
        result = '<' + str(self.pattern) + '>'
        if self.name is not None:
            result += ' (' + self.name + ')'
        return result

    def __repr__(self):
        result = 'Example(' + repr(self.pattern)
        if self.name is not None:
            result += ', name = \'' + self.name + '\''
        return result + ')'

    def __eq__(self, other):
        if type(self) == type(other):
            return self.pattern == other.pattern
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.pattern)

    def __nonzero__(self):
        return self.pattern


class LabeledExample(Example):
    r"""
    A labeled example.

    Labeled examples consist of a *pattern* and a *label*. Patterns are
    **always** described by a sequence (even when they consist of a sole
    value); On the other hand, labels can either be a single numeric value or
    a sequence of numeric values. It is further possible to assign a mnemonic
    name to the example.

    The string representation of labeled examples encloses their pattern and
    label in angle brackets, separating them through a comma and followed by
    the example name in parentheses, if provided. Moreover, the :obj:`pattern`,
    :obj:`label`, and :obj:`name` fields give access to the example's
    components.

    :param pattern: numeric value(s) describing a pattern.

    :type pattern: sequence

    :param label: numeric value(s) describing a label.

    :type label: number or sequence

    :param name: mnemonic name describing the example.

    :type name: string, default: ``None``

    EXAMPLES:

    A labeled example and its string representation:

    >>> from yaplf.data import LabeledExample
    >>> lex = LabeledExample((1, 2, 1, 4), 5)
    >>> print lex
    <(1, 2, 1, 4), 5>

    A more concrete example with indication of a symbolic name:

    >>> XORexample = LabeledExample((1, 1), 0,
    ... name = 'peculiar XOR example')
    >>> print XORexample
    <(1, 1), 0> (peculiar XOR example)

    Full support for python introspection is provided:

    >>> lex.__repr__()
    'LabeledExample((1, 2, 1, 4), 5)'
    >>> lex
    LabeledExample((1, 2, 1, 4), 5)
    >>> XORexample
    LabeledExample((1, 1), 0, name = 'peculiar XOR example')

    :class:`LabeledExample` objects can be accessed using the :obj:`pattern`,
    :obj:`label` and :obj:`name` fields:

    >>> lex.pattern
    (1, 2, 1, 4)
    >>> lex.label
    5
    >>> lex.name is None
    True
    >>> XORexample.pattern
    (1, 1)
    >>> XORexample.name
    'peculiar XOR example'

    :obj:`LabeledExample` objects are compared w.r.t. their patterns and
    labels, regardless of their name, if any:

     >>> XORexample != LabeledExample((1, 1), 0, name = 'foo')
     False

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, pattern, label, **kwargs):
        r"""
        See ``LabeledExample`` for full documentation.
        """

        Example.__init__(self, pattern, **kwargs)
        self.label = label

    def __str__(self):
        result = '<' + str(self.pattern) + ', ' + str(self.label) + '>'
        if self.name is not None:
            result += ' (' + self.name + ')'
        return result

    def __repr__(self):

        result = 'LabeledExample(' + repr(self.pattern) + ', ' + \
            repr(self.label)
        if self.name is not None:
            result += ', name = \'' + self.name + '\''
        return result + ')'

    def __eq__(self, other):
        if type(self) == type(other):
            return Example.__eq__(self, other) and self.label == other.label
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Example.__hash__(self), hash(self.label)))

    def __nonzero__(self):
        return Example.__nonzero__(self) and self.label


class ExampleDecorator(object):
    r"""
    base class for each decorator of the :class:`Example` class.

    A decorator [Gamma et al., 1995] is a software architecture enabling to
    dynamically extend a given class, allowing to specialize a system without
    generating an exponential number of subclasses. The role of this base
    class is simply that of encapsulating an :class`Example` object and letting
    it be accessible to the :obj:`example` field.

    :param example: example to be decorated.

    :type example: :class:`Example`

    EXAMPLES

    See the examples for :class:`AccuracyExample`.

    REFERENCES

    [Gamma et al., 1995] Erich Gamma, Richard Helm, Ralph Johnoson, John
    Vlissides, Design patterns: elements of reusable object-oriented software,
    Reading, Mass.: Addison-Wesley, 1995 (ISBN: 0201633612).

    """

    def __init__(self, example):
        r"""
        See ``ExampleDecorator`` for full documentation.

        """

        self.example = example

    def __eq__(self, other):
        if type(self) == type(other):
            return self.example == other.example
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.example)

    def __nonzero__(self):
        return self.example


class AccuracyExample(ExampleDecorator):
    r"""
    decorator for :class:`Example` adding to an example a quality value
    quantifying its accuracy, intended as measurement error.

    :param example: generic example

    :type example: :class:`Example`

    :param accuracy: example accuracy

    :type accuracy`` number

    EXAMPLES:

    An example coupled with its accuracy value, and its string representation:

    >>> from yaplf.data import LabeledExample, AccuracyExample
    >>> ae = AccuracyExample(LabeledExample((1, 2, 5), -1), 3)
    >>> print ae
    <(1, 2, 5), -1> (accuracy 3)

    :class:`AccuracyExample` objects can be accessed through their
    :obj:`example` and :obj:`accuracy` fields:

    >>> ae.example
    LabeledExample((1, 2, 5), -1)
    >>> ae.accuracy
    3

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, example, accuracy):
        r"""
        See ``AccuracyExample`` for full documentation.

        """

        ExampleDecorator.__init__(self, example)
        self.accuracy = accuracy

    def __str__(self):
        return str(self.example) + ' (accuracy ' + str(self.accuracy) + ')'

    def __repr__(self):
        return 'AccuracyExample(' + repr(self.example) + ', ' + \
            repr(self.accuracy) + ')'

    def __eq__(self, other):
        if type(self) == type(other):
            return ExampleDecorator.__eq__(self, other) and \
                self.accuracy == other.accuracy
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ExampleDecorator.__hash__(self), hash(self.accuracy)))

    def __nonzero__(self):
        return ExampleDecorator.__nonzero__(self) and self.accuracy

def get_precomputed_kernel_sample(labels, kernel):
    r"""
    Automatically generates a labeled sample on the basis of its labels
    and a corresponding precomputed kernel object.

    :param labels: labels for the examples composing the sample

    :type labels: iterable

    :param kernel: precomputed kernel

    :type kernel: :class:`yaplf.models.kernel.PrecomputedKernel`

    :returns: labeled sample

    :rtype: list of LabeledExample objects

    """

    if len(labels) != len(kernel.kernel_computations):
        raise ValueError('labels and kernel have incompatible dimensions')

    return [LabeledExample((i,), labels[i]) \
        for i in range(len(kernel.kernel_computations))]
