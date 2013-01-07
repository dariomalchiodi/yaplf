
r"""
Module handling utility classes and functions in yaplf

Module :mod:`yaplf.utility` contains all the utility classes and functions in
yaplf.


AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

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


from copy import copy

from numpy import exp, random, array, mean


def kronecker_delta(i, j):
    r"""
    Implements the Kronecker delta function, whose value is ``1`` when the two
    values passed as arguments coincide and ``0`` otherwise.

    :param i: first function argument

    :type i: integer

    :param j: second function argument

    :type j: integer

    :returns: 1.0 when the arguments coincide, 0.0 otherwise

    :rtype: number

    EXAMPLES:

    >>> from yaplf.utility import kronecker_delta
    >>> kronecker_delta(7, 4)
    0.0
    >>> kronecker_delta(2, 2)
    1.0

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    return (1.0 if i == j else 0.0)


def to_column(vector):
    r"""
    Returns a column-like version of the supplied vector, i.e. a column
    vector.

    :param vector: vector to be converted

    :type vector: list, tuple or numpy array

    :returns: column vector version of :obj:`vector`.

    :rtype: numpy array


    EXAMPLES:

    The function returns the same result regardless of the particular vectorial
    form used in order to encode the input:

    >>> from yaplf.utility import to_column
    >>> from numpy import array
    >>> to_column(array((1, 2, 3))) #doctest: +NORMALIZE_WHITESPACE
    array([[1],
    [2],
    [3]])
    >>> to_column([1, 2, 3]) #doctest: +NORMALIZE_WHITESPACE
    array([[1],
    [2],
    [3]])
    >>> to_column((1, 2, 3)) #doctest: +NORMALIZE_WHITESPACE
    array([[1],
    [2],
    [3]])

    """

    return array([[elem] for elem in vector])


def has_homogeneous_type(values):
    r"""
    Check whether all the components of a list/tuple share the same type.

    :param values: elements to be checked.

    :type values: list or tuple

    :returns: ``True`` if :obj:`values` is made up of same-type elements,
      ``False`` otherwise.

    :rtype: boolean

    EXAMPLES:

    >>> from yaplf.utility import has_homogeneous_type
    >>> has_homogeneous_type((1, 5, 3, 4, 2))
    True
    >>> has_homogeneous_type(("one", "two", "three"))
    True
    >>> has_homogeneous_type((3.0, 1.0, 6.5))
    True
    >>> has_homogeneous_type((1, 2, 3.0, 4))
    False
    >>> has_homogeneous_type((1, "a", 4.0))
    False
    >>> has_homogeneous_type((1, "a", 4))
    False
    >>> has_homogeneous_type([1, 5, 3, 4, 2])
    True
    >>> has_homogeneous_type(["one", "two", "three"])
    True
    >>> has_homogeneous_type([3.0, 1.0, 6.5])
    True
    >>> has_homogeneous_type([1, 2, 3.0, 4])
    False
    >>> has_homogeneous_type([1, "a", 4.0])
    False
    >>> has_homogeneous_type([1, "a", 4])
    False

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    first_value = type(values[0])
    for value in values[1:]:
        if type(value) != first_value:
            return False
    return True


def is_iterable(candidate):
    r"""
    Check whether or not the supplied argument is iterable.

    :param candidate: whatever item to be checked for iterability.

    :type candidate: anything.

    :returns: ``True`` if :obj:`candidate` is iterable, ``False`` otherwise.

    :rtype: boolean

    EXAMPLES:

    >>> from yaplf.utility import is_iterable
    >>> is_iterable(8)
    False
    >>> is_iterable("string")
    True
    >>> is_iterable((1, 4, 3))
    True
    >>> is_iterable([4, "w"])
    True
    >>> is_iterable(())
    True
    >>> is_iterable(8.9)
    False

    """
    try:
        iter(candidate)
        return True
    except TypeError:
        return False


def chop(data, **kwargs):
    r"""
    Given a number and an interval, convert the latter into one interval's
    extremes if they are close enough.

    :param data: number to be checked.

    :type data: float

    :param left: left extreme of the interval.

    :type left: float, default: 0)

    :param right: right extreme of the interval

    :type right: float, default: ``None``, meaning :math:`+\infty`

    :param tolerance: minimum distance w.r.t. :obj:`left` or :obj:`right` in
      order not to change the value of :obj:`data`.

    :type tolerance: float, default: ``10**-6``

    :returns: :obj:`data` if it is farther than :obj:`tolerance` w.r.t.
      :obj:`left` and :obj:`right`, and the closer value otherwise.

    :rtype: float

    EXAMPLES:

    The default choice uses :math:`[0, + \infty)` as reference interval:

    >>> from yaplf.utility import chop
    >>> chop(3)
    3

    The order relation is intended in strong sense, that is the tolerance
    value corresponds to the first (or last) value which is not chopped:

    >>> chop(10. ** -6)
    9.9999999999999995e-07
    >>> chop(10 ** -7)
    0.0

    The reference interval and tolerance value can be custimzed through named
    arguments :obj:`left`, :obj:`right` and :obj:`tolerance`:

    >>> chop(10 ** -7, tolerance = 0.001)
    0.0
    >>> chop(2.999999999, right = 3)
    3

    """

    try:
        left = kwargs['left']
    except KeyError:
        left = 0.0

    try:
        right = kwargs['right']
    except KeyError:
        right = None

    if right is not None and left > right:
        raise ValueError('left extreme in chop should be lower or equal to \
        right extreme.')
    if data < left or (right is not None and data > right):
        raise ValueError('data to be chopped should belong to the specified \
        interval')

    try:
        tolerance = kwargs['tolerance']
    except KeyError:
        tolerance = 10 ** -6

    if data - left < tolerance:
        return left
    elif right is not None and right - data < tolerance:
        return right
    else:
        return data


def split(sample, proportions, **kwargs):
    r"""
    Split the sample given as argument in two or more parts, each
    (approximately) composed of a given fraction of the total elements,
    as specified in proportions. The latter parameter should be set to
    a list or tuple whose elements are all non-negative and sum to 1.

    Raises a :exc:`ValueError` when the specified proportions do not sum to 1
    or when any of them is not a positive number.

    :param sample: sample to be subdivided.

    :type sample: list/tuple

    :param proportions: sequence of fractions summing to 1.

    :type proportions: list/tuple of numbers

    :param random: flag triggering random shuffling for the original elements.

    :type random: boolean, default: True

    :returns: obtained subsamples

    :rtype: list

    EXAMPLES:

    The default behaviour randomly shuffles data before partitioning, so that
    in order to preserve data ordering it is necessary to explicitly use the
    :obj:`random` named argument:

    >>> from yaplf.utility import split
    >>> split(range(10), (.1, .5, .2, .2)) #doctest: +SKIP
    [[3], [5, 6, 1, 0, 2], [9, 8], [7, 4]]
    >>> split(range(10), (.1, .5, .2, .2), random = False)
    [[0], [1, 2, 3, 4, 5], [6, 7], [8, 9]]

    The function raises a :exc:`ValueError` when the number of data or one of
    the proportions is so small that at least one of the resulting subsamples
    has no elements:

    >>> split(range(5), (.1, .5, .2, .2))
    Traceback (most recent call last):
       ...
    ValueError: empty subsample

    The same error is raised when the specified proportions do not sum to 1 or
    contain any non-positive number:

    >>> split(range(10), (.3, .5, .2, .2))
    Traceback (most recent call last):
       ...
    ValueError: proportion list (0.300000000000000, 0.500000000000000,
    0.200000000000000, 0.200000000000000) does not sum to 1
    >>> split(range(10), (.4, .7, -.1))
    Traceback (most recent call last):
       ...
    ValueError: non-positive elements in proportions list
    (0.400000000000000, 0.700000000000000, -0.100000000000000)

    """

    for prop in proportions:
        if prop <= 0:
            raise ValueError('non-positive elements in proportions list ' + \
                str(proportions))

    if sum(proportions) != 1:
        raise ValueError('proportion list ' + str(proportions) + \
            ' does not sum to 1')

    sample_copy = copy(sample)

    try:
        rnd = kwargs['random']
    except KeyError:
        rnd = True

    if rnd:
        random.shuffle(sample_copy)

    lengths = [int(p * len(sample_copy)) for p in proportions]
    for len_ in lengths:
        if len_ == 0:
            raise ValueError('empty subsample')
    pos = [sum(lengths[:i]) for i in range(len(lengths))]
    return [sample_copy[pos[i]:pos[i + 1]] for i in range(len(pos) - 1)] + \
        [sample_copy[pos[len(pos) - 1]:]]


def flatten(list_, ltypes=(list, tuple)):
    r"""
    Flatten the lists or tuples contained in the specified argument, at
    arbitrary depth, into a unidimensional list.

    :param list_: nested lists or tuples to be flattened

    :type list_: list/tuple

    :param ltypes: types to be flattened

    :type ltypes: any of ``list``, ``tuple`` or ``(list, tuple)``, respectively
      triggering the flattening of lists, tuples or both, default:
      ``(list, tuple)``

    :returns: flattened list or tuple.

    :rtype: list or tuple

    EXAMPLES:

    In its default version the function flattens out elements of list/tuples,
    returning a tuple:

    >>> from yaplf.utility import flatten
    >>> nested_list = [1, [2, 3], [4, [5, 6]], 7, [8, 9, 0]]
    >>> flatten(nested_list)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    >>> nested_tuple = ()
    >>> for i in range(10):
    ...     nested_tuple = (nested_tuple, i)
    >>> flatten(nested_tuple)
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    The :obj:`ltypes` argument allow to flatten out only lists or tuples. For
    instance, the following instructions will leave unchanged the list [7, 8]:

    >>> mixed = (((1, 2), (3,)), ((4, 5, 6), [7, 8]), (9,))
    >>> flatten(mixed, tuple)
    (1, 2, 3, 4, 5, 6, [7, 8], 9)

    When a similar operation is performed using :obj:`list` as second argument,
    the argument is left unchanged as it consists of a list:

    >>> flatten(mixed, list)
    (((1, 2), (3,)), ((4, 5, 6), [7, 8]), (9,))

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    ltype = type(list_)
    list_ = list(list_)
    i = 0
    while i < len(list_):
        while isinstance(list_[i], ltypes):
            if not list_[i]:
                list_.pop(i)
                i -= 1
                break
            else:
                list_[i:i + 1] = list_[i]
        i += 1
    return ltype(list_)


def cartesian_product_generator(list_, *lists):
    r"""
    Compute a generator yielding the elements of the Cartesian product for
    the lists given as arguments.

    :param list_: first Cartesian product argument.

    :type list_: list

    :param lists: remaining Cartesian product arguments.

    :type lists: list

    :returns: object yielding the Cartesian product elements.

    :rtype: generator

    EXAMPLES:

    The value returned by the function is a generator whose invocations
    enumerate all Cartesian product elements, thus a quick way for obtaining
    the result as a whole is that of building a tuple from the generator:

    >>> from yaplf.utility import cartesian_product
    >>> tuple(cartesian_product(*((1, 2), ('a', 'b'))))
    ((1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'))

    The function is not limited to binary Cartesian products:

    >>> tuple(cartesian_product(*((1, 2, 3), ('a', 'b'), (-1, -2)))) \
    ... #doctest: +NORMALIZE_WHITESPACE
    ((1, 'a', -1), (1, 'a', -2), (1, 'b', -1), (1, 'b', -2), (2, 'a', -1),
    (2, 'a', -2), (2, 'b', -1), (2, 'b', -2), (3, 'a', -1), (3, 'a', -2),
    (3, 'b', -1), (3, 'b', -2))

    AUTHORS:

    - Dario Malchiodi (2011-01-22)

    """

    if not lists:
        for element in list_:
            yield (element,)
    else:
        for element in list_:
            for inner_element in \
                cartesian_product_generator(lists[0], *lists[1:]):
                yield (element,) + inner_element


def cartesian_product(list_, *lists):
    r"""
    Compute the Cartesian product of the lists given as arguments.

    :param list_: first Cartesian product argument.

    :type list_: list

    :param lists: remaining Cartesian product arguments.

    :type lists: list

    :returns: Cartesian product.

    :rtype: tuple

    EXAMPLES:

    The value returned by the function is a tuple whose elements are in turn
    tuples enumerating all Cartesian product elements:

    >>> from yaplf.utility import cartesian_product
    >>> cartesian_product((1, 2), ('a', 'b'))
    ((1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'))

    The function is not limited to binary Cartesian products:

    >>> cartesian_product((1, 2, 3), ('a', 'b'), (-1, -2)) \
    ... #doctest: +NORMALIZE_WHITESPACE
    ((1, 'a', -1), (1, 'a', -2), (1, 'b', -1), (1, 'b', -2), (2, 'a', -1),
    (2, 'a', -2), (2, 'b', -1), (2, 'b', -2), (3, 'a', -1), (3, 'a', -2),
    (3, 'b', -1), (3, 'b', -2))


    AUTHORS:

    - Dario Malchiodi (2011-01-22)

    """

    return tuple(cartesian_product_generator(list_, *lists))


def filter_arguments(original_args, keys_to_be_filtered):
    r"""
    Filter out a set of keys from a dictionary. Used in order to filter
    named arguments passed to a method before subsequently passing them to
    another method. The original dictionary is not altered.

    :param original_args: original entries to be filtered out.

    :type original_args: dictionary

    :param keys_to_be_filtered: keys to be filtered out.

    :type keys_to_be_filtered: list or tuple

    :returns: filtered entries.

    :rtype: dictionary

    EXAMPLE:

    The filtered dictionary is returned by the function. The original one is
    left unchanged:

    >>> from yaplf.utility import filter_arguments
    >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> filter_arguments(d, ('b', 'c'))
    {'a': 1, 'd': 4}
    >>> d
    {'a': 1, 'c': 3, 'b': 2, 'd': 4}

    The function doesn't complain when inexisting keys are filtered out:

    >>> filter_arguments(d, ('x',))
    {'a': 1, 'c': 3, 'b': 2, 'd': 4}

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    filtered_args = copy(original_args)
    for key in keys_to_be_filtered:
        try:
            del filtered_args[key]
        except KeyError:
            pass  # don't complain about filtering inexisting keys
    return filtered_args


class Observable(object):
    r"""
    Base class for observable objects, intended as in pattern
    observer-observable [Gamma et al., 1995]. All subclasses should invoke
    :meth:`notify_observers` each time their status undergoes a change.
    Likewise, all observers should register to their observable invoking
    :meth:`add_observer` once and should implement a method update which is
    called when the observable changes.

    EXAMPLES:

    See the examples for specific subclasses such as
    :class:`GradientPerceptronAlgorithm` in module
    :mod:`yaplf.algorithms.neural`.

    REFERENCES:

    [Gamma et al., 1995] Erich Gamma, Richard Helm, Ralph Johnoson, John
    Vlissides, Design patterns: elements of reusable object-oriented software,
    Reading, Mass.: Addison-Wesley, 1995 (ISBN: 0201633612).

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self):
        r"""
        See :class:`Observable` for full documentation.

        """

        self.observers = []

    def add_observer(self, observer):
        r"""
        Add an observer to the list of objects observing the state. The
        invocation is ignored if the observer is already containted in the
        observers' list.

        :param observer: observer to be added to the observers' list

        :type observer: :class:`Observer`

        :rtype: void

        EXAMPLES:

        See the examples for specific subclasses such as
        :class:`GradientPerceptronAlgorithm` in module
        :mod:`yaplf.algorithms.neural`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer):
        r"""
        Remove an observer from the list of objects observing the state.
        The invocation is ignored if the observer is not contained in the
        observers' list.

        :param observer: observer to be removed from the observers' list.

        :type observer: :class:`Observer`

        :rtype: void

        EXAMPLES:

        See the examples for specific subclasses such as
        :class:`GradientPerceptronAlgorithm` in module
        :mod:`yaplf.algorithms.neural`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self):
        r"""
        Notify all the observers a change in the status through invocation
        of their update method. Each observer will then query the observable
        in order to get renewed information.

        :rtype: void


        EXAMPLES:

        This method is automatically invoked within the observer/observable
        interaction.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        for observer in self.observers:
            observer.update(self)


class Observer(object):
    r"""
    Base class for Observer classes, intended as in pattern
    observer-observable [Gamma et al., 1995]. All subclasses should provide
    implementation of a method :meth:`update` to be automatically called when
    the observable status is changed, and a :meth:`sync_state` method to be
    used in order to collect a trajectory showing how the observable state
    evolves through time.

    :param observable: object to be observed

    :type observable: :class:`Observable`


    EXAPMLES:

    See the examples for specific subclasses such as
    :class:`yaplf.graph.trajectory.PerceptronWeightTrajectory`.

    REFERENCES:

    [Gamma et al., 1995] Erich Gamma, Richard Helm, Ralph Johnoson, John
    Vlissides, Design patterns: elements of reusable object-oriented software,
    Reading, Mass.: Addison-Wesley, 1995 (ISBN: 0201633612).

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, observable):
        r"""
        See :class:`Observer` for full documentation.

        """

        observable.add_observer(self)
        self.observable = observable

    def update(self, observable):
        r"""
        Method called when the observable changes its status.

        :param observable: observable whose status has changed.

        :type observable: :class:`Observable`

        :rtype: void

        EXAMPLES:

        See the examples for specific subclasses such as
        :class:`yaplf.graph.trajectory.PerceptronWeightTrajectory`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        pass
