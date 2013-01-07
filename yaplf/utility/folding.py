
r"""
Module handling sample folding in yaplf

Module :mod:`yaplf.utility.folding` contains all the classes handling sample
folding in yaplf.

AUTHORS:

- Dario Malchiodi (2011-01-21): initial version

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from numpy import random
from copy import copy

from yaplf.utility import cartesian_product, flatten


class SampleFolder(object):
    r"""
    :class:`SampleFolder` represents an abstract strategy in order to
    build the partition of a sample. More precisely, denoted by
    :math:`s=\{x_1, \dots, x_m\}` a sample of data, a *partition* of :math:`s`
    is a finite number of *folds* :math:`s_1, \dots, s_r`, where each
    fold correspond to a subset of :math:`s`, such that
    :math:`\cup_{i=1}^r s_i = s` and :math:`s_i \cap s_j = \emptyset`,
    that is mutually disjoint subsets of :math:`s` covering it.

    Folds in the partition of a sample are typically used in order to evaluate
    some probabilistical property on models inferred by the sample itself.

    :param sample: sample to be folded.

    :type sample: sequence of :class:`yaplf.data.Example` instances

    :param shuffle: flag triggering random shuffling of examples in
      :obj:`sample` before executing the folding procedure.

    :type shuffle: boolean

    Each subclass of :class:`SampleFolder` should implement a :func:`fold`
    method actually partitioning the sample specified during object
    instantiation.

    AUTHORS:

    - Dario Malchiodi (2011-01-21)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See :class:`SampleFolder` for full documentation.
        """

        self.sample = list(copy(sample))
        try:
            self.shuffle = kwargs['shuffle']
        except KeyError:
            self.shuffle = False

    @classmethod
    def partition(cls, sequence, num_folds):
        r"""
        Class method allowing the partition of a sequence into a given number
        of folds approximately having the same size.

        :param sequence: sequence to be partitioned.

        :type sequence: list or tuple typically containing
          :class:`yaplf.data.Example` instances

        :param num_folds: number of folds in the partition

        :type num_folds: integer

        :returns: the equi-sized elements' partition.

        :rtype: list of lists

        EXAMPLES

        In order to simplify examples, the following code snipplets will refer
        to the partition of sequences composed by numbers rather than instances
        of :class:`yaplf.data.Example`.

        When the number of elements is a multiple of the requred folds the
        sample is equipartitioned:

        >>> from yaplf.utility.folding import SampleFolder
        >>> SampleFolder.partition(range(10), 3)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        Otherwise the sample is partitioned in folds only approximately having
        the same size. In any case, the maximum difference between the folds'
        cardinalities is 1:

        >>> SampleFolder.partition(range(10), 7)
        [[0], [1], [2, 3], [4], [5, 6], [7], [8, 9]]

        AUTHORS:

        - Dario Malchiodi (2011-01-21)

        """

        num_elements = len(sequence)
        return [sequence[int(position * num_elements / num_folds): \
            int((position + 1) * num_elements / num_folds)]
            for position in range(num_folds)]

    def _check_and_shuffle(self):
        r"""
        Method to be overridden in subclasses in order to compute the sample
        folding.
        """

        if self.shuffle:
            random.shuffle(self.sample)


class ProportionalSampleFolder(SampleFolder):
    r"""
    :class:`ProportionalSampleFolder` allows for partitioning a sample in folds
    each having size approximately equal to specific proportion of the total
    length of the sample itself. These proportions are specified as a tuple
    :math:`p_1, \dots, p_m` such that :math:`\sum_{i=1}^m p_i = 1` and
    :math:`p_i \geq 0` for each :math:`i = 1, \dots, m`. For instance,
    :math:`p_2 = 0.5` means that the second fold will approximately contain
    half of the sample elements. The corresponding sequence of folds can be
    obtained invoking the :func:`fold` function.

    :param sample: sample to be folded.

    :type sample: sequence of :class:`yaplf.data.Example` instances

    :param proportions: sequence of proportions determining the size of folds.

    :type proportions: list or tuple of numeric values

    :param shuffle: flag triggering random shuffling of examples in ``sample``
      before executing the folding procedure.

    :type shuffle: boolean

    :raises: :exc:`ValueError` when the elements in :obj:`proportions` either
      contain negative numbers or do not sum to 1.


    EXAMPLES

    In order to simplify examples, the following code snipplets will refer
    to the partition of sequences composed by dummy examples described by
    a number:

    >>> from yaplf.data import Example
    >>> sample = [Example(p) for p in range(10)]
    >>> from yaplf.utility.folding import ProportionalSampleFolder
    >>> prop = ProportionalSampleFolder(sample, (.5, .3, .1, .1))
    >>> prop.fold(4) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1), Example(2), Example(3), Example(4)], [Example(5),
    Example(6), Example(7)], [Example(8)], [Example(9)]]

    The default behaviour doesn't shuffle data before partitioning, so that
    subsequent calls to :func:`fold` have a same result:

    >>> prop.fold(4) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1), Example(2), Example(3), Example(4)], [Example(5),
    Example(6), Example(7)], [Example(8)], [Example(9)]]

    This behaviour can be modified at the instantiation stage through the
    :obj:`shuffle` argument in the constructor:

    >>> prop = ProportionalSampleFolder(sample, (.5, .3, .1, .1), shuffle=True)
    >>> prop.fold(4) #doctest: +SKIP
    [[Example(0), Example(5), Example(6), Example(7), Example(4)], [Example(1),
    Example(8), Example(3)], [Example(9)], [Example(2)]]
    >>> prop.fold(4) #doctest: +SKIP
    [[Example(2), Example(7), Example(1), Example(0), Example(5)], [Example(3),
    Example(6), Example(9)], [Example(4)], [Example(8)]]

    A :exc:`ValueError` is thrown each time :obj:`proportions` do not describe
    a valid partition:

    >>> ProportionalSampleFolder(sample, (3, .1 ,.4))
    Traceback (most recent call last):
    ...
    ValueError: proportion list (3, 0.1, 0.4) does not sum to 1
    >>> ProportionalSampleFolder(sample, (-.5, .1 ,.4))
    Traceback (most recent call last):
    ...
    ValueError: non-positive elements in proportions list (-0.5, 0.1, 0.4)

    AUTHORS:

    - Dario Malchiodi (2011-01-21)

    """

    def __init__(self, sample, proportions, **kwargs):
        r"""
        See :class:`ProportionalSampleFolder` for full documentation.
        """

        SampleFolder.__init__(self, sample, **kwargs)

        for prop in proportions:
            if prop <= 0:
                raise ValueError('non-positive elements ' + \
                    'in proportions list ' + str(proportions))

        if sum(proportions) != 1:
            raise ValueError('proportion list ' + str(proportions) + \
                ' does not sum to 1')

        self.proportions = proportions

    def fold(self, num_folds):
        r"""
        Compute and return a partition of the sample supplied during
        instantiation.

        :param num_folds: number of folds in the partition.

        :type num_folds: integer

        :returns: the sample partition.

        :rtype: list of lists

        :raises: :exc:`ValueError` when :obj:`num_folds` does not equal the
          length of :obj:`proportions` or when at least one of the generated
          folds is the empty set.

        EXAMPLES

        In order to simplify examples, the following code snipplets will refer
        to the partition of sequences composed by dummy examples described by
        a number:

        >>> from yaplf.data import Example
        >>> sample = [Example(p) for p in range(10)]
        >>> from yaplf.utility.folding import ProportionalSampleFolder
        >>> prop = ProportionalSampleFolder(sample, (.5, .3, .1, .1))
        >>> prop.fold(4) #doctest: +NORMALIZE_WHITESPACE
        [[Example(0), Example(1), Example(2), Example(3), Example(4)],
        [Example(5), Example(6), Example(7)], [Example(8)], [Example(9)]]

        The method raises a :exc:`ValueError` when the requested number of
        folds is incompatible with the specified number of proportions:

        >>> prop.fold(3)
        Traceback (most recent call last):
        ...
        ValueError: required number of folds (3) different from number of
        proportions (4)

        A :exc:`ValueError` is also thrown when the computed partition contains
        at least an empty fold:

        >>> prop = ProportionalSampleFolder(sample, (.01, .5, .49))
        >>> prop.fold(3)
        Traceback (most recent call last):
        ...
        ValueError: empty subsample

        AUTHORS:

        - Dario Malchiodi (2011-01-21)

        """

        SampleFolder._check_and_shuffle(self)

        if num_folds != len(self.proportions):
            raise ValueError('required number of folds (' + str(num_folds) + \
                ') different from number of proportions (' + \
                str(len(self.proportions)) + ')')

        lengths = [int(prop * len(self.sample)) for prop in self.proportions]
        for len_ in lengths:
            if len_ == 0:
                raise ValueError('empty subsample')

        pos = [sum(lengths[:i]) for i in range(len(lengths))]
        return [self.sample[pos[i]:pos[i + 1]] \
            for i in range(len(pos) - 1)] + [self.sample[pos[len(pos) - 1]:]]


class UniformSampleFolder(SampleFolder):
    r"""
    :class:`UniformSampleFolder` allows for partitioning a sample in folds
    each having an approximately equal size.

    :param sample: sample to be folded.

    :type sample: sequence of :class:`yaplf.data.Example` instance

    :param shuffle: flag triggering random shuffling of examples in
      :obj:`sample` before executing the folding procedure.

    :type shuffle: boolean


    EXAMPLES

    In order to simplify examples, the following code snipplets will refer
    to the partition of sequences composed by dummy examples described by
    a number:

    >>> from yaplf.data import Example
    >>> sample = [Example(p) for p in range(10)]
    >>> from yaplf.utility.folding import UniformSampleFolder
    >>> unif = UniformSampleFolder(sample)
    >>> unif.fold(5) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1)], [Example(2), Example(3)], [Example(4),
    Example(5)], [Example(6), Example(7)], [Example(8), Example(9)]]
    >>> unif.fold(4) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1)], [Example(2), Example(3), Example(4)],
    [Example(5), Example(6)], [Example(7), Example(8), Example(9)]]

    The default behaviour doesn't shuffle data before partitioning, so that
    subsequent calls to :meth:`fold` have a same result:

    >>> unif.fold(4) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1)], [Example(2), Example(3), Example(4)],
    [Example(5), Example(6)], [Example(7), Example(8), Example(9)]]

    This behaviour can be modified at the instantiation stage through the
    ``shuffle`` argument in the constructor:

    >>> unif = UniformSampleFolder(sample, shuffle=True)
    >>> unif.fold(4) #doctest: +SKIP
    [[Example(9), Example(0)], [Example(4), Example(5), Example(1)],
    [Example(3), Example(7)], [Example(8), Example(6), Example(2)]]
    >>> unif.fold(4) #doctest: +SKIP
    [[Example(5), Example(4)], [Example(2), Example(6), Example(8)],
    [Example(9), Example(0)], [Example(3), Example(1), Example(7)]]

    AUTHORS:

    - Dario Malchiodi (2011-01-21)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See :class:`ProportionalSampleFolder` for full documentation.
        """

        SampleFolder.__init__(self, sample, **kwargs)

    def fold(self, num_folds):
        r"""
        Compute and return a partition of the sample supplied during
        instantiation.

        :param num_folds: number of folds in the partition.

        :type num_folds: integer

        :returns: the sample partition.

        :rtype: list of lists

        EXAMPLES

        In order to simplify examples, the following code snipplets will refer
        to the partition of sequences composed by dummy examples described by
        a number:

        >>> from yaplf.data import Example
        >>> sample = [Example(p) for p in range(10)]
        >>> from yaplf.utility.folding import UniformSampleFolder
        >>> unif = UniformSampleFolder(sample)
        >>> unif.fold(5) #doctest: +NORMALIZE_WHITESPACE
        [[Example(0), Example(1)], [Example(2), Example(3)], [Example(4),
        Example(5)], [Example(6), Example(7)], [Example(8), Example(9)]]
        >>> unif.fold(4) #doctest: +NORMALIZE_WHITESPACE
        [[Example(0), Example(1)], [Example(2), Example(3), Example(4)],
        [Example(5), Example(6)], [Example(7), Example(8), Example(9)]]

        The default behaviour doesn't shuffle data before partitioning, so that
        subsequent calls to :meth:`fold` have a same result:

        >>> unif.fold(4) #doctest: +NORMALIZE_WHITESPACE
        [[Example(0), Example(1)], [Example(2), Example(3), Example(4)],
        [Example(5), Example(6)], [Example(7), Example(8), Example(9)]]

        This behaviour can be modified at the instantiation stage through the
        ``shuffle`` argument in the constructor:

        AUTHORS:

        - Dario Malchiodi (2011-01-21)

        """

        SampleFolder._check_and_shuffle(self)
        return SampleFolder.partition(self.sample, num_folds)


class StratifiedSampleFolder(SampleFolder):
    r"""
    :class:`StratifiedSampleFolder` allows for partitioning a sample in folds
    each having an approximately equal size while guaranteeing that
    approximately a same number of items in each fold satisfies a given
    property.

    :param sample: sample to be folded.

    :type sample: sequence of :class:`yaplf.data.Example` instance

    :param shuffle: flag triggering random shuffling of examples in ``sample``
      before executing the folding procedure.

    :type shuffle: boolean

    :param stratification_data: values used in order to check properties to be
      satisfied by folds.

    :type stratification_data: lists


    EXAMPLES

    In order to show how properties are specified let's consider the following
    toy sample

    >>> from yaplf.data import Example
    >>> sample = 5*(Example(10),) + 7*(Example(11),) + 8*(Example(12),) +\
    ... 10*(Example(-10),) + 20*(Example(-11),) + 50*(Example(-12),)

    That is, sample will contain five copies of the item `10`, seven copies
    of `11` and so on. Suppose that each item is functionally associated to a
    *class* and to a *quality profile* as follows:

    - the most significative digit (including its sign) identifies the item's
      class;
    - the less significative digit identifies the item's quality.

    Therefore, items in the above mentioned sample can belong either to class
    `1` or to class `-1`, and their quality range from `0` to `2`. For
    instance, the first item (`10`) belongs to class `1` and has `0` as
    quality. So, it is easy to check that this sample contains 50 elements
    having quality 0 and belonging to class `1`, 80 elements belonging to
    class `-1` and so on. In spite of the fact that each item explicitly
    include its values for class and quality (this is not really what happens
    in the real world), let's also build two lists containing the class and the
    quality of each item:

    >>> classes = 5*(1,) + 7*(1,) + 8*(1,) + 10*(-1,) + 20*(-1,) + 50*(-1,)
    >>> qualities = 5*(0,) + 7*(1,) + 8*(2,) + 10*(0,) + 20*(1,) + 50*(2,)

    Given variables describing data, classes and quality it is possible to
    create an instance of :class:`StratifiedSampleFolder` and invoke its
    :meth:`fold` method:

    >>> from yaplf.utility.folding import StratifiedSampleFolder
    >>> strat = StratifiedSampleFolder(sample, classes, qualities)
    >>> partition = strat.fold(5)
    >>> partition #doctest: +NORMALIZE_WHITESPACE
    [[Example(10), Example(11), Example(12), Example(-10), Example(-10),
    Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
    Example(11), Example(12), Example(12), Example(-10), Example(-10),
    Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
    Example(11), Example(11), Example(12), Example(-10), Example(-10),
    Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
    Example(11), Example(12), Example(12), Example(-10), Example(-10),
    Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
    Example(11), Example(11), Example(12), Example(12), Example(-10),
    Example(-10), Example(-11), Example(-11), Example(-11), Example(-11),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
    Example(-12), Example(-12), Example(-12), Example(-12), Example(-12)]]

    In order to check whether or not the partitioning did preserve the
    percentage of elements belonging to the two classes in each fold (recall
    the whole sample contained 80% of items in class `-1`) it is possible to
    count the percentage of such elements within the various folds:

    >>> [100*len([item for item in fold if item.pattern<0])/len(fold) \
    ... for fold in partition]
    [84, 80, 80, 80, 76]

    Analogously, it is possible to check that the percentage of items having
    `2` as quality measure is approximately 58% (it is easy to obtain this
    percentage through inspection of the above sample definition):

    >>> [100*len([item for item in fold \
    ... if str(item.pattern)[-1]=='2'])/len(fold) for fold in partition]
    [57, 60, 55, 60, 57]

    Subsequent invocations of :meth:`fold` will produce the same output, unless
    specifying the named argument ``shuffle=True`` during instantiation of the
    sample folder:

    >>> sample = [Example(p) for p in range(10)]
    >>> classes = 5*(1,) + 5*(-1,)
    >>> strat = StratifiedSampleFolder(sample, classes)
    >>> strat.fold(2) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1), Example(5), Example(6)], [Example(2), Example(3),
    Example(4), Example(7), Example(8), Example(9)]]
    >>> strat.fold(2) #doctest: +NORMALIZE_WHITESPACE
    [[Example(0), Example(1), Example(5), Example(6)], [Example(2), Example(3),
    Example(4), Example(7), Example(8), Example(9)]]
    >>> strat = StratifiedSampleFolder(sample, classes, shuffle=True)
    >>> strat.fold(2) #doctest: +SKIP
    [[Example(7), Example(0), Example(9), Example(2)], [Example(1), Example(8),
    Example(4), Example(6), Example(3), Example(5)]]
    >>> strat.fold(2) #doctest: +SKIP
    [[Example(0), Example(4), Example(2), Example(5)], [Example(3), Example(7),
    Example(1), Example(6), Example(9), Example(8)]]

    The instantiation of :class:`StratifiedSampleFolder` will fail whenever
    any of the list/tuples passed as arguments describing the properties to
    stratify on will have different size w.r.t. the specified sample:

    >>> sample = [Example(p) for p in range(10)]
    >>> StratifiedSampleFolder(sample, range(20), range(8))
    Traceback (most recent call last):
    ...
    ValueError: each stratification data must have the same length of sample


    AUTHORS:

    - Dario Malchiodi (2011-01-21)

    """

    def __init__(self, sample, *args, **kwargs):
        r"""
        See :class:`ProportionalSampleFolder` for full documentation.
        """
        SampleFolder.__init__(self, sample, **kwargs)
        self.stratification_data = args
        for data in self.stratification_data:
            if len(data) != len(self.sample):
                raise ValueError('each stratification data ' \
                'must have the same length of sample')

    def fold(self, num_folds):
        r"""
        Compute and return a partition of the sample supplied during
        instantiation, stratifying on the properties summarized by the
        additional lists or tuples passed to the contstructor.

        :param num_folds: number of folds in the partition.

        :type num_folds: integer

        :returns: the sample partition.

        :rtype: list of lists

        EXAMPLES

        In order to show how properties are specified let's consider the
        following toy sample

        >>> from yaplf.data import Example
        >>> sample = 5*(Example(10),) + 7*(Example(11),) + 8*(Example(12),) +\
        ... 10*(Example(-10),) + 20*(Example(-11),) + 50*(Example(-12),)

        That is, sample will contain five copies of the item `10`, seven
        copies of `11` and so on. Suppose that each item is functionally
        associated to a *class* and to a *quality profile* as follows:

        - the most significative digit (including its sign) identifies the
          item's class;
        - the less significative digit identifies the item's quality.

        Therefore, items in the above mentioned sample can belong either to
        class `1` or to class `-1`, and their quality range from `0` to
        `2`. For instance, the first item (`10`) belongs to class `1` and
        has `0` as quality. So, it is easy to check that this sample contains
        50 elements having quality 0 and belonging to class `1`, 80 elements
        belonging to class `-1` and so on. In spite of the fact that each
        item explicitly include its values for class and quality (this is not
        really what happens in the real world), let's also build two lists
        containing the class and the quality of each item:

        >>> classes = 5*(1,) + 7*(1,) + 8*(1,) + 10*(-1,) + 20*(-1,) + 50*(-1,)
        >>> qualities = 5*(0,) + 7*(1,) + 8*(2,) + 10*(0,) + 20*(1,) + 50*(2,)

        Given variables describing data, classes and quality it is possible to
        create an instance of :class:`StratifiedSampleFolder` and invoke its
        :meth:`fold` method:

        >>> from yaplf.utility.folding import StratifiedSampleFolder
        >>> strat = StratifiedSampleFolder(sample, classes, qualities)
        >>> partition = strat.fold(5)
        >>> partition #doctest: +NORMALIZE_WHITESPACE
        [[Example(10), Example(11), Example(12), Example(-10), Example(-10),
        Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
        Example(11), Example(12), Example(12), Example(-10), Example(-10),
        Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
        Example(11), Example(11), Example(12), Example(-10), Example(-10),
        Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
        Example(11), Example(12), Example(12), Example(-10), Example(-10),
        Example(-11), Example(-11), Example(-11), Example(-11), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12)], [Example(10),
        Example(11), Example(11), Example(12), Example(12), Example(-10),
        Example(-10), Example(-11), Example(-11), Example(-11), Example(-11),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12),
        Example(-12), Example(-12), Example(-12), Example(-12), Example(-12)]]

        In order to check whether or not the partitioning did preserve the
        percentage of elements belonging to the two classes in each fold
        (recall the whole sample contained 80% of items in class `-1`) it is
        possible to count the percentage of such elements within the various
        folds:

        >>> [100*len([item for item in fold if item.pattern<0])/len(fold) \
        ... for fold in partition]
        [84, 80, 80, 80, 76]

        Analogously, it is possible to check that the percentage of items
        having `2` as quality measure is approximately 58% (it is easy to
        obtain this percentage through inspection of the above sample
        definition):

        >>> [100*len([item for item in fold \
        ... if str(item.pattern)[-1]=='2'])/len(fold) for fold in partition]
        [57, 60, 55, 60, 57]

        AUTHORS:

        - Dario Malchiodi (2011-01-21)

        """

        SampleFolder._check_and_shuffle(self)

        distinct_data = tuple([tuple(set(data)) \
            for data in self.stratification_data])

        distinct_combinations = cartesian_product(*distinct_data)

        groups_with_equal_combination = [[self.sample[pos] \
            for pos in range(len(self.sample)) \
            if tuple(data[pos] \
                for data in self.stratification_data) == combination] \
                for combination in distinct_combinations]

        partitioned_groups = [SampleFolder.partition(group, num_folds) \
            for group in groups_with_equal_combination]

        return [flatten([group[fold] for group in partitioned_groups]) \
            for fold in range(num_folds)]
