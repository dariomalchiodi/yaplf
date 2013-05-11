
r"""
Module handling error models in yaplf

Module :mod:yaplf.utility.error contains error models in yaplf.

AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  :mod:yaplf.utility, containing base class, mean-square and maximum error
  models.

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


from numpy import array


class ErrorModel(object):
    r"""
    Base class for each error model, intended as a measure of how a model
    correctly approximates a given data set. This measure is obtained through
    invocation of the the function ``compute``, which should be implemented
    in each subclass.

    EXAMPLES:

    See the examples for specific subclasses, such as :class:`MSE` in this
    module.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def compute(self, sample, model, *args, **kwargs):
        r"""
        Return a measure of how model correctly approximates sample.
        When invoked in the base class raises NotImplementedError.

        :param self: object on which the function is invoked

        :type self: :class:`ErrorModel`

        :param sample: sample to be checked.

        :type sample: list or tuple of :class:`yaplf.data.Example`

        :param model: model to be checked.

        :type model: :class:`yaplf.models.Model`

        :param verbose: flag triggering verbose output.

        :type verbose: boolean, default: ``False``

        :returns: approximation error of the given model on the specified
          sample.

        :rtype: float

        EXAMPLES:

        See the examples for specific subclasses, such as :class:`MSE` in this
        module.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError('the ErrorModel class you are using' + \
            'does not implement compute method')


class MSE(ErrorModel):
    r"""
    Error model computing the mean square error. Given a sequence of values
    `y_1, \dots, y_n` and their approximations :math:`t_1, \dots, t_n`, this
    model computes the approximation capability in terms of the mean square
    error as :math:`\frac{1}{n} \sum_{i = 1}^n (y_i - t_i)^2`.

    EXAMPLES:

    Consider the following sample whose labels are respectively ``-1``, ``0``
    and ``1``, coupled with a dummy model outputting ``0`` regardless of the
    fed input. Testing this model on the above sample will score a unit square
    error on the first and last example, and a null error on the remaining one,
    so that the mean squared error will be :math:`\frac{2}{3}`:

    >>> from yaplf.utility.error import MSE
    >>> from yaplf.models import ConstantModel # outputs a constant value
    >>> from yaplf.data import LabeledExample
    >>> model = ConstantModel(0)
    >>> sample = (LabeledExample((-1,), (-1,)),
    ... LabeledExample((0,), (0,)), LabeledExample((1,), (1,)))
    >>> error_model = MSE()
    >>> error_model.compute(sample, model)
    0.66666666666666663

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __repr__(self):
        return 'MSE()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash("MSE")

    def __nonzero__(self):
        return True

    def compute(self, sample, model, *args, **kwargs):
        r"""
        Return the mean of squared error obtained when approximating
        sample using model.

        :param sample: sample to be checked.

        :type sample: list or tuple of :class:`yaplf.data.Example`

        :param model: model to be checked.

        :type model: :class:`yaplf.models.Model`

        :param verbose: flag triggering verbose output.

        :type verbose: boolean, default: ``False``

        :returns: mean squared approximation error of the given model on the
          specified sample.

        :rtype float:

        EXAMPLES:

        Consider the following sample whose labels are respectively ``-1``,
        ``0`` and ``1``, coupled with a dummy model outputting 0 regardless of
        the fed input. Testing this model on the above sample will score a unit
        square error on the first and last example, and a null error on the
        remaining one, so that the mean squared error will be
        :math:`\frac{2}{3}`:

        >>> from yaplf.utility.error import MSE
        >>> from yaplf.models import ConstantModel
        >>> from yaplf.data import LabeledExample
        >>> model = ConstantModel(0)
        >>> sample = (LabeledExample((-1,), (-1,)),
        ... LabeledExample((0,), (0,)), LabeledExample((1,), (1,)))
        >>> error_model = MSE()
        >>> error_model.compute(sample, model)
        0.66666666666666663

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False
        error = sum(array([(array(model.compute(e.pattern)) - \
            array(e.label)) ** 2 for e in sample]).ravel()) * 1. / len(sample)
        if verbose:
            for elem in sample:
                print str(elem.pattern) + ' mapped to ' + \
                    str(model.compute(elem.pattern)) + ', label is ' + \
                    str(elem.label) + ', error ' + \
                    str((array(model.compute(elem.pattern)) - \
                    array(elem.label)) ** 2)
            print 'MSE ' + str(error)
        return error


class MaxError(ErrorModel):
    r"""
    Error model computing the maximum squared error.


    Error model computing the maximum squared error. Given a sequence of
    values :math:`y_1, \dots, y_n` and their approximations
    :math:`t_1, \dots, t_n`, this model computes the approximation capability
    in terms of the maximum square error as
    :math:`\max_{i = 1, \dots, n} (y_i - t_i)^2`.

    EXAMPLES:

    Consider the following sample whose labels are respectively ``-1``, ``0``
    and ``1``, coupled with a dummy model outputting ``0`` regardless of the
    fed input. Testing this model on the above sample will score a unit square
    error on the first and last example, and a null error on the remaining one,
    so that the mean squared error will be ``1``:

    >>> from yaplf.utility.error import MaxError
    >>> from yaplf.models import ConstantModel # Outputs a constant value
    >>> from yaplf.data import LabeledExample
    >>> model = ConstantModel(0)
    >>> sample = (LabeledExample((-1,), (-1,)),
    ... LabeledExample((0,), (0,)), LabeledExample((1,), (1,)))
    >>> error_model = MaxError()
    >>> error_model.compute(sample, model)
    1

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __repr__(self):
        return 'MaxError()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash("MaxError")

    def __nonzero__(self):
        return True

    def compute(self, sample, model, *args, **kwargs):
        r"""
        Return the maximum squared error obtained when approximating sample
        using model.

        :param sample: sample to be checked.

        :type sample: list or tuple of :class:`yaplf.data.Example`

        :param model: model to be checked.

        :type model: :class:`yaplf.models.Model`

        :param verbose: flag triggering verbose output.

        :type verbose: boolean, default: ``False``

        :returns: maximum error of the given model on the specified sample.

        :rtype float:

        EXAMPLES:

        Consider the following sample whose labels are respectively ``-1``,
        ``0`` and ``1``, coupled with a dummy model outputting ``0`` regardless
        of the fed input. Testing this model on the above sample will score a
        unit square error on the first and last example, and a null error on
        the remaining one, so that the mean squared error will be :math:`1`:

        >>> from yaplf.utility.error import MaxError
        >>> from yaplf.models import ConstantModel
        >>> from yaplf.data import LabeledExample
        >>> model = ConstantModel(0)
        >>> sample = (LabeledExample((-1,), (-1,)),
        ... LabeledExample((0,), (0,)), LabeledExample((1,), (1,)))
        >>> error_model = MaxError()
        >>> error_model.compute(sample, model)
        1

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

        if verbose:
            for elem in sample:
                print str(elem.pattern) + ' mapped to ' + \
                    str(model.compute(elem.pattern)) + ', label is ' + \
                    str(elem.label) + ', error ' + \
                    str(sum((array(model.compute(elem.pattern)) - \
                    array(elem.label)) ** 2))
        max_error = max([sum((array(model.compute(elem.pattern)) - \
            array(elem.label)) ** 2) for elem in sample])
        if verbose:
            print 'Maximum error: ' + str(max_error)
        return max_error


class Norm(object):
    def compute(self, x1, x2):
        pass


class PNorm(Norm):
    def __init__(self, degree):
        Norm.__init__(self)
        if degree <= 0 or int(degree) != degree:
            raise ValueError('the specified degree parameter is not \
                a positive integer')
        self.degree = degree

    def compute(self, x1, x2):
        return sum([(x[0]- x[1]) ** self.degree for x in array((array(x1).flatten(), array(x2).flatten())).T ]) ** (1.0/self.degree)


class InfinityNorm(Norm):
    def __init__(self):
        Norm.__init__(self)

    def compute(self, x1, x2):
        return max([(x[0]- x[1]) ** self.degree for x in array((array(x1).flatten(), array(x2).flatten())).T ]) ** (1.0/self.degree)

