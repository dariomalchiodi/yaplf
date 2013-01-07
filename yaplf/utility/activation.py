
r"""
Module handling activation functions in yaplf

Module :mod:`yaplf.utility.activation` contains activation functions in yaplf.

AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  :mod:`yaplf.utility`, containing base class, sigmoidal, hyperbolic tangent,
  linear and Heaviside activations.

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


from yaplf.utility import exp


class ActivationFunction(object):
    r"""
    Base class for activation functions. Activation functions are typically
    used in combination with neural network components, in order to add
    nonlinear behaviour. In its base version, this object is nothing but a
    function :math:`f: \mathbb R \mapsto \mathbb R`.

    Each subclass should implement a method compute, returning the activation
    function value for a specific argument. Furthermore, subclasses should
    suitably implement :obj:`__repr__`, :obj:`__hash__`, :obj:`__eq__`,
    :obj:`__ne__` and :obj:`__nonzero__`.

    EXAMPLES:

    See examples for specific subclasses, such as
    :class:`SigmoidActivationFunction` in this module.

    AUTHORS:

    - Dario Malchiodi (2010-02-22): base version
    - Dario Malchiodi (2010-03-26): added :meth:`compute_derivative` method

    """

    def compute(self, arg):
        r"""
        Return the activation function value for a specific argument. Raise
        :exc:`ValueError` if called in the base class.

        :param arg: activation function argument.

        :type arg: float

        :returns: activation function value.

        :rtype: float

        EXAMPLES:

        See examples for specific subclasses, such as
        :class:`SigmoidActivationFunction` in this module.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError('compute() not callable in base class')

    def compute_derivative(self, arg, **kwargs):
        r"""
        Return the activation function's firsst derivative value for a specific
        argument. Raise :exc:`ValueError` if called in the base class.

        :param arg: activation function derivative argument.

        :type arg: float

        :param func_value: activation function value; this argument is
          specified for sake of computational efficiency when the derivative
          value is a simple expression of the function value, such as for
          instance with sigmoid functions.

        :type func_value: float, default: ``None``, which implies the function
          value is computed anew

        :returns: activation function derivative value.

        :rtype: float

        EXAMPLES:

        See examples for specific subclasses, such as
        :class:`SigmoidActivationFunction` in this module.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError(\
            'compute_derivative() not callable in base class')


class SigmoidActivationFunction(ActivationFunction):
    r"""
    The sigmoid activation function, whose analytic form is

    .. math:: f(x) = \frac{ 1 }{ 1 + \mathrm e^{- \beta x} }

    where :math:`\beta > 0` is a parameter whose value affects the function
    steepness around 0 (the higher this value, the more the function can be
    thought of as a differentiable approximation of a step function).

    Raise a :exc:`ValueError` when the specified value for :math:`\beta` is not
    positive.

    :parameter beta: steepness value :math:`\beta`.

    :type beta: float, default: 1

    EXAMPLES:

    When the specified value for :obj:`beta` is not positive a
    :exc:`ValueError` isthrown:

    >>> from yaplf.utility.activation import SigmoidActivationFunction
    >>> SigmoidActivationFunction()
    SigmoidActivationFunction()
    >>> SigmoidActivationFunction(2)
    SigmoidActivationFunction(2)
    >>> SigmoidActivationFunction(0)
    Traceback (most recent call last):
       ...
    ValueError: the specified beta parameter is not positive

    Once an instance is created, the activation function values are obtainted
    through invocation of the :meth:`compute` method:

    >>> f = SigmoidActivationFunction()
    >>> f.compute(0)
    0.5
    >>> f.compute(1)
    0.7310585786300049

    The higher the value for :obj:`beta`, the more a sigmoidal function
    approximates the Heaviside step function (i.e., that constantly equal to
    ``0`` when its argument is negative, constantly equal to ``1`` otherwise):

    >>> f = SigmoidActivationFunction(10)
    >>> f.compute(0)
    0.5
    >>> f.compute(1)
    0.99995460213129761

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, beta=1):
        r"""
        See :class:`SigmoidActivationFunction` for full documentation.

        """

        ActivationFunction.__init__(self)
        if beta <= 0:
            raise ValueError('the specified beta parameter is not positive')
        self.beta = beta

    def __repr__(self):
        result = 'SigmoidActivationFunction('
        if self.beta != 1:
            result += str(self.beta)
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if type(self) == type(other):
            return self.beta == other.beta
        else:
            return False

    def __hash__(self):
        return hash(("SigmoidActivationFunction", hash(self.beta)))

    def __nonzero__(self):
        return True

    def compute(self, arg):
        r"""
        Compute the sigmoid function value for a specified argument.

        :param arg: argument to the sigmoid function.

        :type arg: float

        :returns: the sigmoid function value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of the :meth;`compute` method:

        >>> from yaplf.utility.activation import SigmoidActivationFunction
        >>> f = SigmoidActivationFunction()
        >>> f.compute(0)
        0.5
        >>> f.compute(1)
        0.7310585786300049

        The higher the value for :obj:`beta`, the more a sigmoidal function
        approximates the Heaviside step function (i.e., that constantly equal
        to ``0`` when its argument is negative, constantly equal to ``1``
        otherwise):

        >>> f = SigmoidActivationFunction(10)
        >>> f.compute(0)
        0.5
        >>> f.compute(1)
        0.99995460213129761

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return 1 / (1 + exp(-1 * self.beta * arg))

    def compute_derivative(self, arg, **kwargs):
        r"""
        Compute the sigmoid function first derivative value for a specified
        argument.

        :param arg: argument to the sigmoid function derivative.

        :type arg: float

        :param func_value: activation function value; this argument is
          specified for sake of computational efficiency, being the derivative
          of a sigmoid is a simple function of the sigmoid istelf, namely
          :math:`f'(x, \beta) = \beta f(x, \beta) (1 - f(x, \beta))`.

        :type func_value: float, default: ``None``, which implies the function
          value is computed anew

        :returns: sigmoid activation function derivative value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of the :meth:`compute` method, while the
        analogous derivative values are obtained through
        :meth:`compute_derivative`, either specifying or not the
        :obj:`func_value` argument:

        >>> f = SigmoidActivationFunction()
        >>> value = f.compute(1)
        >>> f.compute_derivative(1)
        0.19661193324148185
        >>> f.compute_derivative(1, func_value = value)
        0.19661193324148185

        The difference between the two approaches is not in the result
        accuracy, as the returned value are equal. However the second way is
        more efficient for it takes a smaller amount of time to complete:

        >>> import time
        >>> [time.time(), f.compute_derivative(1), time.time()] #doctest: +SKIP
        [1269616217.777189, 0.19661193324148185, 1269616217.7772729]
        >>> _[2] - _[0] #doctest: +SKIP
        8.392333984375e-05
        >>> [time.time(), f.compute_derivative(1, func_value = value),
        ... time.time()] #doctest: +SKIP
        [1269616246.24194, 0.19661193324148185, 1269616246.241956]
        >>> _[2] - _[0] #doctest: +SKIP
        1.5974044799804688e-05

        AUTHORS:

        - Dario Malchiodi (2010-02-22): base version
        - Dario Malchiodi (2010-03-26): added :meth:`compute_derivative`

        """

        try:
            func_value = kwargs['func_value']
        except KeyError:
            func_value = self.compute(arg)

        return self.beta * func_value * (1 - func_value)


class HyperbolicTangentActivationFunction(ActivationFunction):
    r"""
    The hyperbolic tangent activation function, whose analytic form is

    .. math:: f(x) = frac{ \mathrm e ^ {\beta x} - 1 }
      { \mathrm e ^ {\beta x} + 1 }

    where :math:`\beta > 0` is a parameter whose value, specified by
    :obj:`beta`, affecting the function steepness around 0 (the higher this
    value, the more the function is a differentiable approximation of a step
    function).

    Raise a :exc:`ValueError` when the specified value for :math:`\beta` is not
    positive.

    :param beta: steepness value :math:`\beta`.

    :type beta: float, default: 1

    EXAMPLES:

    When the specified value for :math:`\beta` is not positive a
    :exc:`ValueError` is thrown:

    >>> from yaplf.utility.activation import \
    ... HyperbolicTangentActivationFunction
    >>> HyperbolicTangentActivationFunction()
    HyperbolicTangentActivationFunction()
    >>> HyperbolicTangentActivationFunction(2)
    HyperbolicTangentActivationFunction(2)
    >>> HyperbolicTangentActivationFunction(0)
    Traceback (most recent call last):
       ...
    ValueError: the specified beta parameter is not positive

    Once an instance is created, the activation function values are obtainted
    through invocation of :meth:`compute`:

    >>> f = HyperbolicTangentActivationFunction()
    >>> f.compute(0)
    0.0
    >>> f.compute(1)
    0.46211715726000974

    The higher the value for :obj:`beta`, the more a sigmoidal function
    approximates the Heaviside step function (i.e., that constantly equal to
    ``0`` when its argument is negative, constantly equal to ``1`` otherwise):

    >>> f = HyperbolicTangentActivationFunction(10)
    >>> f.compute(0)
    0.0
    >>> f.compute(1)
    0.99990920426259511

    AUTHORS:

    - Dario Malchiodi (2010-04-02)

    """

    def __init__(self, beta=1):
        r"""
        See :class:`HyperbolicTangentActivationFunction` for full
        documentation.

        """

        ActivationFunction.__init__(self)
        if beta <= 0:
            raise ValueError('the specified beta parameter is not positive')
        self.beta = beta

    def __repr__(self):
        result = 'HyperbolicTangentActivationFunction('
        if self.beta != 1:
            result += str(self.beta)
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if type(self) == type(other):
            return self.beta == other.beta
        else:
            return False

    def __hash__(self):
        return hash(("HyperbolicTangentActivationFunction", hash(self.beta)))

    def __nonzero__(self):
        return True

    def compute(self, arg):
        r"""
        Compute the hyperbolic tangent function value for a specified argument.

        :param arg: argument to the hyperbolic tangent function.

        :type arg: float

        :returns: the hyperbolic tangent function value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of :meth:`compute`:

        >>> from yaplf.utility.activation import \
        ... HyperbolicTangentActivationFunction
        >>> f = HyperbolicTangentActivationFunction()
        >>> f.compute(0)
        0.0
        >>> f.compute(1)
        0.46211715726000974

        The higher the value for :obj:`beta`, the more a sigmoidal function
        approximates a step function (i.e., that constantly equal
        to ``-1`` when its argument is negative, constantly equal to ``1``
        otherwise):

        >>> f = HyperbolicTangentActivationFunction(10)
        >>> f.compute(0)
        0.0
        >>> f.compute(1)
        0.99990920426259511

        AUTHORS:

        - Dario Malchiodi (2010-04-02)

        """

        return (exp(self.beta * arg) - 1) / (exp(self.beta * arg) + 1)

    def compute_derivative(self, arg, **kwargs):
        r"""
        Compute the hyperbolic tangent function first derivative value for a
        specified argument.

        :param arg: argument to the hyperbolic tangent derivative.

        :type arg: float

        :param func_value: hyperbolic tangent value for the specified
          argument; this parameter is used for sake of computational
          efficiency, for the derivative of a hyperbolic tangent is a simple
          function of the hyperbolic tangent istelf, namely
          :math:`f'(x, \beta) = \beta (1 - f(x, \beta) ^ 2)`.

        :type func_value: float, default: ``None``, that is the original
          function value is computed anew

        :returns: the hyperbolic tangent function derivative value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of :meth:`compute`, while the
        analogous derivative values are obtained through
        :meth;`compute_derivative`, either specifying or not the
        argument :obj:`func_value`:

        >>> f = HyperbolicTangentActivationFunction()
        >>> value = f.compute(1)
        >>> f.compute_derivative(1)
        0.3932238664829637
        >>> f.compute_derivative(1, func_value = value)
        0.3932238664829637

        The difference between the two approaches is not in the result
        accuracy, as the returned value are equal. However the second way is
        more efficient for it takes a smaller amount of time to complete:

        >>> import time
        >>> [time.time(), f.compute_derivative(1), time.time()] #doctest: +SKIP
        [1270205961.0312369, 0.3932238664829637, 1270205961.0313699]
        >>> _[2] - _[0] #doctest: +SKIP
        0.00013303756713867188
        >>> [time.time(), f.compute_derivative(1, func_value = value),
        ... time.time()] #doctest: +SKIP
        [1270213102.345525, 0.3932238664829637, 1270213102.345551]
        >>> _[2] - _[0] #doctest: +SKIP
        2.5987625122070312e-05

        AUTHORS:

        - Dario Malchiodi (2010-04-02)

        """

        try:
            func_value = kwargs['func_value']
        except KeyError:
            func_value = self.compute(arg)

        return self.beta / 2.0 * (1 - func_value * func_value)


class LinearActivationFunction(ActivationFunction):
    r"""
    The linear activation function, whose analytic form is

    .. math:: f(x) = \beta x

    where :math:`\beta > 0` is a parameter whose value, specified by
    :obj:`beta`, affecting the function steepness around 0 (the higher this
    value, the more the function is a differentiable approximation of a step
    function).

    Raise a :exc:`ValueError` when the specified value for :obj:`beta` is not
    positive.

    :param beta: steepness value :math:`\beta`.

    type beta: float, default: 1

    EXAMPLES:

    When the specified value for :obj:`beta` is not positive a
    :exc:`ValueError` is thrown:

    >>> from yaplf.utility.activation import LinearActivationFunction
    >>> LinearActivationFunction()
    LinearActivationFunction()
    >>> LinearActivationFunction(2)
    LinearActivationFunction(2)
    >>> LinearActivationFunction(0)
    Traceback (most recent call last):
       ...
    ValueError: the specified beta parameter is not positive

    Once an instance is created, the activation function values are obtainted
    through invocation of :meth:`compute`:

    >>> f = LinearActivationFunction()
    >>> f.compute(0)
    0.0
    >>> f.compute(1)
    1.0

    AUTHORS:

    - Dario Malchiodi (2010-04-02)

    """

    def __init__(self, beta=1):
        r"""
        See ``LinearActivationFunction`` for full documentation.

        """

        ActivationFunction.__init__(self)
        if beta <= 0:
            raise ValueError('the specified beta parameter is not positive')
        self.beta = beta

    def __repr__(self):
        result = 'LinearActivationFunction('
        if self.beta != 1:
            result += str(self.beta)
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if type(self) == type(other):
            return self.beta == other.beta
        else:
            return False

    def __hash__(self):
        return hash(("LinearActivationFunction", hash(self.beta)))

    def __nonzero__(self):
        return True

    def compute(self, arg):
        r"""
        Compute the linear function value for a specified argument.

        :param arg: argument to the linear function.

        :type arg: float

        :returns: the linear function value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of :meth:`compute`:

        >>> from yaplf.utility.activation import LinearActivationFunction
        >>> f = LinearActivationFunction()
        >>> f.compute(0)
        0.0
        >>> f.compute(1)
        1.0

        The higher the value for :obj:`beta`, the more the function grows
        quickly around ``0``:

        >>> f = LinearActivationFunction(10)
        >>> f.compute(0)
        0.0
        >>> f.compute(1)
        10.0

        AUTHORS:

        - Dario Malchiodi (2010-04-02)

        """

        return float(self.beta * arg)

    def compute_derivative(self, arg, **kwargs):
        r"""
        Compute the linear function first derivative value for a
        specified argument.

        :param arg: argument to the linear function derivative.

        :type arg: float

        :arg func_value: function value for the specified argument; this
          parameter is maintained for sake of uniformity although it is not
          used as :math:`f'(x, \beta) = \beta` independently of :math:`x`.

        :type func_value: float, default value: ``None``, any specified value
          is ignored

        :returns: the linear function derivative value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of :meth:`compute`, while the
        analogous derivative values are obtained through
        :meth:`compute_derivative`:

        >>> f = LinearActivationFunction()
        >>> f.compute_derivative(1)
        1.0

        AUTHORS:

        - Dario Malchiodi (2010-04-02)

        """

        return float(self.beta)


class HeavisideActivationFunction(ActivationFunction):
    r"""
    The Heaviside activation function, returning ``1`` if its argument is
    greater or equal to zero, and ``0`` otherwise.

    :param left: value for the function when its argument is a negative
      number

    :type left: float, default: 0

    :param right: value for the function when its argument is a non-negative
      number

    :type right: float, default: 1

    EXAMPLES:

    Once an instance is created, the activation function values are obtainted
    through invocation of :meth:`compute`. According to the definition, any
    positive argument is mapped to ``1`` and any negative one is mapped to
    ``0``:

    >>> from numpy import random
    >>> from yaplf.utility.activation import HeavisideActivationFunction
    >>> h = HeavisideActivationFunction()
    >>> h.compute(random.normal()**2)
    1
    >>> h.compute(-1*random.normal()**2)
    0
    >>> h.compute(0)
    1

    The Heaviside function values can be scaled through using the :obj:`left`
    and :obj:`right`: parameters:

    >>> h = HeavisideActivationFunction(left=-2, right=0.5)
    >>> h.compute(3)
    0.5
    >>> h.compute(-1)
    -2

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, left=0, right=1):
        r"""
        See :class:`HeavisideActivationFunction` for full documentation.

        """

        ActivationFunction.__init__(self)
        self.left = left
        self.right = right

    def __repr__(self):
        result = 'HeavisideActivationFunction('
        if self.left != 0:
            result += 'left=' + self.left
            if self.right != 0:
                result += ', '
        if self.right != 0:
            result += 'right=' + self.right
        return result + ')'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other) and self.left == other.left \
            and self.right == other.right

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("HeavisideActivationFunction"), self.left, self.right)

    def __nonzero__(self):
        return True

    def compute(self, arg):
        r"""
        Compute the Heaviside function value for a specified argument.

        :param arg: argument to the Heaviside function.

        :type arg: float

        :returns: the Heaviside function value.

        :rtype: float

        EXAMPLES:

        Once an instance is created, the activation function values are
        obtainted through invocation of :meth:`compute`. According
        to the definition, any positive argument is mapped onto ``1`` and any
        negative one is mapped onto ``0``:

        >>> from numpy import random
        >>> from yaplf.utility.activation import HeavisideActivationFunction
        >>> h = HeavisideActivationFunction()
        >>> h.compute(random.normal()**2)
        1
        >>> h.compute(-1*random.normal()**2)
        0
        >>> h.compute(0)
        1

        The Heaviside function values can be scaled through using the
        :obj:`left` and :obj:`right`: parameters:

        >>> h = HeavisideActivationFunction(left=-2, right=0.5)
        >>> h.compute(3)
        0.5
        >>> h.compute(-1)
        -2

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return (self.left if arg < 0 else self.right)

    def compute_derivative(self, arg, **kwargs):
        r"""
        Compute the function first derivative value for a specified
        argument. Inherited from :class:`ActivationFunction` altough not
        implemented for the Heaviside function is not differentiable: a
        :exc:`NotImplementedError` is thrown when this function is invoked.


        - Dario Malchiodi (2010-03-26)

        """

        if arg != 0:
            return 0

        raise NotImplementedError('compute_derivative not callable in' + \
            'HeavisideActivationFunction')
