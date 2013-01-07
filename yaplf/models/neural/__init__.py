
r"""
Package handling neural models in yaplf

Module :mod:`yaplf.models.neural` contains all the classes handling neural
models in yaplf.

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

from numpy import array, dot, hstack, sqrt, arccos, sin, sign, transpose

from matplotlib import pyplot
from matplotlib.patches import Circle, Arrow
from matplotlib.cm import Greys

from yaplf.utility.activation import ActivationFunction, \
    HeavisideActivationFunction
from yaplf.utility import to_column
from yaplf.models import Classifier


class Perceptron(Classifier):
    r"""
    Model representing (one layer-) perceptrons, that is models abstractly
    consisting of :math:`n` input units and :math:`m` output units. A given
    perceptron maps a sequence :math:`x_1, \dots, x_n` of *input values*, that
    is numeric values for the input units, into another sequence
    :math:`y_1, \dots, y_m` of *output values* (numeric values for the output
    units). The mapping depends on:

    - a :math:`n \times m` real matrix :math:`W = [w_{ij}]_{i=1..n}^{j=1..m}`,
      where :math:`w_{ij}` identifies a *connection* between :math:`i`-th input
      and :math:`j`-th output unit.

    - a sequence of :math:`m` numeric values :math:`\theta_1, \dots, \theta_m`,
      where :math:`\theta_j` identifies a *threshold* for :math:`j`-th output
      unit.

    - a function :math:`f: \mathbb R \mapsto \mathbb R` identifying an
      *activation function* for the output units.

    More precisely, the value for :math:`j`-th output unit is obtained as
    :math:`y_i = f \left( \sum_{i=1}^n x_i w_{ij} - \theta_i \right)`.

    .. function:: Perceptron(weights[, threshold=(0, ..., 0),
      activation=HeavisideActivationFunction()])

    :param weights: perceptron weights; length of this argument identifies
      the number of output units; each element, corresponding to a given output
      unit, is in turn a list or tuple of numeric values describing the
      connections between each input and this output unit. Thus, all elements
      of this argument should have the same length and this length identifies
      the number of input units.

    :type weights: sequence of sequences of numeric values

    :param threshold: thresholds for the output units (default value: a tuple
      filled with zeroes, corresponding to the absence of thresholds).

    :type threshold: sequence of numeric values

    :param activation: activation function for the output units (default value:
      ``HeavisideActivationFunction()``).

    :type activation: :class:`yaplf.utility.activations.ActivationFunction`


    :returns: a :class:`Perceptron` instance.

    :raises: ``ValueError``

    EXAMPLES

    The only mandatory argument in the constructor is the one describing the
    weight matrix; the sequence specified as argument identifies both the
    number of input and output values. Precisely, its length will be equal to
    the number of outputs, while each element should be in turn a numeric
    sequence of fixed length, corresponding to the number of inputs. For
    instance, the following instructions build a :class:`Perceptron` instance
    where the ``weights`` constructor argument is ``((1, 1),)``, that is a
    tuple containing a 2-elements tuple. This will account for a perceptron
    with one output and two inputs, where both input-to-output connections will
    be set to 1:

        >>> from yaplf.models.neural import Perceptron
        >>> Perceptron(((1, 1),))
        Perceptron([array([1, 1])])

    It is worth nothing that, as all the named arguments are set to their
    default values, this perceptron will have a null-threshold,
    Heaviside-activated output unit. Similarly, the following instruction will
    build a perceptron with two input units and two output units:

        >>> Perceptron(((1, 1), (8, -4)))
        Perceptron([array([1, 1]), array([ 8, -4])])

    As the ``weight`` argument is a numeric sequence encoding a matrix,
    whenever their arguments have not the same size a ``ValueError`` is thrown:

        >>> Perceptron(((1, 1), (8, -4, 9)))
        Traceback (most recent call last):
        ...
        ValueError: weights in ((1, 1), (8, -4, 9)) have different lengths

    Specification of threshold values is done through the ``threshold`` named
    argument, as in the following examples (note that in the first instruction
    the 1-element tuple requires a trailing comma so as to avoid that `(1)` is
    intrepreted as the constant value `1`:

        >>> Perceptron(((1, 1),), threshold=(-1,))
        Perceptron([array([1, 1])], threshold=[-1])
        >>> Perceptron(((1, 1), (8, -4)), threshold=(-1, 1))
        Perceptron([array([1, 1]), array([ 8, -4])], threshold=[-1, 1])

    If the ``weights`` and ``threshold`` named arguments values have
    incompatible shapes (that is, if their size is not equal) a ``ValueError``
    is thrown, as both quantities identify the number of output units):

        >>> Perceptron(((1, 1), (8, -4)), threshold=(-1,))
        Traceback (most recent call last):
        ...
        ValueError: weights in ((1, 1), (8, -4)) and thresholds in (-1,)
        refer to different output vectors

    The named argument ``activation`` is used in order to use specific
    activation functions. The corresponding values are instances of subclasses
    of :class:`yaplf.utility.activation.ActivationFunction`; for instance, the
    following code builds a perceptron whose output unit is equipped with a
    sigmoidal activation function:

        >>> from yaplf.utility.activation import SigmoidActivationFunction
        >>> s = SigmoidActivationFunction()
        >>> Perceptron(((1, 1),), threshold=(-1,), activation=s)
        Perceptron([array([1, 1])], threshold=[-1], activation=
        SigmoidActivationFunction())

    Once a :class:`Perceptron` instance is available, the outputs corresponding
    to specific input values can be obtained through invocation of the
    :meth:`compute` function, inherited from :class:`yaplf.models.Classifier`:

        >>> p = Perceptron(((-2, 4, 0.6), (-1, -5, 9)))
        >>> p.compute((-2, 0, -1))
        array([1, 0])
        >>> s = SigmoidActivationFunction(beta=.1)
        >>> p = Perceptron(((.3, 9.56),), threshold=(1.7,), activation=s)
        >>> p.compute((0, 4))
        0.97476587330696185

    Note how the value returned by :meth:`compute` is a numpy array when the
    perceptron has more than one output unit, and a numeric value otherwise.
    Consider the following perceprton expressly tailored in order to compute
    the bitwise AND. Apart from invoking repeatedly :meth:`compute`,
    there is a easier way in order to verify the latter statement; it consists
    in calling the :meth:`yaplf.models.Model.test` method specifying as
    argument a labeled sample to be tested:

        >>> p = Perceptron(((4, 4),), threshold=(6,),
        ... activation=SigmoidActivationFunction(0.8))
        >>> from yaplf.data import LabeledExample
        >>> and_sample = (LabeledExample((1., 1.), (1,)),
        ... LabeledExample((0., 0.), (0,)), LabeledExample((0, 1), (0,)),
        ... LabeledExample((1, 0), (0,)))
        >>> p.test(and_sample)
        0.021180024091718493

    Another way to visualize a perceptron's behaviour is through the
    :class:`yaplf.models.neural.plot.PerceptronDecisionFunctionPlot`. Instances
    of this class, once created specifying a :class:`Perceptron` object as
    parameter, can invoke the :meth:`plot` in order to produce a graphic object
    summarizing the outputs for a given range of possible inputs:

        >>> from yaplf.models.neural.plot import PerceptronDecisionFunctionPlot
        >>> dfp = PerceptronDecisionFunctionPlot(p)
        >>> dfp.plot((-5, 5), (-5, 5), plot_points = 100,
        ... contours = (0.1, 0.5, 0.9),
        ... contour_color = ('red', 'green', 'blue'), shading=True)

    Here the first two arguments represent the ranges for the possible values
    for the two perceptron input units, and the obtained graph contains a
    colored gradient shading from white to black in order to visualize how the
    perceptron output varies w.r.t. the possible input values (named argument
    ``shading``), highlighting through colored curves specific output values
    (where named arguments ``contours`` and ``contour_color`` specify these
    values and the color of the corresponding curves, while ``plot_points``
    refers to the precision to be used in order to approximate those curves
    through a set of successive segments).

    Only perceptrons having two or three inputs allow invocation of the
    :meth:`plot` function. In the second case it will be necessary to specify
    three input value ranges, and the result will be a 3D graph:

        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,),
        ... activation = SigmoidActivationFunction(beta=.1))
        >>> p.plot((-5, 5), (-5, 5), (-5, 5), plot_points=20,
        ... contours=(0.1, 0.5, 0.9), contour_color=('red', 'green', 'blue'),
        ... shading=True)

    :class:`Perceptron` objects have three properties named ``weights``,
    ``threshold`` and ``activation`` returning the corresponding object
    components:

        >>> p.weights
        [array([ 0.3 ,  9.56,  0.2 ])]
        >>> p.threshold
        [1.7]
        >>> p.activation
        SigmoidActivationFunction(0.1)

    These properties can be used also in order to set the components' values:

        >>> p.threshold = (1,)

    If such properties are used in order to leave a perceptron in an incoherent
    state (that is, with a number of thresholds different from the number of
    output units), a ``ValueError`` is thrown:

        >>> p.threshold = (1, 0.5)
        ...
        ValueError: weights in [array([ 0.3 ,  9.56,  0.2 ])] and thresholds in
        (1, 0.5) refer to different output vectors

    The method :meth:`set_weights_and_threshold` allow the simultaneous
    modification of weights and thresholds:

        >>> p.set_weights_and_threshold([[2, 6, 4]], [5])
        >>> p
        Perceptron([array([2, 6, 4])], threshold=[5], activation=
        SigmoidActivationFunction(0.1))


    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, weights, **kwargs):
        r"""
        See :class:`Perceptron` for full documentation.

        """

        Classifier.__init__(self)
        self.__weights = self.__activation = None
        try:
            threshold = kwargs['threshold']
            self.has_threshold = True
        except KeyError:
            self.has_threshold = False
            threshold = (0,) * len(weights)

        Perceptron.check_weights_and_threshold(weights, threshold)
        self.set_weights_and_threshold(weights, threshold)

        try:
            self.activation = kwargs['activation']
            #activation function
        except KeyError:
            self.activation = HeavisideActivationFunction()

    @classmethod
    def check_weights_and_threshold(cls, weights, threshold):
        r"""
        Class method used in order to check that two sequences describing,
        respectively, weights and thresholds of a perceptron have compatible
        shapes.

        .. function:: check_weights_and_threshold(cls, weights, threshold)

        :param cls: class on which the method is invoked.

        :type cls: :class:`Perceptron`

        :param weights: weights for a perceptron

        :type weights: sequence of sequences of numeric values

        :param threshold: thresholds for a perceptron

        :type threshold: sequence of numeric values

        :returns: the method doesn't return anything if ``weights`` and
          ``threshold`` have valid values

        :raises: ``ValueError`` when ``weights`` and ``threshold`` have
          incompatible values

        EXAMPLES

        This method checks two conditions:

        - all elements in ``weights`` should be sequences of numeric values
          having the same length;

        - the number of sequences in ``weights`` should be equal to the length
          of ``threshold``.

        >>> Perceptron.check_weights_and_threshold(((2, 5, -1), (0.5, 7, 12)),
        ... (5, 6))
        >>> Perceptron.check_weights_and_threshold(((2, 5, -1), (0.5, 7, 12)),
        ... (5,))
        ...
        ValueError: weights in ((2, 5, -1), (0.5, 7, 12)) and thresholds in
        (5,) refer to different output vectors
        >>> Perceptron.check_weights_and_threshold(((2, 5, -1), (0.5, 7)),
        ... (5, 6))
        ...
        ValueError: weights in ((2, 5, -1), (0.500000000000000, 7)) have
        different lengths

        """

        length = len(weights[0])
        for weight in weights[1:]:
            if len(weight) != length:
                raise ValueError('weights in ' + str(weights) + \
                    ' have different lengths')

        if len(threshold) != len(weights):
            raise ValueError('weights in ' + str(weights) + \
                ' and thresholds in ' + str(threshold) + \
                ' refer to different output vectors')

    def __set_weights_and_threshold(self, weights, threshold):
        r"""
        Private method setting weights and threshold of a perceptron.
        The method **does not** check whether weights and threshold refer
        to the same number of input and output units. It should be only
        called by the public setter :meth:`set_weights_and_threshold`.

        .. function:: __set_weights_and_threshold(self, weights, threshold)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param weights: weights for a perceptron

        :type weights: sequence of sequences of numeric values

        :param threshold: thresholds for a perceptron

        :type threshold: sequence of numeric values

        """

        self.__weights = [hstack((weights[i], (threshold[i],)))
                for i in range(len(threshold))]
        self.notify_observers()

    def set_weights_and_threshold(self, weights, threshold):
        r"""
        Method used in order to set both weights and thresholds in a
        perceptron.

        .. function:: set_weights_and_threshold(self, weights, threshold)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param weights: weights for a perceptron

        :type weights: sequence of sequences of numeric values

        :param threshold: thresholds for a perceptron

        :type threshold: sequence of numeric values

        :raises: ``ValueError`` when ``weights`` and ``threshold`` have
          incompatible values

        EXAMPLES

        The method does not return a value:

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.set_weights_and_threshold([[2, 6, 4]], [5])
        >>> p
        Perceptron([array([2, 6, 4])], threshold=[5])

        If the parameters are incompatible the method raises a ``ValueError``
        without modifying the object state:

        >>> p.set_weights_and_threshold(((2, 5, -1), (0.5, 7, 12)), (5,))
        ...
        ValueError: weights in ((2, 5, -1), (0.5, 7, 12)) and thresholds in
        (5,) refer to different output vectors
        >>> p.set_weights_and_threshold(((2, 5, -1), (0.5, 7)), (5, 6))
        ...
        ValueError: weights in ((2, 5, -1), (0.500000000000000, 7)) have
        different lengths

        """

        Perceptron.check_weights_and_threshold(weights, threshold)
        self.__set_weights_and_threshold(weights, threshold)

    def get_weights(self):
        r"""
        Getter method returning a perceptron's weights.

        .. function:: get_weights(self)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :returns: perceptron weights

        :rtype: sequence of sequences of numeric values

        EXAMPLES

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.get_weights()
        [array([ 0.3 ,  9.56,  0.2 ])]

        """

        return [w[:-1] for w in self.__weights]

    def get_threshold(self):
        r"""
        Getter method returning a perceptron's thresholds.

        .. function:: get_threshold(self)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :returns: perceptron thresholds

        :rtype: sequence of numeric values

        EXAMPLES

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.get_threshold()
        [1.7]

        """

        return [w[-1] for w in self.__weights]

    def set_weights(self, weights):
        r"""
        Setter method modifying a perceptron's weights

        .. function:: set_weights(self, weights)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param weights: weights for a perceptron

        :type weights: sequence of sequences of numeric values

        :raises: ``ValueError`` when ``weights`` is incompatible with the
          current state of the perceptron

        EXAMPLES

        The method does not return a value:

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.set_weights([[2, 6, 4]])
        >>> p
        Perceptron([array([2., 6., 4.])], threshold=[1.7])

        If the parameter ``weight`` is incompatible with the current state of
        the perceptron (that is, its length differs with the number of output
        units or its elements do not have the same length) the method raises a
        ``ValueError`` without modifying the object state:

        >>> p.set_weights(((2, 5, -1), (0.5, 7, 12)))
        ...
        ValueError: weights in ((2, 5, -1), (0.5, 7, 12)) and thresholds in
        [1.7] refer to different output vectors
        >>> p.set_weights(((2, 5, -1), (0.5,)))
        ...
        ValueError: weights in ((2, 5, -1), (0.5,)) have different lengths

        """

        threshold = self.get_threshold()
        Perceptron.check_weights_and_threshold(weights, threshold)
        self.__set_weights_and_threshold(weights, threshold)

    def set_threshold(self, threshold):
        r"""
        Setter method modifying a perceptron's thresholds

        .. function:: set_threshold(self, threshold)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param threshold: thresholds for a perceptron

        :type threshold: sequence of numeric values

        :raises: ``ValueError`` when ``threshold`` is incompatible with the
          current state of the perceptron

        EXAMPLES

        The method does not return a value:

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.set_threshold([-1])
        >>> p
        Perceptron([array([ 0.3,  9.56,  0.2 ])], threshold = [-1.0])

        If the parameter ``threshold`` is incompatible with the current state
        of the perceptron (that is, its length differs with the number of
        output units) the method raises a ``ValueError`` without modifying the
        object state:

        >>> p.set_threshold((-1, 0))
        ...
        ValueError: weights in [array([ 0.3 ,  9.56,  0.2 ])] and thresholds
        in (-1, 0) refer to different output vectors

        """

        weights = self.get_weights()
        Perceptron.check_weights_and_threshold(weights, threshold)
        self.__set_weights_and_threshold(weights, threshold)

    def get_activation(self):
        r"""
        Getter method returning a perceptron's activation function.

        .. function:: get_activation(self)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :returns: activation function of the perceptron

        :rtype: :class:`yaplf.utility.activation.ActivationFunction`

        EXAMPLES

        When a perceptron is instantiated without specifying an activation
        function, the latter defaults to an instance
        of :class:`yaplf.utility.activation.HeavisideActivationFunction`:

        >>> from yaplf.models.neural import Perceptron
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.get_activation()
        HeavisideActivationFunction()

        """

        return self.__activation

    def set_activation(self, activation):
        r"""
        Setter method modifying a perceptron's activation function

        .. function:: set_activation(self, threshold)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param activation: activation function for a perceptron

        :type activation: :class:`yaplf.utility.activation.ActivationFunction`

        :raises: ``ValueError`` when ``activation`` is incompatible with the
          required type

        EXAMPLES

        >>> from yaplf.models.neural import Perceptron
        >>> from yaplf.utility.activation import SigmoidActivationFunction
        >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,))
        >>> p.set_activation(SigmoidActivationFunction(beta=2))
        >>> p
        Perceptron([array([ 0.3,  9.56,  0.2 ])], threshold=[1.7],
        activation=SigmoidActivationFunction(2))

        If the parameter ``activation`` is not an instance of a subclass of
        :class:`yaplf.utility.activation.ActivationFunction` the method raises
        a ``ValueError`` without modifying the object state:

        >>> p.set_activation(9)
        ...
        ValueError: 9 is not an activation funcion

        """

        if isinstance(activation, ActivationFunction):
            self.__activation = activation
            self.notify_observers()
        else:
            raise ValueError(str(activation) + ' is not an activation funcion')

    # properties linked to the perceptron's weights, thresholds and
    # activation function

    weights = property(get_weights, set_weights)
    threshold = property(get_threshold, set_threshold)
    activation = property(get_activation, set_activation)

    def __repr__(self):
        r"""
        Private method returning a valid description for the perceptron object.
        """

        result = 'Perceptron(' + str(self.weights)
        if self.has_threshold:
            result += ', threshold=' + str(self.threshold)
        if self.__activation != HeavisideActivationFunction():
            result += ', activation=' + self.__activation.__repr__()
        result += ')'
        return result

    def __eq__(self, other):
        r"""
        Private method checking wether a perceptron object has the same
        contents of another objects. By definition, equality holds only when
        the compared object is an instance of :class:`Perceptron` having the
        same weights, thresholds and activation function.

        """

        if type(self) == type(other):
            return self.weights == other.weights and \
                self.threshold == other.threshold and \
                self.activation == other.activation
        else:
            return False

    def __ne__(self, other):
        r"""
        Private method checking wether a perceptron object has different
        contents w.r.t. another object. Its behaviour has been obtained
        through negation of the value returned by :meth:`__eq__`.

        """

        return not self == other

    def __hash__(self):
        r"""
        Private method generating a hash value for :class:`Perceptron` objects.
        As an instance is identified by the augmented weight matrix (that is,
        the matrix containing both connection weights and thresholds), presence
        of thresholds and activation function, the hash value is obtained by
        collecting these objects in a tuple and returning the hash value of the
        latter.

        """

        return hash((self.__weights, self.has_threshold, self.activation))

    def __nonzero__(self):
        r"""
        Private method returning True if the object has non-null content.

        """

        return self.__weights and self.has_threshold and self.activation

    def get_num_inputs(self):
        r"""
        Returns the number of input units.

        .. function:: get_num_inputs(self)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :returns: number of input units

        :rtype: integer

        EXAMPLES:

        The number of input units is indirectly specified when invoking the
        class constructor, as this quantity should be equal to the size of
        all weights list/tuple elements:

        ::
            >>> from yaplf.models.neural import Perceptron
            >>> p = Perceptron(((1, 1),))
            >>> p.get_num_inputs()
            2
            >>> p = Perceptron(((1, 1), (8, -4)), threshold = (0, -1))
            >>> p.get_num_inputs()
            2
            >>> p = Perceptron(((3, 1, 7, 4), (-4, 3, 1.5, 5)))
            >>> p.get_num_inputs()
            4

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return len(self.weights[0])

    def get_num_outputs(self):
        r"""
        Returns the number of output units.

        .. function:: get_num_outputs(self)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :returns: number of output units

        :rtype: integer


        EXAMPLES:

        The number of output units is indirectly specified when invoking the
        class constructor, as this quantity should be equal to the size of the
        list/tuple used in order to specify weights:

        ::
            >>> from yaplf.models.neural import Perceptron
            >>> p = Perceptron(((1, 1),))
            >>> p.get_num_outputs()
            1
            >>> p = Perceptron(((1, 1), (8, -4)), threshold = (0, -1))
            >>> p.get_num_outputs()
            2
            >>> p = Perceptron(((3, 1, 7, 4), (-4, 3, 1.5, 5)))
            >>> p.get_num_outputs()
            2

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return len(self.weights)

    def decision_function(self, pattern):
        r"""
        Compute the decision function value for the supplied pattern. In a
        perceptron, this value equals the output units'.

        .. function:: decision_function(self, pattern)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param pattern: pattern to be fed to the input units.

        :type pattern: sequence of numeric values

        :returns: decision function value(s) for the output units when the
          content of ``pattern`` is fed to the input units.

        :rtype: number or numpy array

        EXAMPLES:

        The ``pattern`` argument should be a list or tuple whose size equals
        the number of input units. The returned value is a number when there
        is only an output unit and a numpy array of numeric values otherwise.

        If no activation function has been specified during object
        initialization, the decision function will take on `0` or `1` values:

            >>> from yaplf.models.neural import Perceptron
            >>> p = Perceptron(((1, 1),))
            >>> p.decision_function((4, 3))
            1
            >>> p.decision_function((-4, 3))
            0

        Decision function values depend on the chosen activation function. For
        instance, when using a sigmoidal activation all decision function
        values range smoothly between `0` and `1`:

            >>> from yaplf.utility.activation import SigmoidActivationFunction
            >>> p = Perceptron(((1, 1),),
            ... activation = SigmoidActivationFunction(beta = 3))
            >>> p.decision_function((4, 3))
            0.99999999924174388
            >>> p.decision_function((-2.5, 3))
            0.81757447619364365
            >>> p.decision_function((-2.5, 2))
            0.18242552380635635
            >>> p.decision_function((-4, 3))
            0.047425873177566781

        When length of ``pattern`` argument is not equal to the number of
        input units a ``ValueError`` is thrown:

            >>> p.decision_function((-4, 3, 0))
            Traceback (most recent call last):
            ...
            ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return self.compute(pattern)

    def compute(self, pattern):
        r"""
        Compute the output unit values for the supplied pattern. In a
        perceptron, this value equals the decision function's.

        .. function:: compute(self, pattern)

        :param self: object on which the method is invoked.

        :type self: :class:`Perceptron`

        :param pattern: pattern to be fed to the input units.

        :type pattern: sequence of numeric values

        :returns: value(s) in the output units when the content of ``pattern``
          is fed tu the input units.

        :rtype: number or numpy array

        EXAMPLES:

        The ``pattern`` argument should be a list or tuple whose size equals
        the number of input units. The returned value is a number when there
        is only an output unit and a numpy array of numeric values otherwise.

        If no activation function has been specified during object
        initialization, output values will take on `0` or `1` values:

            >>> from yaplf.models.neural import Perceptron
            >>> p = Perceptron(((1, 1),))
            >>> p.compute((4, 3))
            1
            >>> p.compute((-4, 3))
            0

        Output values depend on the chosen activation function. For instance,
        when using a sigmoidal activation all outputs range smoothly between
        `0` and `1`:

            >>> from yaplf.utility.activation import SigmoidActivationFunction
            >>> p = Perceptron(((1, 1),),
            ... activation = SigmoidActivationFunction(beta = 3))
            >>> p.compute((4, 3))
            0.99999999924174388
            >>> p.compute((-2.5, 3))
            0.81757447619364365
            >>> p.compute((-2.5, 2))
            0.18242552380635635
            >>> p.compute((-4, 3))
            0.047425873177566781

        When length of ``pattern`` argument is not equal to the number of
        input units a ``ValueError`` is thrown:

            >>> p.compute((-4, 3, 0))
            Traceback (most recent call last):
                ...
            ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(self.__weights) == 1:
            return self.activation.compute(dot(self.__weights[0],
                hstack((pattern, (-1,)))))
        else:
            return array([self.activation.compute(dot(w,
                hstack((pattern, (-1,))))) for w in self.__weights])
