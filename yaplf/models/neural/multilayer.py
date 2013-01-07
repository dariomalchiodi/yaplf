
r"""
Package handling multilayer perceptron models in yaplf

Package yaplf.models.neural contains all the classes handling multilayer
perceptrons in yaplf.

TODO:

- MLPerceptron compute net named argument documentation
- Perceptron.graph documewntation
- pep8 checked
- pylint score: 9.54

AUTHORS:

- Dario Malchiodi (2011-01-04): factored out from yaplf.models.neural

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************


from numpy import array, dot, hstack, transpose

from yaplf.utility.activation import HeavisideActivationFunction
from yaplf.utility import to_column
from yaplf.models import Classifier


class MultilayerPerceptron(Classifier):
    r"""
    A multilayer perceptron is made up by a set of regular perceptrons stacked
    up in layers.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``dimensions`` -- list or tuple of `n` integers containing the layer
      dimensions.

    - ``connections`` -- list or tuple of `n-1` list or tuples of numeric
      values for the multilayer perceptron's connections. Note that the
      elements of this argument have a size automatically determined by the
      value specified for ``dimensions``.

    - ``thresholds`` -- list or tuple (default: list of zeroes, meaning the
      absence of thresholds) containing the multilayer perceptron's thresholds.
      Again, the shape of this argument is determined by the value specified
      for ``dimensions``.

    - ``activation`` -- ActivationFunction or list/tuple of ActivationFunction
      (default: ``HeavisideActivationFunction()``) activation function(s) to
      be used for the various units.

    OUTPUT:

    Classifier object containing the multilayer perceptron.

    EXAMPLES:

    Conbsider the following typical example of a two-layers perceptron able
    to compute the binary XOR function:

        >>> from yaplf.models.neural import MultilayerPerceptron
        >>> dimensions = (2, 2, 1)
        >>> connections = (((1, -1), (-1, 1)), ((1, 1),))
        >>> thr = ((-1, -1), (-1,))
        >>> p = MultilayerPerceptron(dimensions, connections, thresholds = thr)
        >>> p
        MultilayerPerceptron((2, 2, 1), (((1, -1), (-1, 1)), ((1, 1),)),
        thresholds = ((-1, -1), (-1,)))
        >>> p.compute((1, 0))
        1
        >>> from yaplf.data import LabeledExample
        >>> xor_sample = [LabeledExample((0, 0), (0,)),
        ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)),
        ... LabeledExample((1, 1), (0,))]
        >>> p.test(xor_sample)
        0.0

    Specification of the ``full_state`` named argument allow the computation
    of the global state of the perceptron, i.e. the set of all units' values
    instead of the sole output ones:

    ::

        >>> p.compute((1, 0), full_state = True)
        [array([1, 0]), array([1])]

    Different activation functions can be attached to each layer. For instance,
    it is possible to modify the previous network using the default activation
    function in the hidden layer and the sigmoidal one in the output layer:

    ::

        >>> from yaplf.utility.activation import HeavisideActivationFunction, \
        ... SigmoidActivationFunction
        >>> q = MultilayerPerceptron(dimensions, connections, thresholds = thr,
        ... activations = (HeavisideActivationFunction(),
        ... SigmoidActivationFunction()))

    Of course this change has an effect on the perceptron performance, for
    the previous structure heavily relied on a Heaviside function peculiarity
    (precisely, h(0) = 0 where h is the Heaviside function):

    ::

        >>> q.test(xor_sample, verbose = True)
        (0, 0) mapped to 0.26894142137, label is (0,), error [ 0.07232949]
        (0, 1) mapped to 0.5, label is (1,), error [ 0.25]
        (1, 0) mapped to 0.5, label is (1,), error [ 0.25]
        (1, 1) mapped to 0.26894142137, label is (0,), error [ 0.07232949]
        MSE 0.161164744064
        0.16116474406425663

    AUTHORS:

    - Dario Malchiodi (2010-03-22)

    """

    def __init__(self, dimensions, connections, **kwargs):
        r"""
        See ``MultilayerPerceptron`` for full documentation.

        """

        Classifier.__init__(self)

        self.dimensions = dimensions
        self.connections = list(connections)

        try:
            self.thresholds = kwargs['thresholds']
            self.has_thresholds = True
        except KeyError:
            self.thresholds = None
            self.has_thresholds = False

        try:
            self.activations = kwargs['activations']
        except KeyError:
            self.activations = HeavisideActivationFunction()

        MultilayerPerceptron.check_size(self.dimensions, self.connections,
            self.thresholds, self.activations)

        self.notify_observers()

    def __repr__(self):
        result = "MultilayerPerceptron(" + str(self.dimensions) + ", "
        result += str(self.connections)
        if self.has_thresholds:
            result += ", thresholds = " + str(self.thresholds)
        if self.activations != HeavisideActivationFunction():
            result += ", activations = " + str(self.activations)
        result += ")"
        return result

    @classmethod
    def check_size(cls, dimensions, connections, thresholds,
        activations=HeavisideActivationFunction()):
        r"""
        Checks whether the specified connections are compatible in order to
        build a multilayer perceptron, that is:

        - each element of ``connections`` should be a bidimensional list or
          tuple whose size is that of two successive values in ``dimensions``;

        - each element of ``thresholds`` should be a list or tuple whose length
          is that of one element of ``dimensions``.

        When these requirements are met the function returns silently,
        otherwise a ``ValueError`` is thrown.

        Note that this class function is used only internally by the class
        constructor in order to check the validity of specified arguments.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``dimensions`` -- list or tuple of integers describing the layers'
          size.

        - ``connections`` -- list or tuple of numeric values containing the
          multilayer perceptron's connection values.

        - ``thresholds`` -- list or tuple of numeric values containing the
          multilayer perceptron's thresholds.

        - ``activations`` -- ``ActivationFunction`` or list/tuple of
          ``ActivationFunction`` (default: ``HeavisideActivationFunction()``)
          describing the units' activation function.

        OUTPUT:

        The function returns silently if the provided data are compatible, and
        throws a ``ValueError`` otherwise.

        EXAMPLES:

        Consider a simple multilayer perceptron with three layers denoted
        respectively as input, hidden and output. More precisely, insert into
        these layers two input units, two hidden units and one output unit. If
        thresholds are used, this architecture requires the following
        specifications to the constructor arguments:

        - ``(2, 2, 1)`` as layer dimensions;

        - concerning connections, any two-elements list whose first component
          is a 2x2 list (as there are two input and two hidden units) and whose
          second one is a two-element list (as there are two hidden units and
          one output one) will be accepted;

        - analogously, any three-elements list whose first two components are
          list containing two elements (as there are two input and hidden
          units) and whose last one is a singleton list (as only one output
          unit is provided will be accepted.

        For intsance, the following values are thus valid:

        ::

            >>> dimensions = [2, 2, 1]
            >>> connections = (((1, 2), (3, 4)), ((5, 6),))
            >>> thresholds = ((0, 1), (2,))

        Consequently, using these variables as arguments in ``get_size`` will
        have no apparent effect

            >>> from yaplf.models.neural import MultilayerPerceptron
            >>> MultilayerPerceptron.check_size(dimensions, connections,
            ... thresholds)

        Note how the function concerns only on the shape of its second and
        third argument, independently of the use of list, tuples, or numpy
        arrays:

        ::

            >>> from numpy import array
            >>> connections = (([1, 2], (3, 4)), array(((5, 6),)))

        Of course the function behaviour changes when the supplied values are
        not compatible. This can happen:

        - when the layer dimensions don't match up between ``dimensions`` and
          ``connections``:

        ::

            >>> MultilayerPerceptron.check_size([1, 3, 1], connections,
            ... thresholds)
            Traceback (most recent call last):
            ...
            ValueError: connections and dimensions not compatible

        - when the layer dimension don't match up between ``dimensions`` and
          ``thresholds``:

        ::

            >>> MultilayerPerceptron.check_size(dimensions, connections,
            ... ((0, 1, 2), (3,)))
            Traceback (most recent call last):
            ...
            ValueError: thresholds and dimensions not compatible

        - when elements in ``connections`` use different lengths when referring
          to a same layer:

        ::

            >>> wrong_conn = (((1, 2), (3,)), ((5, 6),))
            >>> MultilayerPerceptron.check_size(dimensions, wrong_conn,
            ... thresholds)
            Traceback (most recent call last):
            ...
            ValueError: connections contains incompatible data

        Consider a more complex architecture made up by four levels (input,
        first hidden, second hidden, and output), having respectively 4, 2, 3
        and 2 units. The setting for dimensions, connections and thresholds is
        now more complex:

        ::

            >>> dimensions = (4, 2, 3, 2)
            >>> connections = (((1, 1, 1, 1), (2, 2, 2, 2)),
            ... ((3, 3), (3, 3), (3, 3)), ((4, 4, 4), (4, 4, 4)))
            >>> thresholds = ((1, 1), (2, 2, 2), (3, 3))
            >>> MultilayerPerceptron.check_size(dimensions, connections,
            ... thresholds)

        AUTHORS:

        - Dario Malchiodi (2010-03-22)

        """

        # check connections and layers dimension compatibility
        if [len(connections[0][0])] + [len(c) for c in connections] != \
            list(dimensions):
            raise ValueError('connections and dimensions not compatible')

        # check each element in connections is made up of list/tuples of
        # equal length (they refer to connections getting to the same units)
        for lengths in [[len(c) for c in conn] for conn in connections]:
            first = lengths[0]
            for len_ in lengths[1:]:
                if len_ != first:
                    raise ValueError('connections contains incompatible data')

        # check each element in connections is compatible with the subsequent
        # one (the same layer has incoming and outcoming connections)
        for pos in range(len(connections) - 1):
            if len(connections[pos]) != len(connections[pos + 1][0]):
                raise ValueError('connections contains incompatible data')

        # check thresholds and layers dimension compatibility
        if thresholds is not None and \
            [len(t) for t in thresholds] != list(dimensions[1:]):
            raise ValueError('thresholds and dimensions not compatible')

        # check activations and layers dimension compatibility
        try:
            if [len(a) for a in activations] != list(dimensions[1:]):
                raise ValueError('thresholds and dimensions not compatible')
        except TypeError:
            pass
            # unique activation function, no check required

    def get_activation(self, num):
        r"""
        Return the activation function for ``num``-th layer.

        INPUT

        - ``self`` object on which the function is invoked.

        - ``num`` -- integer denoting the layer whose activation function is
          requested.

        OUTPUT

        ActivationFunction -- Activation function for the specified layer

        EXAMPLES

        This function is internally used by the ``run`` function.

        AUTHORS

        - Dario Malchiodi (2010-03-22)

        """

        if type(self.activations) is type([]) or \
            type(self.activations) is type(()):
            return self.activations[num]
        else:
            return self.activations

    def compute(self, pattern, full_state=False, show_net=False,
        no_unbox=False):
        r"""
        Compute and returns the output layer values when a given pattern is
        fed into the input layer.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``pattern`` -- list or tuple or numpy array of numeric values
          describing the pattern to be fed in input to the multilayer
          perceptron.

        - ``full_state`` -- boolean (default: ``False``, i.e. the output units'
          value is returned) flag setting the result returned: when set to
          ``True`` the function returns a list containing an element for each
          layer, and these elements are in turn lists containing the
          corresponding unit's value; otherwise the reutrned value is a list
          containing the output units' value.

        - ``net`` -- boolean (default: ``False``) value triggering the output
          of net values together with output ones.

        - ``no_unbox`` -- boolean (default: ``False``) value avoiding automatic
          unboxing of the perceptron computation, that is always returning a
          list even if it contains just one element.

        OUTPUT:

        numpy array -- values in the whole perceptron units or in the output
        layer, according to the specified value of ``full_state``.

        EXAMPLES:

        Conbsider the following typical example of a two-layers perceptron able
        to compute the binary XOR function:

            >>> from yaplf.models.neural import MultilayerPerceptron
            >>> dimensions = (2, 2, 1)
            >>> connections = (((1, -1), (-1, 1)), ((1, 1),))
            >>> thr = ((-1, -1), (-1,))
            >>> p = MultilayerPerceptron(dimensions, connections,
            ... thresholds = thr)
            >>> p.compute((1, 0))
            1

        Specification of the ``full_state`` named argument allow the
        computation of the global state of the perceptron, i.e. the set of all
        units' values instead of the sole output ones:

        ::

            >>> p.compute((1, 0), full_state = True)
            [array([1, 0]), array([1])]

        Note that the function returns a number instead of a list when the
        output layer consists of only one unit.

        Different activation functions can be attached to each layer. For
        instance, it is possible to modify the previous network using the
        default activation function in the hidden layer and the sigmoidal one
        in the output layer:

        ::

            >>> from yaplf.utility.activation import \
            ... HeavisideActivationFunction, SigmoidActivationFunction
            >>> q = MultilayerPerceptron(dimensions, connections,
            ... thresholds = thr, activations = (HeavisideActivationFunction(),
            ... SigmoidActivationFunction()))

        Of course this change has an effect on the perceptron performance, for
        the previous structure heavily relied on a Heaviside function
        peculiarity (precisely, h(0) = 0 where h is the Heaviside function):

        ::

        >>> q.compute((1, 0))
        0.5

        Setting the named argument ``show_net`` to ``True`` has the effect of
        returning a couple containing the actual perceptron ouput and the net
        value generating it, that is the value fed into the activation
        function:

        ::

            >>> dimensions = (2, 2, 2)
            >>> connections = ([(1, 1), (1, 1)], [(1, 1), (1, 1)])
            >>> p = MultilayerPerceptron(dimensions, connections)
            >>> p.compute((0, 0))
            array([1, 1])
            >>> p.compute((0, 0), show_net = True)
            array([[1, 2]], [1, 2]])

        Note how the result of a perceptron computation is generally a numpy
        array, apart when there is only one output unit. In this case the
        array containing one element is automatically unboxed into a float.
        This behaviour can be overridden using the ``no_unbox`` named argument:
        when it is set to ``True`` the output will always be an array:

        ::

            >>> q.compute((1, 0), no_unbox = True)
            array([ 0.5])

        The effect of ``full_state`` and ``show_net`` can be combined, in order
        to obtain all the units' output together with the corresponding net
        values:

        ::

            >>> p.compute((0, 0), full_state = True, show_net = True)
            [array([[1, 0], [1, 0]]), array([[1, 2], [1, 2]])]


        In the latter case, the result is a tuple whose first element contains
        all the units' value and whose second one contains the net values.

        AUTHORS:

        - Dario Malchiodi (2010-03-22)

        """

        if type(pattern) != type(array([])):
            curr_layer = array(pattern)

        if full_state:
            perceptron_status = [[]] * (len(self.dimensions) - 1)

        for i in range(len(self.dimensions) - 1):
            if self.has_thresholds:
                augmented_connections = hstack((self.connections[i],
                    to_column(self.thresholds[i])))
                curr_layer = hstack((curr_layer, array((1,))))
            else:
                augmented_connections = self.connections[i]
            act = self.get_activation(i)
            nets = array(dot(augmented_connections, curr_layer))
            curr_layer = array([act.compute(net) for net in nets])
            if full_state:
                if show_net:
                    perceptron_status[i] = transpose((curr_layer, nets))
                else:
                    perceptron_status[i] = curr_layer

        if full_state:
            if show_net:
                return [[(p, 0) for p in pattern]] + perceptron_status
            else:
                return [[pattern]] + perceptron_status

        elif self.dimensions[-1] != 1 or no_unbox:
            if show_net:
                return transpose((curr_layer, nets))
            else:
                return curr_layer
        else:
            if show_net:
                return transpose((curr_layer, nets))[0]
            else:
                return curr_layer[0]

    def decision_function(self, pattern):
        r"""
        Compute the decision function value for the supplied pattern. In a
        multilayer perceptron, this value equals the output units'.

        INPUT:

        - ``self`` -- Perceptron object on which the function is invoked.

        - ``pattern`` -- list/tuple representing a pattern to be fed to the
          input units.

        OUTPUT:

        Number or numpy array of numeric values -- value(s) in the output
        units when the values specified in ``pattern`` are fed as inputs.

        EXAMPLES:

        The ``pattern`` argument should be a list or tuple whose size equals
        the number of input units. The returned value is a number when there
        is only an output unit and a numpy array of numeric values otherwise.

        If no activation function has been specified during object
        initialization, the decision function will take on `0` or `1` values:

        ::

            >>> from yaplf.models.neural import MultilayerPerceptron
            >>> p = MultilayerPerceptron([2, 2, 1], [[(1, -1), (-1, 1)],
            ... [(1, 1)]], thresholds = [(-1, -1), (-1,)])
            >>> p.decision_function((0, 1))
            1
            >>> p.decision_function((1, 1))
            0

        When a specific activation function is used, e.g. the sigmoidal one,
        the decision function values range smoothly between `0` and `1`:

        ::

            >>> from yaplf.utility.activation import SigmoidActivationFunction
            >>> p = MultilayerPerceptron([2, 2, 1], [[(1, -1), (-1, 1)],
            ... [(1, 1)]], thresholds = [(-1, -1), (-1,)],
            ... activations = SigmoidActivationFunction(beta = 0.1))
            >>> p.decision_function((0, 1))
            0.49875415264549638
            >>> p.decision_function((1, 1))
            0.49875104322371477

        The length of ``pattern`` argument should be equal to the number of
        input units, as a ValueError is otherwise thrown:

        ::

            >>> p.decision_function((1, 0, 1))
            Traceback (most recent call last):
                ...
            ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-03-22)

        """

        return self.compute(pattern)

