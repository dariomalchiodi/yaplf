
r"""
Module handling models in yaplf

Module :mod:`yaplf.models` contains all classes handling basic models in yaplf.

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from yaplf.utility import Observable
from yaplf.utility.error import MSE
#from yaplf.graph import PlotterFactory


class Model(Observable):
    r"""
    Base class of every model, where a model is a function in
    output of a learning algorithm, and basically represents a function
    associating values to patterns (the latter intended as component of
    :class:`yaplf.data.Example` objects). The semantic of these values depends
    on the particular kind of model: for instance, they represent predicted
    labels in classifiers and regressors. Each subclass should implement the
    method :meth:`compute`, precisely realizing the above mentioned function.
    The base class itself implements a :meth:`test` method used in order to get
    the approximation error of the model on a given set of examples.

    Note that models are observables. They can be linked to any number of
    specific observers which will be notified when the model state underwent
    a change. This allows for instance the decoupling of a model with the
    corresponding graphic visualizers. 

    EXAMPLES:

    See the examples section for concrete subclasses, such as
    :class:`ConstantModel` or :class:`yaplf.models.svm.SVMClassifier`.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self):
        r"""
        See :class:`Model` for full documentation.

        """

        Observable.__init__(self)

    def compute(self, pattern):
        r"""
        Return the value associated by the model to pattern.

        :param pattern: pattern to be fed to the model.

        :type pattern: iterable

        :returns: output associated by the model to the supplied pattern.

        :rtype: number or iterable of numbers

        EXAMPLES:

        When invoked in the base class the method raises a
        :exc:`NotImplementedError`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError(
            'this class does not implement compute method')

    def test(self, sample, error_model=MSE(), *args, **kwargs):
        r"""
        Test the model against a sample w.r.t. a given error model.


        :param sample: sample to be used for testing the model.

        :type sample: iterable of :class:`yaplf.data.LabeledExample` objects

        :param error_model: criterion to be used in order to measure distances
          between expected and actual output.

        :type error_model: :class:`yaplf.utility.error.ErrorModel`, default:
          class:`yaplf.utility.error.MSE`, corresponding to mean squared error 

        :param verbose: flag to be used in order to get verbose output.

        :type verbose: boolean, default: ``False``

        :returns: overall approximation error

        :rtype: float

        Test the model against a sample. Each example in :obj:`sample` is
        considered and the corresponding pattern is fed to the model. The
        obtained output is then compared to the original example according to
        :obj:`error_model`, which is responsible for computing an overall
        measure of the approximation for sample given by model. The latter
        parameter defaults to the mean squared error.

        EXAMPLES:

        Consider a basic sample whose labels are ``-1``, ``0`` and ``1``, coupled
        with a model outputting the constant value ``0`` regardless of the fed
        pattern:

        >>> from yaplf.models import ConstantModel # Constant output
        >>> from yaplf.data import LabeledExample
        >>> sample = (LabeledExample((-1,), (-1,)), \
        ... LabeledExample((0,), (0,)), LabeledExample((1,), (1,)))
        >>> model = ConstantModel(0)

        Now, the model will score null error on the second example and unit
        error on the remaining ones, thus the mean square error for the whole
        example set will be :math:`\frac{2}{3}`:

        >>> model.test(sample)
        0.66666666666666663

        Different error models refer to different metrics to be used in order
        to measure errors between expected and actual model outputs. For
        instance, using :class:`yaplf.utility.error.MaxError` as error model
        will yield the maximum error throughout the sample, that is 1:

        >>> from yaplf.utility.error import MaxError
        >>> model.test(sample, MaxError())
        1

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return error_model.compute(sample, self, *args, **kwargs)


class Classifier(Model):
    r"""
    Base class for classifier models. A classifier associates patterns to
    classes (former and latter identify as the components of
    :class:`yaplf.data.LabeledExample` objects.) Each subclass shoult implement
    the method :meth:`decision_function` outputting a (possibly) float value to
    be subsequently transformed into the true output of the classifier through
    the :meth:`compute` method, inherited from :class:`python.models.Model` and
    to be implemented in the subclassess, too.

    EXAMPLES:

    See the examples section for concrete subclasses, such as
    :class:`ConstantModel` or
    :class:`yaplf.models.svm.classification.SVMClassifier`.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self):
        r"""
        See :class:`Classifier` for full documentation.
        """

        Model.__init__(self)

    def decision_function(self, pattern):
        r"""
        Method computing the decision function for a given pattern

        :param pattern: pattern to be fed to the model.

        :type pattern: iterable

        :returns: decision function value associated to the supplied pattern.

        :rtype: float

        EXAMPLES:

        When invoked in the base class the method raises a
        :exc:`NotImplementedError`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError('Method decision_function not implemented \
        in base class Classifier.')

#    def plot(self, *args, **kwargs):
#        r"""
#        Returns a figure containing the classifier decision
#        function plot, either as a bi- or three-dimensional plot according
#        to the following invocation syntax:
#
#        - ``plot(x_range, y_range)`` returns a 2D plot.
#
#        - ``plot(x_range, y_range, z_range)`` returns a 3D plot.
#
#        INPUT:
#
#        - ``self`` -- Classifier object executing the method.
#
#        - ``x_range`` -- bidimensional iterable containing the range of x
#          variable.
#
#        - ``y_range`` -- bidimensional iterable containing the range of y
#          variable.
#
#        - ``z_range`` -- bidimensional iterable containing the range of z
#          variable.
#
#        - ``x_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
#          samples in x range.
#
#        - ``y_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
#          samples in y range.
#
#        - ``z_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
#          samples in z range.
#
#        - ``output`` -- integer (default: unused) output to be selected when
#          drawing the decision function if the model has several outputs, not
#          present when the model has a unique output.
#
#        - ``base`` -- matplotlib figure (default: a new figure) figure to be
#          used to draw the decision function.
#
#        - ``gradient`` -- boolean (default: False) flag setting the
#          visualization of a colored gradient.
#
#        - ``gradient_color`` -- colormap (default: Greys) color map to be used
#          for colored gradient visualization.
#
#        - ``color_bar`` -- boolean (default: False) flag setting the
#          visualization of a color legend.
#
#        - ``contours``-- iterable of numeric values (default: see concrete
#          implementation in subclasses) values corresponding to the contours
#          to be drawn.
#
#        - ``contour_color`` -- iterable of colors or single color value
#          (default: 'gray') colors of the drawn contours; if a single value is
#          supplied, it refers to all contours.
#
#        - ``contour_width`` -- iterable of numberic values or single numeric
#          value (default: 1) width of the drawn contours; if a single numeric
#          value is supplied, it refers to all contours.
#
#        - ``contour_style`` -- iterable or single value of a valid style
#          (default: '-') style of the drawn contours; if a single value is
#          supplied, it refers to all contours.
#
#        - ``plotter`` -- Plotter object (default: SagePlotter() when the code
#          is run within sage and MatplotlibPlotter() otherwise) to be used in
#          order to render graphics.
#
#        EXAMPLES:
#
#        See the examples section for concrete subclasses, such as
#        SVMClassifier in package yaplf.models.svm.
#
#        AUTHORS:
#
#        - Dario Malchiodi (2010-02-22)
#
#        """
#
#        try:
#            output_unit = kwargs['output']
#
#            def classify(*args):
#                """Classification function created on-the-fly."""
#
#                return self.decision_function(args)[output_unit]
#        except KeyError:
#
#            def classify(*args):
#                """Classification function created on-the-fly."""
#
#                return self.decision_function(args)
#
#        try:
#            plotter = kwargs['plotter']
#            del kwargs['plotter']
#        except KeyError:
#            plotter = PlotterFactory.get_plotter()
#
#        return plotter.decision_function_plot(args, classify, **kwargs)


class ConstantModel(Classifier):
    r"""
    Dummy model outputting a constant value regardless of the fed input.

    :param value: object constantly in output of the model.

    EXAMPLES:

    A :class:`ConstantModel` instance outputs the same value regardless the fed
    pattern, being this a number, a tuple or a generic object, either fixed
    or randomly drawn:

    >>> from yaplf.models import ConstantModel
    >>> model = ConstantModel(0)
    >>> model.compute(1)
    0
    >>> model.compute((1,3))
    0
    >>> model.compute("string")
    0
    >>> from numpy import random
    >>> model.compute(random.normal())
    0

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, value):
        r"""
        See :class:`ConstantModel` for full documentation.
        """

        Classifier.__init__(self)
        self.value = value
        self.notify_observers()

    def __repr__(self):
        return 'ConstantModel(' + str(self.value) + ')'

    def __str___(self):
        return self.__repr__()

    def __eq__(self, other):
        if type(self) == type(other):
            return self.value == other.value
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.value)

    def __nonzero__(self):
        return self.value

    def compute(self, pattern):
        r"""
        Output the model constant value regardless of the supplied pattern.

        :param pattern: pattern to be fed to the model.

        :type pattern: iterable

        :returns: value specified when building the :class:`ConstantModel`
          instance


        EXAMPLES:

        A :class:`ConstantModel` instance outputs the same value regardless of
        the fed pattern, being this a number, a tuple or a generic object,
        either fixed or randomly drawn:

        >>> from yaplf.models import ConstantModel
        >>> model = ConstantModel(0)
        >>> model.compute(1)
        0
        >>> model.compute((1,3))
        0
        >>> model.compute("string")
        0
        >>> from numpy import random
        >>> model.compute(random.normal())
        0

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return self.value

    def decision_function(self, pattern):
        r"""
        Output the model constant value regardless of the supplied pattern.

        :param pattern: pattern to be fed to the model.

        :type pattern: iterable

        :returns: value specified when building the :class:`ConstantModel`
          instance

        EXAMPLES:

        A :class:`ConstantModel` instance outputs as decision function the same
        value regardless of the fed pattern, being this a number, a tuple or a
        generic object, either fixed or randomly drawn:

        >>> from yaplf.models import ConstantModel
        >>> model = ConstantModel(0)
        >>> model.decision_function(1)
        0
        >>> model.decision_function((1,3))
        0
        >>> model.decision_function("string")
        0
        >>> from numpy import random
        >>> model.decision_function(random.normal())
        0

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return self.compute(pattern)
