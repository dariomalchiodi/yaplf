
r"""
Package handling learning trajectories in yaplf

Package yaplf.graph.trajectory contains learning trajectories in yaplf.

- pep8 checked
- pylint score: 8.87

AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  yaplf.utility, containing base class, weight- and error-based trajectories.

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


from yaplf.utility import Observer
from yaplf.graph import PlotterFactory
from yaplf.utility.error import MSE


class GenericTrajectory(Observer):
    r"""
    Generic observable for iterative learning algorithms, repeatedly
    collecting a given value and storing it in a list. All subclasses should
    implement a ``get_value`` function returning the current value to be
    added to the trajectory.

    INPUT:

    - ``self`` -- ``GenericTrajectory`` object on which the function is
      invoked.

    - ``learning_algorithm`` -- ``LearningAlgorithm`` object whose state is
      to be monitored.

    - ``num_steps`` -- integer (default: 1) number of learning iterations to
      be executed before saving the algorithm status.

    OUTPUT:

    No output.

    EXAMPLES:

    See the examples for specific subclasses such as
    ``PerceptronWeightTrajectory`` in this package.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, learning_algorithm, num_steps=1):
        r"""
        See ``GenericTrajectory`` for full documentation.
        """

        Observer.__init__(self, learning_algorithm)
        self.num_steps = num_steps
        self.trajectory_time = []
        self.trajectory_value = []
        self.num_invocations = 0

    def reset(self):
        r"""Resets the trajectory.

        INPUT:

        - ``self`` -- ``GenericTrajectory`` object on which the function is
        invoked.

        OUTPUT:

        No output.

        EXAMPLES:

        See the examples for specific subclasses such as
        ``PerceptronWeightTrajectory`` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.trajectory_time = []
        self.trajectory_value = []
        self.num_invocations = 0

    def update(self, observable):
        r"""
        Add a couple (time, value) to a trajectory showing how an observable
        state changes through time. The couple is actually added only each
        ``self.num_steps`` invocations.

        INPUT:

        - ``self`` -- ``Observer`` object on which the function is invoked.

        - ``observable`` -- ``Observable`` whose status has changed.

        OUTPUT:

        No output.

        EXAMPLES:

        See the examples for specific subclasses such as
        ``PerceptronWeightTrajectory`` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.num_invocations += 1
        if self.num_invocations % self.num_steps == 0:
            self.trajectory_time.append(self.num_invocations)
            self.trajectory_value.append(self.get_value(observable))

    def get_value(self, observable):
        r"""
        Returns the currently computed trajectory.

        INPUT:

        - ``self`` -- ``Observer`` object on which the function is invoked.

        - ``observable`` -- ``Observable`` whose status has changed.

        OUTPUT:

        No output.

        EXAMPLES:

        See the examples for specific subclasses such as
        ``PerceptronWeightTrajectory`` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        pass


class PerceptronWeightTrajectory(GenericTrajectory):
    r"""
    Observable for perceptron-based iterative learning algorithms
    collecting a perceptron output unit's weights during the learning phase.

    INPUT:

    - ``self`` -- ``PerceptronWeightTrajectory`` object on which the function
      is invoked.

    - ``learning_algorithm`` -- ``LearningAlgorithm`` object whose state is
      to be monitored.

    - ``num_steps`` -- integer (default: 1) number of learning iterations to
      be executed before saving the algorithm status.

    - ``output`` -- integer (default: 0) output unit whose weights should be
      monitored.

    - ``mask`` -- list or tuple of integers (default: None, meaning that all
      weights should be considered) corresponding to the positions of weights
      which should be included in the trajectory (so that also perceptron
      having a high number of inputs can be monitored using a 2D or a 3D
      projection of their weights).

    OUTPUT:

    ``PerceptronWeightTrajectory`` object

    EXAMPLES:

    Consider a full training set for the binary AND function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1., 1.), (1,)),
        ... LabeledExample((0., 0.), (0,)), LabeledExample((0, 1), (0,)),
        ... LabeledExample((1, 0), (0,))]

    In order to get a perceptron computing the above function, the first
    thing to do is creating an instance of a suitable learning algorithm,
    such as the one learning through gradient-descent optimization:

    ::

        >>> from yaplf.algorithms.neural import GradientPerceptronAlgorithm
        >>> alg = GradientPerceptronAlgorithm(and_sample, threshold = True,
        ... weight_bound = 0.1, beta = 0.8)

    Before running the algorithm it is possible to tie it to a
    ``PerceptronWeightTrajectory`` which will observe the algorithm during its
    search for the best perceptron, saving at each iteration the current
    model's weights (three in total as we have two inputs and a threshold):

    ::

        >>> from yaplf.graph.trajectory import PerceptronWeightTrajectory
        >>> weightObs = PerceptronWeightTrajectory(alg)

    The observer instantiation insures that each time the learning algorithm
    will perform an iteration, its state will automatically saved into the
    ``PerceptronWeightTrajectory``'s one, thus one can run the algorithm and
    subsequently call the observer's ``get_trajectory`` function, returning
    a graphic of the weigth trajectory during training:

    ::

        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(5000), batch = False,
        ... learning_rate = .1)
        >>> weightObs.get_trajectory(joined = True)

    Note how ``get_trajectory`` accepts a ``color`` and a ``joined``
    arguments, fowarded to the plotter actually rendering the graphic.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, learning_algorithm, *args, **kwargs):
        r"""
        See ``PerceptronWeightTrajectory`` for full documentation.

        """

        GenericTrajectory.__init__(self, learning_algorithm, *args)
        try:
            self.output = kwargs['output']
        except KeyError:
            self.output = 0

        try:
            self.mask = kwargs['mask']
        except KeyError:
            self.mask = None

    def get_value(self, observable):
        r"""
        Returns a trajectory element, that is the linked perceptron weights
        at the current learning iteration.

        - ``self`` -- ``PerceptronWeightTrajectory`` on which the function is
          invoked.

        - ``observable`` -- ``LearningAlgorithm`` whose current model is to
          be browsed in order to get its weights.

        OUTPUT:

        numpy array -- weights of the perceptron currently inferred by the
        learning algorithm.

        EXAMPLES:

        This function is automatically called by ``sync_state``.

        """

        num_weights = len(observable.model.weights[self.output])
        if not observable.threshold:
            num_weights -= 1

        if self.mask is None:
            return observable.model.weights[self.output][:num_weights]
        else:
            return [observable.model.weights[self.output][:num_weights][ind]
                for ind in self.mask]

    def get_trajectory(self, **kwargs):
        r"""
        Returns the currently computed trajectory.

        INPUT:

        - ``self`` -- ``Observer`` object on which the function is invoked.

        - ``observable`` -- ``Observable`` whose status has changed.

        - ``plotter`` -- ``Plotter`` object (default: automatically detects a
          running sage or matplotlib environment) to be used in order to
          render graphics.

        The function forwards other options affecting the graphics style to
        the ``list_plot`` function of the specified or detected plotter.

        OUTPUT:

        No output.

        EXAMPLES:

        See ``PerceptronWeightTrajectory``.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            plotter = kwargs['plotter']
            del kwargs['plotter']
        except KeyError:
            #if detect_sage():
            #    plotter = SagePlotter()
            #else:
            #    plotter = MatplotlibPlotter()
            plotter = PlotterFactory.get_plotter()

        return plotter.list_plot(self.trajectory_value, **kwargs)


class ErrorTrajectory(GenericTrajectory):
    r"""
    Observable for iterative learning algorithms collecting an error
    measure value during the learning phase.

    INPUT:

    - ``self`` -- ``ErrorTrajectory`` object on which the function is
      invoked.

    - ``learning_algorithm`` -- ``LearningAlgorithm`` object whose state is
      to be monitored.

    - ``num_steps`` -- integer (default: 1) number of learning iterations to
      be executed before saving the algorithm status.

    - ``test_set`` -- list/tuple of ``Example`` (default: the learning
    algorithm's test set defined in its stopping criterion, if it exists,
    otherwise the learning algorithm's train set) sample on which the error
    should be computed.

    - ``error_measure`` -- ``ErrorModel`` (default: MSE()) to be used in order
      to compute errors.

    OUTPUT:

    ``ErrorTrajectory`` object

    EXAMPLES:

    Consider a full training set for the binary AND function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1., 1.), (1,)),
        ... LabeledExample((0., 0.), (0,)), LabeledExample((0, 1), (0,)),
        ... LabeledExample((1, 0), (0,))]

    In order to get a perceptron computing the above function, the first
    thing to do is creating an instance of a suitable learning algorithm,
    such as the one learning through gradient-descent optimization:

    ::

        >>> from yaplf.algorithms.neural import GradientPerceptronAlgorithm
        >>> alg = GradientPerceptronAlgorithm(and_sample, threshold = True,
        ... weight_bound = 0.1, beta = 0.8)

    Before running the algorithm it is possible to tie it to a
    ``ErrorTrajectory`` which will observe the algorithm during its
    search for the best perceptron, computing and saving at each iteration the
    error with which the currently inferred model approximates the examples
    in ``and_sample``:

    ::

        >>> from yaplf.graph.trajectory import ErrorTrajectory
        >>> errObs = ErrorTrajectory(alg)

    In this way one can run the algorithm and
    subsequently call the observer's ``get_trajectory`` function, returning
    a graphic of the model approximation versus learning iterations:

    ::

        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(5000), batch = False,
        ... learning_rate = .1)
        >>> errObs.get_trajectory(color='red', joined = True)

    Note how ``get_trajectory`` accepts a ``color`` and a ``joined``
    arguments, fowarded to the plotter actually rendering the graphic.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, learning_algorithm, *args, **kwargs):
        r"""
        See ``ErrorTrajectory`` for full documentation.

        """

        GenericTrajectory.__init__(self, learning_algorithm, *args)
        try:
            self.test_set = kwargs['test_set']
        except KeyError:
            try:
                self.test_set = learning_algorithm.stop_criterion.test_set
            except AttributeError:
                self.test_set = learning_algorithm.sample

        try:
            self.error_measure = kwargs['error_measure']
        except KeyError:
            try:
                self.error_measure = \
                    learning_algorithm.stop_criterion.error_measure
            except AttributeError:
                self.error_measure = MSE()

    def get_value(self, observable):
        r"""
        Returns a trajectory element, that is the linked model approximation
        of a given example set at the current learning iteration.

        - ``self`` -- ``ErrorTrajectory`` on which the function is
          invoked.

        - ``observable`` -- ``LearningAlgorithm`` whose current model is to
          be browsed in order to get its model approximation.

        OUTPUT:

        float -- approximation capability of the currently inferred model.

        EXAMPLES:

        This function is automatically called by ``sync_state``.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """
        return observable.model.test(self.test_set, self.error_measure)

    def get_trajectory(self, **kwargs):
        r"""
        Returns the currently computed trajectory.

        INPUT:

        - ``self`` -- ``Observer`` object on which the function is invoked.

        - ``observable`` -- ``Observable`` whose status has changed.

        - ``plotter`` -- ``Plotter`` object (default: automatically detects a
          running sage or matplotlib environment) to be used in order to
          render graphics.

        The function forwards other options affecting the graphics style to
        the ``list_plot`` function of the specified or detected plotter.

        OUTPUT:

        No output.

        EXAMPLES:

        See ``ErrorTrajectory``.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)
        """

        try:
            plotter = kwargs['plotter']
            del kwargs['plotter']
        except KeyError:
            #if detect_sage():
            #    plotter = SagePlotter()
            #else:
            #    plotter = MatplotlibPlotter()
            plotter = PlotterFactory.get_plotter()

        return plotter.list_plot(zip(self.trajectory_time, \
            self.trajectory_value), **kwargs)
