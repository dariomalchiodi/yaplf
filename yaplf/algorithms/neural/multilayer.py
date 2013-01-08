
r"""
Package handling multilayer perceptron learning algorithms in yaplf.

Package yaplf.algorithms.neural.multilayer contains all the classes handling
multilayer perceptrons learning algorithms in yaplf.



AUTHORS:

- Dario Malchiodi (2011-01-94): initial version factored out from
  yaplf.algorithms.neural

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


from numpy import random, array, transpose, dot, outer, zeros

from yaplf.utility.activation import SigmoidActivationFunction
from yaplf.models.neural.multilayer import MultilayerPerceptron
from yaplf.algorithms import LearningAlgorithm, IterativeAlgorithm


class BackpropagationAlgorithm(IterativeAlgorithm):
    r"""Backpropagation algorithm for multilayer perceptrons.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``LabeledExample`` containing the sample
      to be learnt.

    - ``dimensions`` -- list or tuple of numerical values describing the number
      of layers and the number of units therein, including the input layer.

    - ``threshold`` -- boolean (default: ``True``) flag setting the use of
      thresholded perceptrons.

    - ``activations`` -- ActivationFunction or list/tuple of ActivationFunction
      (default: SigmoidActivationFunction()) activation functions to be used
      for the perceptron units. When a single value is specified, it applies to
      all units. When a list/tuple is specified, each element corresponds to
      all units in a perceptron layer.

    - ``weight_bound`` -- float (default: 0.1) upper bound of the interval in
      which the initial weights and thresholds are chosen uniformly at random
      (the lower bound of this interval is ``-1 * weight_bound``). A
      ``ValueError`` is thrown if this parameter is not positive.


    OUTPUT:

    ``BackpropagationAlgorithm`` object.

    EXAMPLES:

    Consider the following data set summarizing the binary XOR function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> xor_sample = [LabeledExample((0, 0), (0,)),
        ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)),
        ... LabeledExample((1, 1), (0,))]

    This is a paradigmatical example of non-linearly separable data set which
    needs a richer model than a perceptron in order to be learnt:

    ::

        >>> from yaplf.utility.activation import SigmoidActivationFunction
        >>> from yaplf.algorithms.neural.multilayer import BackpropagationAlgorithm
        >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
        ... threshold = True, activations = SigmoidActivationFunction(10))

    In order to actually run the algorithm it is necessary to specify a
    stopping criterion (the default behaviour would only execute a learning
    iteration, probably not going so far). In order to keep it simple, one can
    chose ``FixedIterationsStoppingCriterion`` so as to run a fixed number of
    iterations, say `5000`:

    ::

        >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(5000), learning_rate = .1)

    The inferred model can be inspected through the ``model`` field in the
    ``LearningAlgorithm`` object:

    ::

        >>> alg.model # random
        MultilayerPerceptron((2, 2, 1), [array([[-0.49748418, -0.48592928],
        [-0.72052151, -0.69609958]]), array([[ 0.78258238, -0.84286672]])],
        thresholds = [array([ 0.71869556,  0.24337534]), array([-0.36203957])],
        activations = SigmoidActivationFunction(10))

    Note that the algorithm can be run in different flavours, described in the
    documentation of ``run``.

    One of the ways of assessing the performance of the algorithm is that of
    invoking the ``test`` function inherited from the ``Model`` class in order
    to see how each example has been classified:

    ::

        >>> alg.model.test(xor_sample, verbose = True) # random
        (0, 0) mapped to 0.0279358676426, label is (0,), error [ 0.00078041]
        (0, 1) mapped to 0.968320708961, label is (1,), error [ 0.00100358]
        (1, 0) mapped to 0.966511649371, label is (1,), error [ 0.00112147]
        (1, 1) mapped to 0.0429967333857, label is (0,), error [ 0.00184872]
        MSE 0.00118854472284
        0.0011885447228408164

    The named argument ``verbose`` activates the verbose output detailing how
    error spreads on each example. Note that the output of ``test`` is likely
    to be different on each run, as when the class constructor is called the
    initial weights are chosen at random. It is also possible that the
    performance is not satisfactory for some examples are not learnt at all.
    This can be an effect of many factors. For instance:

    - learning has not converged, and more iterations are needed; thus other
      ``run`` invocations should be performed, suitably chosing the function
      parameters;

    - despite the chosen multilayer perceptron architecture can learn the data
      set, learning converged to an unsatisfactory model because of the initial
      values randomly picked; thus learning should be restarted, either
      invoking ``reset`` on the ``BackpropagationAlgorithm`` object or creating
      a new instance of this class;

    - the chosen architecture cannot learn the data set, thus the whole process
      should be repeated modifying the initially chosen multilayer perceptron
      architecture, for instance chosing a different number of layers or a
      different number of units in layers.

    Another way of evaluating the inferred model is in this case that of
    graphically visualizing it. As the perceptron has two inputs it is indeed
    possible to call its ``plot`` method specifying a region containing all
    the pattern supplied to the learning algorithm:

    ::

        >>> alg.model.plot((0, 1), (0, 1), shading = True)

    Learning can be monitored using the class ``ErrorTrajectory`` in package
    ``yaplf.graph.trajectory`` as follows:

   ::

        >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
        ... activations = SigmoidActivationFunction(10))
        >>> errObs = ErrorTrajectory(alg)
        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(1500))
        >>> errObs.get_trajectory(color='red', joined = True)

    In this way, at each learning iteration the inferred model is tested
    against the training set, so that the ``get_trajectory`` function returns
    a graph of the related error versus the iteration number.

    As a final remark, it should be highlighted that these examples are shown
    for illustrative purpose. The suitable way of assessing a learnt model
    performance involves more complex techniques involving for instance the use
    of:

    - a test set in order to assess when learning should stop, using a
      different stopping criterion such as ``TestSetStoppingCriterion``;

    - a test set in order to graphically inspect how error changes as learning
      proceeds, specifying different arguments to the constructor of
      ``ErrorTrajectory``;

    - a cross validation procedure (see function ``cross_validation`` in
      package ``yaplf.utility.validation``) in order to choose the best
      perceptron architecture.

    Concerning last point, the cross validation procedure can select among a
    given set of choices the one minimizing the inferred error. More precisely,
    consider the next code snippet:

    ::

        >>> tc_sample = (LabeledExample((0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9,
        ... 0.9, 0.9), (0.1,)), LabeledExample((0.9, 0.9, 0.9,  0.1, 0.9, 0.1,
        ...  0.1, 0.9, 0.1), (0.9,)), LabeledExample((0.9, 0.9, 0.9, 0.9, 0.1,
        ... 0.9, 0.9, 0.1, 0.9), (0.1,)), LabeledExample((0.1, 0.1, 0.9, 0.9,
        ... 0.9, 0.9, 0.1, 0.1, 0.9), (0.9,)), LabeledExample((0.9, 0.9, 0.9,
        ... 0.1, 0.1, 0.9, 0.9, 0.9, 0.9), (0.1,)), LabeledExample((0.1, 0.9,
        ... 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9), (0.9,)), LabeledExample((0.9,
        ... 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9), (0.1,)),
        ... LabeledExample((0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1),
        ... (0.9,)))
        >>> from yaplf.utility.validation import cross_validation
        >>> p = cross_validation(BackpropagationAlgorithm, tc_sample,
        ... {'activations': (SigmoidActivationFunction(2),
        ... SigmoidActivationFunction(3), SigmoidActivationFunction(5),
        ... SigmoidActivationFunction(10))},
        ... fixed_parameters = {'dimensions': (9, 2, 1)},
        ... run_parameters = {'stopping_criterion': \
        ... FixedIterationsStoppingCriterion(1000), 'learning_rate': 1},
        ... num_folds = 4, verbose = True)
        Errors: [0.030319163781913711, 0.060571552036313515,
        0.029535618302241346, 0.4099711142423762]
        Minimum error in position 2
        Selected parameters (SigmoidActivationFunction(5),)

    Its effect is that of training a multilayer perceptron through the
    backpropagation algorithm starting by the sample in ``tc_sample``, as
    specified by the first two arguments and analyzing four different
    activation functions (third argument, note that in this case it consists
    of a singleton tuple and more generally can be a tuple involving different
    named argument of ``BackpropagationAlgorithm`` or other learning
    algorithms.) For each choice of the activation function, learning evolves
    as follows:

    - the sample is partitioned in 4 subsets, as specified by the ``num_folds``
      named argument;

    - the learning algorithm is instantiated excluding from the sample the
      first subset in previous point, and using as named argument those
      specified in ``fixed_parameters``, joined with the one specifying the
      selected activation function; subsequently, the algorithm is run using
      the named arguments in ``run_parameters``;

    - the inferred multilayer perceptron is tested on the originally excluded
      subset of examples;

    - the whole process is repeated starting from the second subset, and so on
      till the fourth; the test errors are then averaged and associated to the
      initially chosen activation function.

    Finally, the activation function yielding the minimum averaged test error
    is selected and used in order to infer a new multilayer perceptron starting
    from all examples.

    AUTHORS:

    - Dario Malchiodi (2010-03-31)

    """

    def __init__(self, sample, dimensions=None, **kwargs):
        r"""
        See ``BackpropagationAlgorithm`` for full documentation.

        """

        IterativeAlgorithm.__init__(self, sample)

        # dimensions needs to be a named argument in order to be able to
        # cross-validate on it. The default value will correspond to three
        # layers, with the input and output one automatically sized in order
        # to fit the provided data set, and the hidden one containing half of
        # the biggest value between number of input and output units.

        if dimensions is not None:
            self.dimensions = dimensions
        else:
            n_in = len(sample[0].pattern)
            n_out = len(sample[0].label)
            self.dimensions = (n_in, int(max(n_in, n_out) / 2), n_out)

        num_input = self.dimensions[0]
        for example in sample:
            if len(example.pattern) != num_input:
                raise ValueError('Sample incompatible with number of units')

        try:
            self.activations = kwargs['activations']
            if len(self.activations) != len(dimensions):
                raise ValueError(\
                    'Activations incompatible with number of layers')
        except KeyError:
            self.activations = SigmoidActivationFunction()
        except TypeError:
            pass
            # Raised by len if the argument is assigned a single activation
        # this calms down pylint
        self.threshold = None

        self.reset(**kwargs)

    def reset(self, **kwargs):
        r"""
        Reset weights and thresholds of the inferred MultilayerPerceptron
        picking values at random.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``threshold`` -- boolean (default: ``True``) flag setting the use of
          thresholded perceptrons.

        - ``weight_bound`` -- float (default: 0.1) upper bound of the interval
          in which the initial weights and thresholds are chosen uniformly at
          random (the lower bound of this interval is ``-1 * weight_bound``). A
          ``ValueError`` is thrown if this parameter is not positive.

        OUTPUT:

        No output. After the invocation the initialized model is available
        through the ``model`` field, in form of a ``MultilayerPerceptron``
        instance.

        EXAMPLES:

        Consider the following data set summarizing the binary XOR function:

        ::

            >>> from yaplf.data import LabeledExample
            >>> xor_sample = [LabeledExample((0, 0), (0,)),
            ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)),
            ... LabeledExample((1, 1), (0,))]

        This is a paradigmatical example of non-linearly separable data set
        which needs a richer model than a perceptron in order to be learnt:

        ::

            >>> from yaplf.utility.validation import SigmoidActivationFunction
            >>> from yaplf.algorithms.neural import BackpropagationAlgorithm
            >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
            ... threshold = True, activations = SigmoidActivationFunction(10))

        Suppose the algorithm is run for, say, 1000 iterations with learning
        rate set to `0.1` with the following results:

    ::

            >>> from yaplf.utility.stopping import \
            ... FixedIterationsStoppingCriterion
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(1000), learning_rate = .1)
            >>> alg.model.test(xor_sample, verbose = True) # random
            (0.100000000000000, 0.100000000000000) mapped to 0.107281883481,
            label is (0.100000000000000,), error [  5.30258270e-05]
            (0.100000000000000, 0.900000000000000) mapped to 0.889216991161,
            label is (0.900000000000000,), error [ 0.00011627]
            (0.900000000000000, 0.100000000000000) mapped to 0.350484814876,
            label is (0.900000000000000,), error [ 0.30196694]
            (0.900000000000000, 0.900000000000000) mapped to 0.353978869018,
            label is (0.100000000000000,), error [ 0.06450527]
            MSE 0.0916603759241
            0.0916603759241

    It is clear that the algorithm has learnt only three examples out of four.
    In order to check whether or not the algorithm converged, one can continue
    its execution for, say, another thousand iterations:

    ::

        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(1000))
        >>> alg.model.test(xor_sample) # random
        0.0916130351364

    As the test error is essentially unchanged we can conclude that learning
    converged to a local minima of the error function. In order to fresh start
    another session, hoping that the random initialization can overcome this
    problem, one can get a new instance of ``BackpropagationAlgorithm``, or
    call ``reset`` on the alrerady available instance:

    ::

        >>> alg.reset()
        >>> alg.run(stopping_criterion =\
        ... FixedIterationsStoppingCriterion(1000))
        >>> alg.model.test(xor_sample) # random
        1.01130579975e-05

        AUTHORS:

        - Dario Malchiodi (2010-03-31)

        """

        try:
            self.threshold = kwargs['threshold']
        except KeyError:
            self.threshold = True

        try:
            # picks initial weights and threshold uniformly
            # between -weight_bound and weight_bound
            init_bound = kwargs['weight_bound']
            if init_bound <= 0:
                raise ValueError(
                    'The weight_bound parameter should be positive')
        except KeyError:
            init_bound = 0.1

        dims = transpose((self.dimensions[1:], self.dimensions[:-1]))
        connections = [random.uniform(-1 * init_bound, init_bound, shape)
            for shape in dims]

        perc_kwargs = {'activations': self.activations}
        if self.threshold:
            thr = [random.uniform(-1 * init_bound, init_bound, shape) \
                for shape in self.dimensions[1:]]
            perc_kwargs['thresholds'] = thr

        self.model = MultilayerPerceptron(self.dimensions, connections,
             **perc_kwargs)

    def run(self, **kwargs):
        r"""
        Run the learning algorithm.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``stopping_criterion`` -- ``StoppingCriterion`` instance (default:
          ``FixedIterationsStoppingCriterion()``, amounting to the execution of
          one learning step) describing the criterion to be fulfilled in order
          to stop the training phase.

        - ``batch`` -- boolean (default: ``False``, amounting to online
          learning mode) flag setting batch learning, i.e. model update after
          the presentation of all examples, instead of online learning, i.e.
          model update at each example presentation.

        - ``selector`` -- iterator (default: ``sequential_selector``, amounting
          to cycling through the available examples) selecting the next sample
          to be fed to the learnin algorithm.

        - ``learning_rate`` -- float (default: 0.1) value to be used as
          learning rate.

        - ``momentum_term`` -- float (default: 0) value tu be used as momentum
          term.

        - ``min_error`` -- float (default: 0, which means connections
          and thresholds will always be updated) error value under which no
          update will occur on connections and thresholds.

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``Perceptron`` instance.

        EXAMPLES:

        Consider the following data set summarizing the binary XOR function,
        and a ``BackpropagationAlgorithm`` instance for it:

        ::

            >>> from yaplf.data import LabeledExample
            >>> xor_sample = [LabeledExample((0, 0), (0,)),
            ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)),
            ... LabeledExample((1, 1), (0,))]
            >>> from yaplf.utility import SigmoidActivationFunction
            >>> from yaplf.algorithms.neural import BackpropagationAlgorithm
            >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
            ... threshold = True, activations = SigmoidActivationFunction(10))

        In order to actually run the algorithm it is necessary to specify a
        stopping criterion (the default behaviour would only execute a learning
        iteration, probably not going so far). In order to keep it simple, one
        can chose ``FixedIterationsStoppingCriterion`` so as to run a fixed
        number of iterations, say `5000`:

        ::

            >>> from yaplf.utility.stopping import \
            ... FixedIterationsStoppingCriterion
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(5000), learning_rate = .1)

        The inferred model can be inspected through the ``model`` field in the
        ``LearningAlgorithm`` object:

        ::

            >>> alg.model # random
            MultilayerPerceptron((2, 2, 1), [array([[-0.49748418, -0.48592928],
            [-0.72052151, -0.69609958]]), array([[ 0.78258238, -0.84286672]])],
            thresholds = [array([ 0.71869556,  0.24337534]),
            array([-0.36203957])], activations = SigmoidActivationFunction(10))

        One of the ways of assessing the performance of the algorithm is that
        of invoking the ``test`` function inherited from the ``Model`` class in
        order to see how each example has been classified:

        ::

            >>> alg.model.test(xor_sample, verbose = True) # random
            (0, 0) mapped to 0.0279358676426, label is (0,), error
            [ 0.00078041]
            (0, 1) mapped to 0.968320708961, label is (1,), error [ 0.00100358]
            (1, 0) mapped to 0.966511649371, label is (1,), error [ 0.00112147]
            (1, 1) mapped to 0.0429967333857, label is (0,), error
            [ 0.00184872]
            MSE 0.00118854472284
            0.0011885447228408164

        The named argument ``verbose`` activates the verbose output detailing
        how error spreads on each example. Note that the output of ``test`` is
        likely to be different on each run, as when the class constructor is
        called the initial weights are chosen at random.

        Another solution is that of running the algorithm until the error on
        its training set is below a given threshold. This can be easily
        attained using ``TrainErrorStoppingCriterion``:

        ::

            >>> from yaplf.utility.stopping import TrainErrorStoppingCriterion
            >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
            ... learning_rate = .1,
            ... activations = SigmoidActivationFunction(10))
            >>> alg.run(stopping_criterion = TrainErrorStoppingCriterion(0.01))

        The argument in the constructor of ``TrainErrorStoppingCriterion`` sets
        the above mentioned threshold. It is also possible (and, besides, more
        correct) to run the learning algorithm using a training set and
        stopping the process when the test error on another data set goes
        below a given threshold. This can be attained using
        ``TestErrorStoppingCriterion`` in package ``yaplf.utility.stopping``.

        Another way of evaluating the inferred model is in this case that of
        graphically visualizing it. As the perceptron has two inputs it is
        indeed possible to call its ``plot`` method specifying a region
        containing all the pattern supplied to the learning algorithm:

        ::

            >>> alg.model.plot((0, 1), (0, 1), shading = True)

        Learning can be also monitored using the class ``ErrorTrajectory`` in
        package ``yaplf.graph.trajectory`` as follows:

       ::

            >>> alg = BackpropagationAlgorithm(xor_sample, (2, 2, 1),
            ... activations = SigmoidActivationFunction(10))
            >>> errObs = ErrorTrajectory(alg)
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(1500))
            >>> errObs.get_trajectory(color='red', joined = True)

        In this way, at each learning iteration the inferred model is tested
        against the training set, so that the ``get_trajectory`` function
        returns a graph of the related error versus the iteration number.

        The ``run`` function has a number of named argument allowing to
        tune how the backpropagation algorithm selects its output, and whose
        meaning requires a bit more information about how the algorithm works:
        basically, during the initialization of ``BackpropagationAlgorithm``
        and at each invokation of ``reset`` a multilayer perceptron is created
        picking its connection weights and its threshold values at random;
        subsequently at each iteration an example is selected and fed to this
        perceptron. The obtained output is compared to the expected one and
        an error is computed. This error, together with other information,
        the computation of a quantity `\Delta w` which will in turn be used
        in order to modify connections and thresholds. More precisely at a
        given time `t`, for each connection weight, say `w_{ij}(t)`, a
        corresponding `\Delta w_{ij}(t)` is computed and the perceptron is
        updated so that `w_{ij}(t+1) = w_{ij}(t) - \eta \Delta w_{ij}(t) +
        \alpha \Delta w_{ij}(t-1)`. This rule implements a local descent in
        the error space so that eventually the obtained minimizes locally the
        training error. The values `\eta` and `\alpha` can be chosen through
        the following named arguments:

        - ``learning_rate`` corresponds to`\eta`, the so-called *learning
          rate*, with a default value of 0.1. The higher this value, the more
          extended will be the local steps of the algorithm. This will improve
          convergence but will also raise the risk of outrunning an optimum
          and starting to oscillate around it.

        - ``momentum_term`` corresponds to `\alpha`, the so-called *momentum
          term* as it features a momentum which increases the actual step
          when the surface is smooth and tends to decrement it otherwise. Its
          default value is 0, corresponding to the original version of the
          backpropagation algorithm.

        The following named argument also affect the learning behaviour:

        - ``selector`` sets an iterator selecting the next example to be fed to
          the learnin algorithm. Its default value cycles through the provided
          sample.

        - ``batch`` -- boolean flag setting batch learning mode, in which the
          update values `\Delta w_{ij}` are cumulated for all example in the
          sample and subsequently used in order to modify the perceptron,
          rather than the standard online mode where the perceptron is updated
          after each single example presentation. Its default value is
          ``False``, corresponding to the online mode previously illustrated.
          It is worth noting that when the algorithm is run for a fixed number
          of iterations (i.e. through ``FixedIterationsStoppingCriterion``),
          an iteration corresponds to one example presentation for online mode
          and to the presentation of the whole sample for batch mode.

        - ``min_error`` -- error value under which no update will occur on
          connections and thresholds. This argument can be used in order to
          avoid that some examples are overlearnt at the expense of the
          remaining ones. The default value is 0, leading to updating the
          perceptron regardless of how small is the error on an example.

        AUTHORS:

        - Dario Malchiodi (2010-03-31)

        """

        IterativeAlgorithm.run(self, **kwargs)
        try:
            # batch or online learning
            batch = kwargs['batch']
        except KeyError:
            batch = False

        try:
            learning_rate = kwargs['learning_rate']
        except KeyError:
            learning_rate = 0.1

        try:
            momentum_term = kwargs['momentum_term']
        except KeyError:
            momentum_term = 0

        try:
            min_error = kwargs['min_error']
        except KeyError:
            min_error = 0

        dims = transpose((self.dimensions[1:], self.dimensions[:-1]))
        last_delta = [zeros(shape) for shape in dims]
        if self.threshold:
            last_delta_threshold = [zeros(shape) \
                for shape in self.dimensions[1:]]
        while self.stop_criterion.stop() == False:
            if batch:
                cumul_delta = last_delta = [zeros(shape) for shape in dims]
                if self.threshold:
                    cumul_delta_thresholds = [zeros(shape) \
                        for shape in self.dimensions[1:]]
                for elem in self.sample:
                    answer = self.model.compute(elem.pattern,
                        full_state=True, show_net=True, no_unbox=True)

                    delta = [[]] * (len(self.dimensions) - 1)

                    error = transpose(answer[-1])[0] - elem.label
                    if abs(error) < min_error:
                        continue
                    derivative = [\
                        self.model.get_activation(-1).compute_derivative(net,
                        func_value=val)
                        for (val, net)  in answer[-1]]

                    delta[-1] = error * derivative

                    for lev in range(1, len(self.dimensions) - 1):
                        derivatives = [self.model.get_activation(-lev - \
                            1).compute_derivative(net, func_value=val) \
                            for (val, net)  in answer[-lev - 1]]

                        prop_delta = \
                            dot(transpose(self.model.connections[-lev]),
                            delta[-lev])
                        delta[-lev - 1] = array(derivatives * prop_delta)

                    for lev in range(1, len(self.dimensions)):
                        new_delta = -learning_rate * outer(delta[-lev],
                           transpose(answer[-lev - 1])[0]) + \
                           momentum_term * last_delta[-lev]
                        cumul_delta[-lev] += new_delta
                        last_delta[-lev] = new_delta

                        if self.threshold:
                            new_delta = -learning_rate * delta[-lev] +\
                                momentum_term * last_delta_threshold[-lev]
                            cumul_delta_thresholds[-lev] += new_delta
                            last_delta_threshold[-lev] = new_delta

                for lev in range(1, len(self.dimensions)):
                    self.model.connections[-lev] += cumul_delta[-lev]

                    if self.threshold:
                        self.model.thresholds[-lev] += \
                            cumul_delta_thresholds[lev]

            else:
                elem = self.sample_selector.next()
                answer = self.model.compute(elem.pattern,
                    full_state=True, show_net=True, no_unbox=True)

                delta = [[]] * (len(self.dimensions) - 1)

                error = transpose(answer[-1])[0] - elem.label
                if abs(error) < min_error:
                    continue
                derivative = [\
                    self.model.get_activation(-1).compute_derivative(net,
                    func_value=val)
                    for (val, net)  in answer[-1]]

                delta[-1] = error * derivative

                for lev in range(1, len(self.dimensions) - 1):
                    derivatives = [self.model.get_activation(-lev - \
                        1).compute_derivative(net, func_value=val) \
                        for (val, net)  in answer[-lev - 1]]

                    prop_delta = dot(transpose(self.model.connections[-lev]),
                        delta[-lev])
                    delta[-lev - 1] = array(derivatives * prop_delta)

                for lev in range(1, len(self.dimensions)):
                    new_delta = -learning_rate * outer(delta[-lev],
                        transpose(answer[-lev - 1])[0]) + \
                        momentum_term * last_delta[-lev]
                    self.model.connections[-lev] += new_delta
                    last_delta[-lev] = new_delta

                    if self.threshold:
                        new_delta = -learning_rate * delta[-lev] +\
                            momentum_term * last_delta_threshold[-lev]
                        self.model.thresholds[-lev] += new_delta
                        last_delta_threshold[-lev] = new_delta

            self.notify_observers()
