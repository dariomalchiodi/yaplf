
r"""
Package handling neural learning algorithms in yaplf.

Package yaplf.algorithms.neural contains all the  classes handling neural
learning algorithms in yaplf.

TODO:

- pep8 checked
- pylint score: 8.72

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

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


from numpy import random, hstack, array

from yaplf.utility.activation import SigmoidActivationFunction
from yaplf.models.neural import Perceptron
from yaplf.algorithms import LearningAlgorithm, IterativeAlgorithm


class PerceptronAlgorithm(LearningAlgorithm):
    r"""
    Base class for iterative algorithms learning a single-layer perceptron.
    The class accounts for checking the use of thresholds and for randomly
    choose the initial values for weights and thresholds, while it delegates
    its superclass for sample memorization into a field.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``LabeledExample`` containing the sample
      to be learnt.

    - ``threshold`` -- boolean (default: ``True``) flag setting the use of
      thresholded perceptrons.

    - ``weight_bound`` -- float:(default: 0.1) upper bound of the interval in
      which the initial weights and thresholds are chosen uniformly at random
      (the lower bound of this interval is ``-1 * weight_bound``). A
      ``ValueError`` is thrown if this parameter is not positive.

    OUTPUT:

    ``PerceptronAlgorithm`` object.

    EXAMPLES:

    See the examples for specific subclasses such as
    ``GradientPerceptronAlgorithm`` in this package.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``PerceptronAlgorithm`` for full documentation.

        """

        try:
            self.threshold = kwargs['threshold']
        except KeyError:
            self.threshold = True
        self.model = None

        LearningAlgorithm.__init__(self, sample)
        self.reset(**kwargs)

    def reset(self, **kwargs):
        r"""
        Reset weights and thresholds of the inferred perceptron to randomly
        chosen values.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``sample`` -- list or tuple of ``LabeledExample`` containing the
          sample to be learnt.

        - ``threshold`` -- boolean (default: ``True``) flag setting the use of
          thresholded perceptrons.

        - ``weight_bound`` -- float:(default: 0.1) upper bound of the interval
          in which the initial weights and thresholds are chosen uniformly at
          random (the lower bound of this interval is ``-1 * weight_bound``). A
          ``ValueError`` is thrown if this parameter is not positive.

        OUTPUT:

        No output.

        EXAMPLES:

        See the examples for specific subclasses such as
        ``GradientPerceptronAlgorithm`` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            # picks initial weights and threshold uniformly
            # between -weight_bound and weight_bound
            init_bound = kwargs['weight_bound']
            if init_bound <= 0:
                raise ValueError(
                    'The weight_bound parameter should be positive')
        except KeyError:
            init_bound = 0.1

#        weights = [[random.uniform(-1 * init_bound, init_bound)
#            for i in range(len(self.sample[0].pattern))]
#            for j in range(len(self.sample[0].label))]
        weights = random.uniform(-1 * init_bound, init_bound,
            (len(self.sample[0].label), len(self.sample[0].pattern)))
        if self.threshold:
#            thr = [random.uniform(-1 * init_bound, init_bound)
#                for i in range(len(self.sample[0].label))]
            thr = random.uniform(-1 * init_bound, init_bound,
                len(self.sample[0].label))
        else:
            thr = [0] * len(self.sample[0].label)
        self.model = Perceptron(weights, threshold=thr)

    def run(self):
        r"""
        Run the learning algorithm. Not implemented in the base class.

        INPUT:

        No input.

        OUTPUT:

        No output.

        EXAMPLES:

        See the examples for specific subclasses such as
        ``GradientPerceptronAlgorithm`` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError()


class RosenblattPerceptronAlgorithm(PerceptronAlgorithm,
    IterativeAlgorithm):
    r"""
    Rosenblatt learning algorithm for one-layer perceptron [Rosenblatt, 1958].

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``LabeledExample`` containing the sample
      to be learnt.

    - ``threshold`` -- boolean (default: ``True``) flag setting the use of
      thresholded perceptrons.

    - ``weight_bound`` -- float:(default: 0.1) upper bound of the interval in
      which the initial weights and thresholds are chosen uniformly at random
      (the lower bound of this interval is ``-1 * weight_bound``). A
      ``ValueError`` is thrown if this parameter is not positive.

    OUTPUT:

    ``RosenblattPerceptronAlgorithm`` object.

    EXAMPLES:

    Consider the following data set summarizing the binary AND function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), (1,)),
        ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
        ... LabeledExample((1, 0), (0,))]

    This sample is linearly separable thus it is learnable through the
    Rosenblatt algorithm, for instance in the following configuration where
    threshold and initial weight bounds are used:

    ::

        >>> from yaplf.algorithms.neural import RosenblattPerceptronAlgorithm
        >>> alg = RosenblattPerceptronAlgorithm(and_sample, threshold = True,
        ... weight_bound = 0.1)

    In order to actually run the algorithm it is necessary to specify a
    stopping criterion (the default behaviour would only execute a learning
    iteration, probably not going so far). In order to keep it simple, one can
    chose ``FixedIterationsStoppingCriterion`` so as to run a fixed number of
    iterations, say `100`:

    ::

        >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
        >>> alg.run(stopping_criterion = FixedIterationsStoppingCriterion(100))

    The inferred model can be inspected through the ``model`` field in the
    ``LearningAlgorithm`` object:

    ::

        >>> alg.model # random
        Perceptron([array([ 2.07686954,  1.00037639])], threshold =
        [2.9671578749763876])

    A better way of assessing the performance of the algorithm is that of
    invoking the ``test`` function inherited from the ``Model`` class in order
    to see how each example has been classified:

    ::

        >>> from yaplf.utility.error import MaxError
        >>> alg.model.test(and_sample, MaxError(), verbose = True)
        (1, 1) mapped to 1, label is (1,), error 0
        (0, 0) mapped to 0, label is (0,), error 0
        (0, 1) mapped to 0, label is (0,), error 0
        (1, 0) mapped to 0, label is (0,), error 0
        Maximum error: 0
        0

    The second argument in ``test`` is an object subclassing ``ErrorModel``,
    allowing to specify a given metric in order to compute the distance between
    expected and actual output. The argument's default, amounting to the mean
    square error, has been in this case overridden to ``MaxError()``, thus
    computing the maximum error. Note also that in this particular case of null
    error in the whole data set the two criterions are equivalent.

    As a final remark, it should be highlighted that these examples are shown
    for illustrative purpose. The suitable way of assessing a learnt model
    performance involves more complex techniques involving the use of a test
    set possibly coupled with a cross validation procedure (see function
    ``cross_validation`` in package ``yaplf.utility.validation``.)

    REFERENCES

    [Rosenblatt, 1958] Frank Rosenblatt, The Perceptron: A Probabilistic Model
    for Information Storage and Organization in the Brain, Psychological
    Review, v65, No. 6, pp. 386-408, doi:10.1037/h0042519.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``RosenblattPerceptronAlgorithm`` for full definition.

        """

        PerceptronAlgorithm.__init__(self, sample, **kwargs)
        IterativeAlgorithm.__init__(self, sample)
        PerceptronAlgorithm.reset(self, **kwargs)

    def run(self, **kwargs):
        r"""
        Run the Rosenblatt learning algorithm.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``stopping_criterion`` -- ``StoppingCriterion`` object (default:
          ``FixedIterationsStoppingCriterion()``, amounting to the execution of
          one iteration step) to be used in order to decide when learning
          should be stopped.

        - ``selector`` -- iterator (default: ``sequential_selector``) yielding
          the examples to be fed to the learning algorithm.

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``Perceptron`` instance.

        EXAMPLES:

        Consider the following linearly separable data set summarizing the
        binary AND function, and a ``RosenblattPerceptronAlgorithm`` instance:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), (1,)),
            ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
            ... LabeledExample((1, 0), (0,))]
            >>> from yaplf.algorithms.neural import \
            ... RosenblattPerceptronAlgorithm
            >>> alg = RosenblattPerceptronAlgorithm(and_sample,
            ... threshold = True, weight_bound = 0.1)

        In order to actually run the algorithm it is necessary to specify a
        stopping criterion (the default behaviour would only execute a learning
        iteration, probably not going so far). In order to keep it simple, one
        can chose ``FixedIterationsStoppingCriterion`` so as to run a fixed
        number of iterations, say `100`:

        ::

            >>> from yaplf.utility.stopping import \
            ... FixedIterationsStoppingCriterion
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(100))

        Another possible variation is that modifying the way examples are
        chosen and fed to the learning algorithm at each iteration. The default
        choice cycles between available examples, but it possible for instance
        to pick examples at random. This requires to specify a value for the
        ``selector`` named argument:

        ::

            >>> alg.reset()
            >>> from yaplf.utility.selection import random_selector
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(100),
            ... selector = random_selector)

        Note how, in order to restart the learning rocedure from fresh, the
        ``reset`` function was called. In this way the inferred model is
        initialized using randomly picked values.

        The inferred model is available through the ``model`` field. In order
        to get some information about its the performance is that of invoking
        the ``test`` function inherited from the ``Model`` class so as to get
        the mean approximation error, evaluated on all fed examples:

        ::

            >>> alg.model.test(and_sample)
            0.0

        As a final remark, it should be highlighted that these examples are
        shown for illustrative purpose. The suitable way of assessing a learnt
        model performance involves more complex techniques involving the use of
        a test set possibly coupled with a cross validation procedure (see
        function ``cross_validation`` in package ``yaplf.utility.validation``.)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        IterativeAlgorithm.run(self, **kwargs)
        while self.stop_criterion.stop() == False:
            elem = self.sample_selector.next()
            answer = self.model.compute(elem.pattern)

            if type(answer) != type(()):
                answer = (answer,)

#            for i in range(len(self.sample[0].label)):
#                if answer[i] == 1 and elem.label[i] == 0:
#                    self.model.weights[i] -= hstack((elem.pattern,
#                        (-1 if self.threshold else 0)))
#
#                if answer[i] == 0 and elem.label[i] == 1:
#                    self.model.weights[i] += hstack((elem.pattern,
#                        (-1 if self.threshold else 0)))

            temporary_weights = self.model.weights
            temporary_threshold = self.model.threshold
            for i in range(len(self.sample[0].label)):
                if answer[i] != elem.label[i]:
                    temporary_weights[i] += \
                        [(elem.label[i] - answer[i]) * p for p in  elem.pattern]
                    temporary_threshold[i] += (elem.label[i] - answer[i]) * \
                        (-1 if self.threshold else 0)

            self.model.set_weights_and_threshold(temporary_weights, temporary_threshold)      


            self.notify_observers()


class GradientPerceptronAlgorithm(PerceptronAlgorithm,
    IterativeAlgorithm):
    r"""Gradient-based algorithm for one-layer perceptron.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``LabeledExample`` containing the sample
      to be learnt.

    - ``threshold`` -- boolean (default: ``True``) flag setting the use of
      thresholded perceptrons.

    - ``weight_bound`` -- float (default: 0.1) upper bound of the interval in
      which the initial weights and thresholds are chosen uniformly at random
      (the lower bound of this interval is ``-1 * weight_bound``). A
      ``ValueError`` is thrown if this parameter is not positive.

    - ``beta`` -- float (default: 1) value for the `\beta` parameter of the
      sigmoidal activation function to be used in order to equip the inferred
      perceptron.

    OUTPUT:

    ``GradientPerceptronAlgorithm`` object.

    EXAMPLES:

    Consider the following data set summarizing the binary AND function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), (1,)),
        ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
        ... LabeledExample((1, 0), (0,))]

    This sample is linearly separable thus it is learnable through the gradient
    descent algorithm, for instance in the following configuration where
    threshold and initial weight bounds are used:

    ::

        >>> from yaplf.algorithms.neural import GradientPerceptronAlgorithm
        >>> alg = GradientPerceptronAlgorithm(and_sample, threshold = True,
        ... weight_bound = 0.1, beta = 0.8)

    In order to actually run the algorithm it is necessary to specify a
    stopping criterion (the default behaviour would only execute a learning
    iteration, probably not going so far). In order to keep it simple, one can
    chose ``FixedIterationsStoppingCriterion`` so as to run a fixed number of
    iterations, say `5000`:

    ::

        >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
        >>> alg.run(stopping_criterion = \
        ... FixedIterationsStoppingCriterion(5000), batch = False,
        ... learning_rate = .1)

    The inferred model can be inspected through the ``model`` field in the
    ``LearningAlgorithm`` object:

    ::

        >>> alg.model # random
        Perceptron([array([ 4.01133491,  4.00742238])], threshold =
        [6.1595362999239365], activation =
        SigmoidActivationFunction(0.800000000000000))

    A better way of assessing the performance of the algorithm is that of
    invoking the ``test`` function inherited from the ``Model`` class in order
    to see how each example has been classified:

    ::

        >>> from yaplf.utility.error import MaxError
        >>> alg.model.test(and_sample, MaxError(), verbose = True) # random
        (1, 1) mapped to 0.815684217643, label is (1,), error 0.0339723076259
        (0, 0) mapped to 0.007191564133, label is (0,), error 5.17185946842e-05
        (0, 1) mapped to 0.151653463349, label is (0,), error 0.0229987729459
        (1, 0) mapped to 0.15205659453, label is (0,), error 0.02312120794
        Maximum error: 0.0339723076259
        0.033972307625870855

    The second argument in ``test`` is an object subclassing ``ErrorModel``,
    allowing to specify a given metric in order to compute the distance between
    expected and actual output. The argument's default, amounting to the mean
    square error, has been in this case overridden to ``MaxError()``, thus
    computing the maximum error.

    As a final remark, it should be highlighted that these examples are shown
    for illustrative purpose. The suitable way of assessing a learnt model
    performance involves more complex techniques involving the use of a test
    set possibly coupled with a cross validation procedure (see function
    ``cross_validation`` in package ``yaplf.utility.validation``.)

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``GradientPerceptronAlgorithm`` for full documentation.

        """

        PerceptronAlgorithm.__init__(self, sample, **kwargs)
        IterativeAlgorithm.__init__(self, sample)
        # This calms down pylint
        self.beta = None
        self.reset(**kwargs)

    def reset(self, **kwargs):
        r"""
        Reset weights and thresholds of the inferred Perceptron picking values
        at random.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``threshold`` -- boolean (default: ``True``) flag setting the use of
          thresholded perceptrons.

        - ``weight_bound`` -- float (default: 0.1) upper bound of the interval
          in which the initial weights and thresholds are chosen uniformly at
          random (the lower bound of this interval is ``-1 * weight_bound``). A
          ``ValueError`` is thrown if this parameter is not positive.

        - ``beta`` -- float (default: 1) value for the `\beta` parameter of the
          sigmoidal activation function to be used in order to equip the
          inferred perceptron.

        OUTPUT:

        No output. After the invocation the initialized model is available
        through the ``model`` field, in form of a ``Perceptron`` instance.

        EXAMPLES:

        Consider the following linearly separable data set summarizing the
        binary AND function, and a ``GradientPerceptronAlgorithm`` instance
        which is run for `100` iterations:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), (1,)),
            ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
            ... LabeledExample((1, 0), (0,))]
            >>> from yaplf.algorithms.neural import \
            ... GradientPerceptronAlgorithm
            >>> alg = GradientPerceptronAlgorithm(and_sample, threshold = True,
            ... weight_bound = 0.1, beta = 0.8)
            >>> from yaplf.utility.stopping import \
            ... FixedIterationsStoppingCriterion
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(100), batch = False,
            ... learning_rate = .1)

        As the currently inferred perceptron is saved into the
        ``GradientPerceptronAlgorithm`` instance state, it is possible to
        continue the learning process for subsequent `1000` iterations, maybe
        increasing the learning rate:

        ::

            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(1000), batch = False,
            ... learning_rate = 1)

        Suppose now that the currently inferred model is completely wrong,
        for instance because of the overly increased learning rate value. In
        such cases one prefers to completely forget the available model and
        start a fresh learning session. Instead of creating a new algorithm
        instance, the new session can be started simply resetting the existing
        algorithm: indeed, a call to the ``reset`` function initializes the
        inferred model using random values:

        ::

            >>> alg.reset()

        Now it is possible to start a new learning phase, for instance
        restoring previous learning rate value and modifying the selection
        policy for examples:

        ::

            >>> from yaplf.utility.selection import random_selector
            >>> alg.run(stopping_criterion = \
            ... FixedIterationsStoppingCriterion(100),
            ... selector = random_selector)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        PerceptronAlgorithm.reset(self, **kwargs)
        try:
            self.beta = kwargs['beta']
        except KeyError:
            self.beta = 1
        self.model.activation = SigmoidActivationFunction(self.beta)

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

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``Perceptron`` instance.

        EXAMPLES:

        Consider the following data set summarizing the binary AND function,
        and an instance of ``GradientPerceptronAlgorithm``:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), (1,)),
            ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (0,)),
            ... LabeledExample((1, 0), (0,))]
            >>> from yaplf.algorithms.neural import GradientPerceptronAlgorithm
            >>> alg = GradientPerceptronAlgorithm(and_sample, threshold = True,
            ... weight_bound = 0.1, beta = 0.8)

        In order to actually run the algorithm it is necessary to specify a
        stopping criterion (the default behaviour would only execute a learning
        iteration, probably not going so far). In order to keep it simple, one
        can chose ``FixedIterationsStoppingCriterion`` so as to run a fixed
        number of iterations, say `5000`:

        ::

            >>> from yaplf.utility.stopping import \
            ... FixedIterationsStoppingCriterion
            >>> alg.run(stopping_criterion = \
            .... FixedIterationsStoppingCriterion(5000), batch = False,
            ... learning_rate = .1)

        The inferred model can be inspected through the ``model`` field in the
        ``LearningAlgorithm`` object:

        ::

             >>> alg.model # random
            Perceptron([array([ 4.01133491,  4.00742238])], threshold =
            [6.1595362999239365], activation =
            SigmoidActivationFunction(0.800000000000000))

        A better way of assessing the performance of the algorithm is that of
        invoking the ``test`` function inherited from the ``Model`` class in
        order to see how each example has been classified:

        ::

            >>> from yaplf.utility.error import MaxError
            >>> alg.model.test(and_sample, MaxError(), verbose = True) # random
            (1, 1) mapped to 0.81568421764, label is (1,), error 0.033972307625
            (0, 0) mapped to 0.00719156413, label is (0,), error 5.17185946e-05
            (0, 1) mapped to 0.15165346334, label is (0,), error 0.022998772945
            (1, 0) mapped to 0.1520565945, label is (0,), error 0.0231212079
            Maximum error: 0.0339723076259
            0.033972307625870855

        The second argument in ``test`` is an object subclassing
        ``ErrorModel``, allowing to specify a given metric in order to compute
        the distance between expected and actual output. The argument's
        default, amounting to the mean square error, has been in this case
        overridden to ``MaxError()``, thus computing the maximum error.

        As a final remark, it should be highlighted that these examples are
        shown for illustrative purpose. The suitable way of assessing a learnt
        model performance involves more complex techniques involving the use of
        a test set possibly coupled with a cross validation procedure (see
        function ``cross_validation`` in package ``yaplf.utility.validation``.)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

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

        while self.stop_criterion.stop() == False:
            if batch:
                delta = array([[0.0] * len(self.sample[0].pattern) + 1] * \
                    len(self.sample[0].label))
                for elem in self.sample:
                    answer = self.model.compute(elem.pattern)
                    if type(answer) != type(()):
                        answer = (answer,)
                    delta += [2 * self.beta * learning_rate * \
                        array([(answer[s] - elem.label[s]) * answer[s] * \
                        (1 - answer[s]) * hstack((elem.pattern, \
                        (-1 if self.threshold else 0)))[t] \
                        for t in range(len(self.sample[0].pattern) + 1)])
                        for s in range(len(self.sample[0].label))]
                delta = delta.tolist()
            else:
                elem = self.sample_selector.next()
                answer = self.model.compute(elem.pattern)
                if type(answer) != type(()):
                    answer = (answer,)
                delta = [2 * self.beta * learning_rate * \
                    array([(answer[s] - elem.label[s]) * answer[s] * \
                    (1 - answer[s]) * hstack((elem.pattern, \
                    (-1 if self.threshold else 0)))[t] \
                    for t in range(len(self.sample[0].pattern) + 1)]) \
                    for s in range(len(self.sample[0].label))]
            self.model.weights = [self.model.weights[i] - delta[i] \
                for i in range(len(delta))]
            self.notify_observers()

