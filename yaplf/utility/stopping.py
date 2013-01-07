

r"""
Module handling stopping criterions in yaplf

Module :mod:`yaplf.utility.stopping` contains stopping criterions in yaplf.

AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  :mod:`yaplf.utility`, containing base, fixed-iteration, train- and test-error
  criterions.

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


from yaplf.utility.error import MSE


class StoppingCriterion(object):
    r"""
    Base class for stopping criteria in iterative learning algorithms.
    Each subclass should implement a method :meth:`stop`, returning a boolean
    value setting whether or not the learning process should be stopped.

    EXAMPLES:

    See examples for specific subclasses, such as
    :class:FixedIterationsStoppingCriterion in this package.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self):
        r"""
        See :class`StoppingCriterion` for full documentation.
        """

        self.learning_algorithm = None

    def stop(self):
        r"""
        Returns a flag indicating whether learning should be stopped.

        :returns: ``True`` if learning should be stopped, ``False`` otherwise.

        :rtype: boolean

        EXAMPLES:

        See examples for specific subclasses, such as
        :class:`FixedIterationsStoppingCriterion` in this package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError('stop() not callable in base class')

    def register_learning_algorithm(self, alg):
        r"""
        Register a learning algorithm as instance variable of the stopping
        criterion. In this way the criterion is able to access to meaningful
        information for its run-time behaviour, such as model and/or sample.

        Note that learning algorithms whose execution is based on an iterative
        process whose termination relies on a stopping criterion should
        subclass the :class:`IterativeLearningAlgorithm` in module
        :mod:`yaplf.algorithms`, which automatically registers on its stopping
        criterion through invocation of this function.

        :param alg: learning algorithm to be registered.

        :type alg: :class:`yaplf.algorithms.LearningAlgorithm`

        EXAMPLES:

        Consider the binary XOR function, and a dummy learning algorithm
        actually not learning anything in that it always infers a model
        constantly outputting ``0``. Moreover, assume you want to test this
        inferred model on the following test set:

        >>> from yaplf.data import LabeledExample
        >>> from yaplf.models import ConstantModel
        >>> xor_sample = [LabeledExample((1, 1), (0,)),
        ... LabeledExample((0, 0), (0,)), LabeledExample((1, 0), (1,))]
        >>> from yaplf.utility.stopping import TestErrorStoppingCriterion
        >>> test_set = [LabeledExample((0, 1), (1,))]
        >>> f = TestErrorStoppingCriterion(test_set, .01)
        >>> from yaplf.algorithms import IdiotAlgorithm
        >>> idiot = IdiotAlgorithm(xor_sample, stopping_criterion=f,
        ... model=ConstantModel((0,)), verbose=False)

        (Of course this approach is completely wrong in order to properly
        learn anything, but the point is that of exemplifying how to tie a
        learning algorithm with a stopping criterion). Now the learning
        algorithm knows about its stopping criterion, so that the former can
        repeatedly query the latter in order to know whether or not it should
        continue the training phase. But what happens if the latter need
        information from the former in order to fulfill its end? For instance,
        this particular case relies on a :class:`TestErrorStoppingCriterion`,
        meaning that training will continue until the inferred model scores a
        sufficiently low error on a test set. But in order to compute this
        error the stopping criterion should access the learning algorithm
        internal state, notably the current form of the inferred model. In
        such cases it is therefore necessary to call the
        :meth:`register_learning_algorithm` function, which allow the stopping
        criterion to maintain a reference to the learning algorithm in order
        to subsequently access it:

        >>> f.register_learning_algorithm(idiot)
        >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
        [False, False, False, False, False, False, False, False, False, False,
        False, False, False]

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.learning_algorithm = alg


class FixedIterationsStoppingCriterion(StoppingCriterion):
    r"""
    Stopping criterion based on the execution of a fixed number of
    iterations.

    :param num_iterations: number of iterations to be executed.

    :type num_iterations: integer, default: 1

    EXAMPLES:

    The :obj:`num_iterations` argument should be set to a positive integer,
    otherwise a :exc:`ValueError` is thrown:


    >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
    >>> FixedIterationsStoppingCriterion()
    FixedIterationsStoppingCriterion()
    >>> FixedIterationsStoppingCriterion(200)
    FixedIterationsStoppingCriterion(200)
    >>> FixedIterationsStoppingCriterion(0)
    Traceback (most recent call last):
       ...
    ValueError: the specified num_iterations parameter is not a positive
    integer
    >>> FixedIterationsStoppingCriterion(2.3)
    Traceback (most recent call last):
       ...
    ValueError: the specified num_iterations parameter is not a positive
    integer

    Once successfully created, the object exposes a :meth:`stop` method whose
    return value will be ``False`` precisely for a number of invocations equal
    to the number of iterations:

    >>> f = FixedIterationsStoppingCriterion(10)
    >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
    [False, False, False, False, False, False, False, False, False, False,
    True, True, True]

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, num_iterations=1):
        r"""
        See :class:`FixedIterationsStoppingCriterion` for full documentation.

        """

        StoppingCriterion.__init__(self)
        if num_iterations <= 0 or int(num_iterations) != num_iterations:
            raise ValueError('the specified num_iterations parameter is not \
                a positive integer')
        self.num_iterations = num_iterations
        self.current_iteration = 0

    def __repr__(self):
        result = 'FixedIterationsStoppingCriterion('
        if self.num_iterations != 1:
            result += str(self.num_iterations)
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other) and \
            self.num_iterations == other.num_iterations

    def __hash__(self):
        return hash(("FixedIterationsStoppingCriterion",
            hash(self.num_iterations)))

    def __nonzero__(self):
        return True

    def reset(self):
        r"""
        Reset the stopping criterion.

        EXAMPLES:

        This method should be used when an instance of the class is to be
        reused for a new training session. It essentially resets to ``0`` the
        current iteration number. In such cases the invocation of :meth:`reset`
        is peculiar, otherwise the first call to :meth:`stop` will always
        return ``True`` no matter the maximum number of iterations:

        >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
        >>> from yaplf.utility.validation import train_and_test
        >>> from yaplf.data import LabeledExample
        >>> from yaplf.algorithms.neural import RosenblattPerceptronAlgorithm
        >>> train_sample = (LabeledExample((0, 0,), (0,)), \
        ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)))
        >>> test_sample = (LabeledExample((1, 1), (1,)),)
        >>> sc = FixedIterationsStoppingCriterion(5000)
        >>> train_and_test(RosenblattPerceptronAlgorithm, train_sample,
        ... test_sample, {'threshold': True, 'weight_bound': 0.1},
        ... run_parameters = {'stopping_criterion': sc})
        0.0
        >>> sc.reset()
        >>> train_and_test(RosenblattPerceptronAlgorithm, train_sample,
        ... test_sample, {'threshold': True, 'weight_bound': 0.1},
        ... run_parameters = {'stopping_criterion': sc})
        0.0

        Without the call to :meth:`reset` it is likely that the second training
        will have a different error score.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.current_iteration = 0

    def stop(self):
        r"""
        Returns a flag indicating whether learning should be stopped.

        :returns: ``True`` if learning should be stopped, ``False`` otherwise.

        :rtype: boolean

        EXAMPLES:

        Once successfully created, the object exposes a :meth:`stop` method
        whose return value will be ``False`` precisely for a number of
        invocations equal to the number of iterations:

        >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
        >>> f = FixedIterationsStoppingCriterion(10)
        >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
        [False, False, False, False, False, False, False, False, False, False,
        True, True, True]

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.current_iteration += 1
        return self.current_iteration > self.num_iterations


class TestErrorStoppingCriterion(StoppingCriterion):
    r"""
    Stopping criterion based on a threshold on the test error.

    :param test_set: examples to be used in order to compute the error.

    :type test_set: list/tuple

    :param max_error: test error threshold.

    :type max_error: double, default: 0.1

    :param error_model: --  error model to be used.

    :type error_model: :class:`yaplf.utility.error.ErrorModel`, default:
      :class:`yaplf.utility.error.MSE`

    :param max_iterations: maximum number of iterations before training is
      stopped anyway. Note that it is always possible to indefinitely run a
      learning algorithm until the threshold is attained through the setting
      ``max_iterations = float("infinity")``.

    :type max_iterations: float, default: 5000


    EXAMPLES:

    The :obj:`max_iterations` argument should be set to a positive integer, as
    well as :obj:`max_error` should be set to a positive number, otherwise a
    :exc:`ValueError` is thrown:

    >>> from yaplf.utility.stopping import TestErrorStoppingCriterion
    >>> from yaplf.data import LabeledExample
    >>> xor_test = [LabeledExample((0, 1), (1,))]
    >>> TestErrorStoppingCriterion(xor_test)
    TestErrorStoppingCriterion([LabeledExample((0, 1), (1,))])
    >>> TestErrorStoppingCriterion(xor_test, 0.01)
    TestErrorStoppingCriterion([LabeledExample((0, 1), (1,))], 0.01)
    >>> TestErrorStoppingCriterion(xor_test, -4)
    Traceback (most recent call last):
       ...
    ValueError: the specified max_error parameter should be positive.

    Once created, a :class:`TestErrorStoppingCriterion` object implements a
    :meth:`stop` method whose return value is ``False`` until either the learnt
    model scores an error lower than :obj:`max_error` or the maximum number of
    iterations is exceeded. Note that in the following instructions the dummy
    learning algorithm is explicitly tied to the stopping criterion through the
    :meth:`register_learning_algorithm` method. This is due to the fact that
    :class:`yaplf.algorithms.IdiotLearningAlgorithm` doesn't subclass
    :class:`yaplf.algorithms.IterativeLearningAlgorithm` which does this job
    automatically:

    >>> f = TestErrorStoppingCriterion(xor_test, .01)
    >>> xor_train = [LabeledExample((1, 1), (0,)),
    ... LabeledExample((0, 0), (0,)), LabeledExample((1, 0), (1,))]
    >>> from yaplf.algorithms import IdiotAlgorithm
    >>> from yaplf.models import ConstantModel
    >>> idiot = IdiotAlgorithm(xor_train, stopping_criterion=f,
    ... model=ConstantModel((0,)), verbose=False)
    >>> f.register_learning_algorithm(idiot)
    >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
    [False, False, False, False, False, False, False, False, False, False,
    False, False, False]

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, test_set, max_error=0.1, error_model=MSE(),
        max_iterations=5000):
        r"""
        See :class:`TestErrorStoppingCriterion` for full documentation.

        """

        StoppingCriterion.__init__(self)
        if max_error < 0:
            raise ValueError('the specified max_error parameter should be \
                positive.')
        if max_iterations <= 0:
            raise ValueError('the maximum number of iterations should be \
                positive.')
        self.test_set = test_set
        self.max_error = max_error
        self.error_model = error_model
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def __repr__(self):
        result = 'TestErrorStoppingCriterion('
        result += str(self.test_set) + ', '
        if self.max_error != 0.1:
            result += str(self.max_error) + ', '
        if self.error_model != MSE():
            result += str(self.error_model) + ', '
        if result[-2:] == ', ':
            result = result[:-2]
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other) and self.test_set == other.test_set \
            and  self.max_error == other.max_error \
            and self.error_model == other.error_model

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("TestErrorStoppingCriterion", hash(self.test_set),
            hash(self.max_error), hash(self.error_model)))

    def __nonzero__(self):
        return True

    def reset(self):
        r"""
        Reset the stopping criterion.

        EXAMPLES:

        This method should be used when an instance of the class is to be
        reused for a new training session. It essentially resets to ``0`` the
        current iteration number. In such cases the invocation of :meth:`reset`
        is peculiar when the maximum number of iterations has been reached,
        otherwise the first call to :meth:`stop` will always return ``True`` no
        matter the maximum number of iterations:

        >>> from yaplf.utility.stopping import TestErrorStoppingCriterion
        >>> from yaplf.utility.validation import train_and_test
        >>> from yaplf.data import LabeledExample
        >>> from yaplf.algorithms.neural import RosenblattPerceptronAlgorithm
        >>> train_sample = (LabeledExample((0, 0,), (0,)), \
        ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)))
        >>> test_sample = (LabeledExample((1, 1), (1,)),)
        >>> sc = TestErrorStoppingCriterion(test_sample)
        >>> train_and_test(RosenblattPerceptronAlgorithm, train_sample,
        ... test_sample, {'threshold': True, 'weight_bound': 0.1},
        ... run_parameters = {'stopping_criterion': sc}, max_iterations = 10)
        0.0
        >>> sc.reset()
        >>> train_and_test(RosenblattPerceptronAlgorithm, train_sample,
        ... test_sample, {'threshold': True, 'weight_bound': 0.1},
        ... run_parameters = {'stopping_criterion': sc}, max_iterations = 10)
        0.0

        Without the call to :meth:`reset` it is likely that the second training
        will have a different error score.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.current_iteration = 0

    def stop(self):
        r"""
        Returns a flag indicating whether learning should be stopped.

        :returns: ``True`` if learning should be stopped, ``False`` otherwise.

        :rtype: boolean

        EXAMPLES:

        Once created, a :class:`TestErrorStoppingCriterion` object implements a
        :meth:`stop` method whose return value is ``False`` until either the
        learnt model scores an error lower than :obj:`max_error` or the maximum
        number of iterations is exceeded. Note that in the following
        instructions the dummy learning algorithm is explicitly tied to the
        stopping criterion through the :meth:`register_learning_algorithm`
        method. This is due to the fact that
        :class:`yaplf.algorithms.IdiotLearningAlgorithm` doesn't subclass
        :class:`yaplf.algorithms.IterativeLearningAlgorithm` which does this
        job automatically:

        >>> from yaplf.utility.stopping import TestErrorStoppingCriterion
        >>> from yaplf.data import LabeledExample
        >>> xor_test = (LabeledExample((0, 1), (1,)),)
        >>> f = TestErrorStoppingCriterion(xor_test, .01)
        >>> xor_train = [LabeledExample((1, 1), (0,)),
        ... LabeledExample((0, 0), (0,)), LabeledExample((1, 0), (1,))]
        >>> from yaplf.algorithms import IdiotAlgorithm
        >>> from yaplf.models import ConstantModel
        >>> idiot = IdiotAlgorithm(xor_train, stopping_criterion = f,
        ... model=ConstantModel((1,)), verbose = False)
        >>> f.register_learning_algorithm(idiot)
        >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
        [True, True, True, True, True, True, True, True, True, True, True,
        True, True]

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.current_iteration += 1
        alg = self.learning_algorithm
        return (alg.model.test(self.test_set, self.error_model)
            < self.max_error and self.current_iteration <= self.max_iterations
            if alg is not None else True)


class TrainErrorStoppingCriterion(TestErrorStoppingCriterion):
    r"""
    Stopping criterion based on a threshold on the train error.

    :param max_error: test error threshold.

    :type max_error: double, default: 0.1

    :param error_model: error model to be used.

    :type error_model: :class:`yaplf.utility.error.ErrorModel`, default:
      :class:`yaplf.utility.error.MSE`

    :param max_iterations: maximum number of iterations before training is
      stopped anyway. Note that it is always possible to indefinitely run a
      learning algorithm until the threshold is attained through the setting
      ``max_iterations = float("infinity")``.

    :type max_iterations: float, default: 5000

    EXAMPLES:

    The :obj:`max_iterations` argument should be set to a positive integer, as
    well as :obj:`max_error` should be set to a positive number, otherwise a
    :exc:`ValueError` is thrown:

    >>> from yaplf.utility.stopping import TrainErrorStoppingCriterion
    >>> TrainErrorStoppingCriterion()
    TrainErrorStoppingCriterion()
    >>> TrainErrorStoppingCriterion(0.01)
    TrainErrorStoppingCriterion(0.01...)
    >>> TrainErrorStoppingCriterion(-4)
    Traceback (most recent call last):
       ...
    ValueError: the specified max_error parameter should be positive.

    Once created, a :class:`TrainErrorStoppingCriterion` object implements a
    :meth:`stop` method whose return value is ``False`` until either the learnt
    model scores an error lower than :obj:`max_error` or the maximum number of
    iterations is exceeded. Note that in the following instructions the dummy
    learning algorithm is explicitly tied to the stopping criterion through the
    :meth:`register_learning_algorithm` method. This is due to the fact that
    :class:`yaplf.algorithms.IdiotLearningAlgorithm` doesn't subclass
    :class:`yaplf.algorithms.IterativeLearningAlgorithm` which does this job
    automatically:

    >>> from yaplf.data import LabeledExample
    >>> from yaplf.algorithms import IdiotAlgorithm
    >>> from yaplf.models import ConstantModel
    >>> f = TrainErrorStoppingCriterion(.1)
    >>> xor_sample = [LabeledExample((1, 1), (0,)),
    ... LabeledExample((0, 0), (0,)), LabeledExample((0, 1), (1,)),
    ... LabeledExample((1, 0), (1,))]
    >>> idiot = IdiotAlgorithm(xor_sample, stopping_criterion=f,
    ... model=ConstantModel((1,)), verbose=False)
    >>> f.register_learning_algorithm(idiot)
    >>> [f.stop() for i in range(13)] #doctest: +NORMALIZE_WHITESPACE
    [False, False, False, False, False, False, False, False, False, False,
    False, False, False]

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, max_error=0.1, error_model=MSE(), max_iterations=5000):
        r"""
        See :class:`TrainErrorStoppingCriterions` for full documentation.

        """

        TestErrorStoppingCriterion.__init__(self, None, max_error,
            error_model, max_iterations)

    def __repr__(self):
        result = 'TrainErrorStoppingCriterion('
        if self.max_error != 0.1:
            result += str(self.max_error) + ', '
        if self.error_model != MSE():
            result += str(self.error_model) + ', '
        if result[-2:] == ', ':
            result = result[:-2]
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other) and \
            self.max_error == other.max_error \
                and self.error_model == other.error_model

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("TrainErrorStoppingCriterion",
            hash(self.max_error), hash(self.error_model)))

    def __nonzero__(self):
        return True

    def register_learning_algorithm(self, alg):
        r"""
        Register a learning algorithm as instance variable of the stopping
        criterion. In this way the criterion is able to access to meaningful
        information for its run-time behaviour. In particular, this class
        overrides the base method inherited from :class:`StoppingCriterion` for
        it also needs to access the learning algorithm training sample.

        Note that learning algorithms whose execution is based on an iterative
        process whose termination relies on a stopping criterion should
        subclass :class:`yaplf.algorithms.IterativeLearningAlgorithm`, which
        automatically registers on its stopping criterion through invocation of
        this method.

        :param alg: -- learning algorithms to be registered.

        :type alg: :class:`yaplf.algorithms.LearningAlgorithm`

        EXAMPLES:

        See the examples for the base class :class:`StoppingCriterion`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.learning_algorithm = alg
        self.test_set = alg.sample
