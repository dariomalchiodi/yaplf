
r"""
Package handling algorithms in yaplf

Package yaplf.algorithms contains all the basic classes handling learning
algorithms in yaplf.

TODO:

- pep8 checked
- pylint score: 9.62

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


from threading import Thread

from yaplf.models import ConstantModel
from yaplf.utility import Observable
from yaplf.utility.stopping import FixedIterationsStoppingCriterion
from yaplf.utility.selection import sequential_selector


class LearningAlgorithm(Thread):
    r"""
    Base class for learning algorithms. Each subclass should implement
    a constructor accepting as argument a sample of data, a ``run`` method
    running the algorithm (and optionally a ``reset`` method whose invocation
    resets subsequent runs), and a model field containing the learnt model
    resulting by invocations of ``run``.

    ``LearningAlgorithm`` subclasses ``Thread``, so that an algorithm can be
    executed either sequentially invoking ``run`` or concurrently invoking
    ``start``.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``Example`` object containing the train
      set in input to the learning algorithm.

    OUTPUT:

    A ``LearningAlgorithm`` object.

    EXAMPLES:

    See examples for concrete subclasses such as
    ``GradientPerceptronLearningAlgorithm`` in package yaplf.algorithms.neural.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample):
        r"""
        See ``LearningAlgorithm`` for full documentation.

        """

        Thread.__init__(self)
        self.sample = sample
        self.model = None

    def reset(self):
        r"""
        Resets the learning algorithm. A subsequent invocation of ``run``
        will start from fresh.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        OUTPUT:

        No output.

        EXAMPLES:

        See examples for concrete subclasses such as ``IdiotAlgorithm`` in this
        package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        self.model = None

    def run(self):
        r"""
        Runs the learning algorithm, possibly in function of the specified
        arguments.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        OUTPUT:

        No output.

        EXAMPLES:

        See examples for concrete subclasses such as ``IdiotAlgorithm`` in this
        package.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        pass


class IterativeAlgorithm(LearningAlgorithm, Observable):
    r"""
    Base class for iterative learning algorithms. Besides the feature
    inherited from ``LearningAlgorithm``, iterative learning algorithms are
    observable (typically in the aim of generating graphics summarizing their
    behaviour) and use an example selector and a stopping criterion.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``Example`` object containing the train
      set in input to the learning algorithm.

    OUTPUT:

    An ``IterativeLearningAlgorithm`` object.

    EXAMPLES:

    See examples for concrete subclasses such as
    ``GradientPerceptronLearningAlgorithm`` in package yaplf.algorithms.neural.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample):
        r"""
        See ``IterativeLearningAlgorithm`` for full documentation.

        """

        LearningAlgorithm.__init__(self, sample)
        Observable.__init__(self)
        self.sample = sample
        #This calms down pylint. The actual values will be set in run
        self.stop_criterion = None
        self.sample_selector = None

    def run(self, **kwargs):
        r"""
        Run the learning algorithm.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        - ``stopping_criterion`` -- ``StoppingCriterion`` object (default:
          ``FixedIterationsStoppingCriterion()``) to be used in order to decide
          when learning should be stopped.

        - ``selector`` -- iterator (default: ``sequential_selector``) yielding
          the examples to be fed to the learning algorithm.

        OUTPUT:

        No output.

        EXAMPLES:

        See examples for concrete subclasses such as
        ``GradientPerceptronLearningAlgorithm`` in package
        yaplf.algorithms.neural.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            self.stop_criterion = kwargs['stopping_criterion']
            self.stop_criterion.reset()
        except KeyError:
            self.stop_criterion = FixedIterationsStoppingCriterion()
        self.stop_criterion.register_learning_algorithm(self)

        try:
            # unit selection
            selector = kwargs['selector']
        except KeyError:
            selector = sequential_selector
        self.sample_selector = selector(self.sample)


class IdiotAlgorithm(LearningAlgorithm):
    r"""
    Learning algorithm actually not learning the specified sample as it
    always outputs ConstantModel(0). As it can print out all parameters
    specified when invoking init, reset and run, it serves for testing
    purposes.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``sample`` -- list or tuple of ``Example`` objects.

    - ``model`` -- constant model in output of the algorithm

    - ``verbose`` -- boolean (default: ``True``) flag triggering verbose
      output.

    OUTPUT:

    ``ConstantModel(0)``, or the model specified, regardless of the fed sample.

    EXAMPLES:

    An ``IdiotAlgorithm`` object is typically used for testing purpose. On the
    one hand it can be used when it is necessary to use a learning algorithm
    but no assumptions are to be done on the particular inferred model. On the
    other one, the class has verbose behaviour by default: on its instantiation
    as well as on each run of the algorithm all the specified parameters will
    be printed on screen:

    ::

        >>> from yaplf.algorithms import IdiotAlgorithm
        >>> alg = IdiotAlgorithm((1, 2), 3, name='ok')
        Initialization with sample (1, 2)
        fixed arguments (3,)
        named arguments {'name': 'ok'}
        reset invocation with fixed arguments ()
        named arguments {}
        >>> alg.run(5, -1, parameter = 9)
        run invocation with fixed arguments (5, -1)
        named arguments {'parameter': 9}
        >>> alg.model
        ConstantModel(0)

    Verbosity can be switched off through the ``verbose`` named argument:

    ::

        >>> from yaplf.algorithms import IdiotAlgorithm
        >>> alg = IdiotAlgorithm((1, 2), 3, name='ok', verbose = False)
        >>> alg.run(5, -1, parameter = 9)
        >>> alg.model
        ConstantModel(0)

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sample, *args, **kwargs):
        r"""
        See ``IdiotAlgorithm`` for full documentation.

        """

        LearningAlgorithm.__init__(self, sample)
        try:
            self.verbose = kwargs['verbose']
        except KeyError:
            self.verbose = True

        if self.verbose:
            print 'Initialization with sample ' + str(sample)
            print 'fixed arguments ' + str(args)
            print 'named arguments ' + str(kwargs)

        try:
            self.model = kwargs['model']
        except KeyError:
            self.model = ConstantModel(0)

        self.reset()

    def reset(self, *args, **kwargs):
        r"""
        Reset the IdiotAlgorithm, that is it does nothing except
        printing out the fixed and named arguments specified.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        OUTPUT:

        No output.

        EXAMPLES:

        This method is typically called in order to reset a learning algorithm
        run, so as to be able to perform a fresh start. As within
        ``IdiotAlgorithm`` there are no actual computations, this method does
        nothing except printing a verbose output (except otherwise stated
        through the ``verbose`` field).

        ::

            >>> from yaplf.algorithms import IdiotAlgorithm
            >>> alg = IdiotAlgorithm((1, 2), 3, name='ok')
            Initialization with sample (1, 2)
            fixed arguments (3,)
            named arguments {'name': 'ok'}
            reset invocation with fixed arguments ()
            named arguments {}
            >>> alg.reset(7, 1, name = 'yes')
            reset invocation with fixed arguments (7, 1)
            named arguments {'name': 'yes'}
            >>> alg.verbose = False
            >>> alg.reset(7, 1, name = 'yes')

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if self.verbose:
            print 'reset invocation with fixed arguments ' + str(args)
            print 'named arguments ' + str(kwargs)

    def run(self, *args, **kwargs):
        r"""
        Run the IdiotAlgorithm, that is it does nothing except
        printing out the fixed and named arguments specified.

        INPUT:

        - ``self`` -- object on which the function is invoked.

        OUTPUT:

        No output.

        EXAMPLES:

        This method is typically called in order to start a learning algorithm
        run. As within ``IdiotAlgorithm`` there are no actual computations,
        this method does nothing except printing a verbose output (except
        otherwise stated through the ``verbose`` field).

        ::

            >>> from yaplf.algorithms import IdiotAlgorithm
            >>> alg = IdiotAlgorithm((1, 2), 3, name='ok')
            Initialization with sample (1, 2)
            fixed arguments (3,)
            named arguments {'name': 'ok'}
            reset invocation with fixed arguments ()
            named arguments {}
            >>> alg.run(5, -1, parameter = 9)
            run invocation with fixed arguments (5, -1)
            named arguments {'parameter': 9}
            >>> alg.verbose = False
            >>> alg.run(5, -1, parameter = 9)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if self.verbose:
            print 'run invocation with fixed arguments ' + str(args)
            print 'named arguments ' + str(kwargs)
            print
