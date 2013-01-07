
r"""
Package handling SV classification learning algorithms in yaplf.

Package yaplf.algorithms.svm.classification contains all the classes handling
SV classification learning algorithms in yaplf.

- pep8 checked
- pylint score: 9.63

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version.

- Dario Malchiodi (2010-04-06): added the customizable default solver
  mechanism in ``SVMClassificationAlgorithm``.

- Dario Malchiodi (2010-04-12): added ``SVMVQClassificationAlgorithm``.

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************


from numpy import mean

from yaplf.algorithms import LearningAlgorithm
from yaplf.models.kernel import LinearKernel
from yaplf.models.svm import SVMClassifier, check_svm_classification_sample
from yaplf.algorithms.svm.solvers import PyMLClassificationSolver, \
    CVXOPTVQClassificationSolver


class SVMClassificationAlgorithm(LearningAlgorithm):
    r"""
    SVM Classification Algorithm.

    INPUT:

    - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``c_value`` -- float (default: None, amounting to the hard-margin version
      of the algorithm) value for the trade-off constant `C` between steepness
      and accuracy in the soft-margin version of the algorithm.

    - ``kernel`` -- ``Kernel`` (default: ``LinearKernel()``) instance defining
      the kernel to be used.

    - ``solver`` -- ``SVMClassificationSolver`` (default:
      ``CVXOPTClassificationSolver()``, unless differently specified through
      the ``SVMClassificationAlgorithm.default_solver`` class field) solver to
      be used in order to find the solution of the SV classification
      optimization problem.

    OUTPUT:

    ``LearningAlgorithm`` instance.

    EXAMPLES:

    SV classification algorithm can be directly applied to any problem whose
    labels are encodable in terms of two classes. For instance, consider the
    binary XOR problem:

    ::

        >>> from yaplf.data import LabeledExample
        >>> xor_sample = [LabeledExample((1, 1), -1),
        ... LabeledExample((0, 0), -1), LabeledExample((0, 1), 1),
        ... LabeledExample((1, 0), 1)]

    As this sample is not linearly separable, a nonlinear kernel is needed in
    order to learn it, for instance a polynomial kernel:

    ::

        >>> from yaplf.algorithms.svm.classification \
        ... import SVMClassificationAlgorithm
        ...
        >>> from yaplf.models.kernel import PolynomialKernel
        >>> alg = SVMClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(2))

    Running the algorithm and subsequently accessing to its ``model`` field
    allows to get the learnt SV classifier:

    ::

        >>> alg.run()
        >>> alg.model
        SVMClassifier([1.9999999995368085, 3.3333333325135617,
        2.6666666660251854, 2.6666666660251854], -0.999999999697,
        [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
        LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)], kernel =
        PolynomialKernel(2))

    The latter can be tested on the original sample in order to verify that
    learning succeeded:

    ::

        >>> alg.model.test(xor_sample)
        0.0

    As an aside remark, it should be highlighted that these examples are shown
    for illustrative purpose. The suitable way of assessing a learnt model
    performance involves more complex techniques involving the use of a test
    set possibly coupled with a cross validation procedure (see function
    ``cross_validation`` in package ``yaplf.utility.validation``.)

    Finally, it is worth noting that this class invokes under the hood a
    solver specialized in finding the solution of a quadratic constrained
    optimization problem. This solver is available in various flavours,
    corresponding to specific subclasses of ``SVMClassificationSolver``, all
    defined in package ``yaplf.algorithms.svm.solvers``. Currently, the
    following solvers are available:

    - ``CVXOPTClassificationSolver``, the default choice, solves generic
      quadratic problems;

    - ``PyMLClassificationSolver`` is tailored on the specific optimization
      problem linked to the SV classification task.

    A specific solver can be selected using the ``solver`` named argument when
    creating an instance of ``SVMClassificationAlgorithm``:

    ::

        >>> from yaplf.algorithms.svm.solvers \
        ... import PyMLClassificationSolver
        >>> alg = SVMClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(2), solver = PyMLClassificationSolver())
        >>> alg.run()
        ...
        >>> alg.model
        SVMClassifier([1.9956464353279999, 3.3260773922133327,
        2.6594107255466666, 2.6623131019946662], -0.997823217664,
        [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
        LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)], kernel =
        PolynomialKernel(2))

    Note how the results slightly change when using different solvers.

    The default solver can be modified through the ``default_solver`` class
    variable:

    ::

        >>> SVMClassificationAlgorithm.default_solver = \
        ... PyMLClassificationSolver()
        >>> alg = SVMClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(2))
        >>> alg.run()
        ...
        >>> alg.model
        SVMClassifier([1.9956464353279999, 3.3260773922133327,
        2.6594107255466666, 2.6623131019946662], -0.997823217664,
        [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
        LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)], kernel =
        PolynomialKernel(2))

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    - Dario Malchiodi (2010-04-06): added customizable default solver.

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``SVMClassificationAlgorithm`` for full documentation.

        """

        LearningAlgorithm.__init__(self, sample)
        check_svm_classification_sample(sample)
        self.sample = sample
        self.model = None

        try:
            self.c_value = kwargs['c_value']
        except KeyError:
            self.c_value = None

        try:
            self.kernel = kwargs['kernel']
        except KeyError:
            self.kernel = LinearKernel()

        try:
            self.solver = kwargs['solver']
        except KeyError:
            #self.solver = CVXOPTClassificationSolver(*args, **kwargs)
            self.solver = SVMClassificationAlgorithm.default_solver

    def run(self):
        r"""
        Run the SVM classification learning algorithm.

        INPUT:

        No input.

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``SVMClassifier`` instance.

        EXAMPLES:

        Consider the following sample describing the binary XOR function, and a
        ``SVMClassificationAlgorithm`` instance dealing with the corresponding
        learning problem:

        ::

            >>> from yaplf.data import LabeledExample
            >>> xor_sample = [LabeledExample((1, 1), -1),
            ... LabeledExample((0, 0), -1), LabeledExample((0, 1), 1),
            ... LabeledExample((1, 0), 1)]
            >>> from yaplf.algorithms.svm.classification \
            ... import SVMClassificationAlgorithm
            >>> from yaplf.models.kernel import PolynomialKernel
            >>> alg = SVMClassificationAlgorithm(xor_sample,
            ... kernel = PolynomialKernel(2))

        Running the algorithm and subsequently accessing to its ``model`` field
        allows to get the learnt SV classifier:

        ::

            >>> alg.run()
            >>> alg.model
            SVMClassifier([1.9999999995368085, 3.3333333325135617,
            2.6666666660251854, 2.6666666660251854], -0.999999999697,
            [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
            LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)], kernel =
            PolynomialKernel(2))

        The latter can be tested on the original sample in order to verify that
        learning succeeded:

        ::

            >>> alg.model.test(xor_sample)
            0.0

        As a final remark, it should be highlighted that these examples are
        shown for illustrative purpose. The suitable way of assessing a learnt
        model performance involves more complex techniques involving the use of
        a test set possibly coupled with a cross validation procedure (see
        function ``cross_validation`` in package ``yaplf.utility``.)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        alpha = self.solver.solve(self.sample, self.c_value, self.kernel)
        num_examples = len(self.sample)

        if self.c_value == None:
            threshold = mean([self.sample[i].label -
                sum([alpha[j] * self.sample[j].label *
                self.kernel.compute(self.sample[j].pattern,
                self.sample[i].pattern)
                for j in range(num_examples)]) for i in range(num_examples)
                if alpha[i] > 0])
        else:
            threshold = mean([self.sample[i].label -
                sum([alpha[j] * self.sample[j].label *
                self.kernel.compute(self.sample[j].pattern,
                self.sample[i].pattern) for j in range(num_examples)])
                for i in range(num_examples)
                if alpha[i] > 0 and alpha[i] < self.c_value])

        self.model = SVMClassifier(alpha, threshold, self.sample,
            kernel=self.kernel)


class SVMVQClassificationAlgorithm(LearningAlgorithm):
    r"""
    SVM Classification Algorithm for data of variable quality, as described in
    [Apolloni et al., 2007].

    INPUT:

    - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``c_value`` -- float (default: None, amounting to the hard-margin version
      of the algorithm) value for the trade-off constant `C` between steepness
      and accuracy in the soft-margin version of the algorithm.

    - ``kernel`` -- ``Kernel`` (default: ``LinearKernel()``) instance defining
      the kernel to be used.

    OUTPUT:

    ``LearningAlgorithm`` instance.

    EXAMPLES:

    Variable-quality SV classification algorithm can be directly applied to any
    problem whose single examples are explicitly associated to a numerical
    evaluation of their quality. For instance, consider the following extension
    of the binary XOR problem, whose upper-left example is given higher
    importance:

    ::

        >>> from yaplf.data import LabeledExample, AccuracyExample
        >>> xor_sample = [AccuracyExample(LabeledExample((1, 1), -1), 0),
        ... AccuracyExample(LabeledExample((0, 0), -1), 0),
        ... AccuracyExample(LabeledExample((0, 1), 1), .3),
        ... AccuracyExample(LabeledExample((1, 0), 1), 0)]

    Training a SV classifier on this sample using a polynomial kernel brings to
    the following model:

    ::

        >>> from yaplf.algorithms.svm.classification \
        ... import SVMVQClassificationAlgorithm
        >>> from yaplf.models.kernel import PolynomialKernel
        >>> alg = SVMVQClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(4))
        >>> alg.run()
        >>> alg.model
        SVMClassifier([0.092013250246483894, 0.39872408400362103,
        0.25518341380777959, 0.23555392044232534], -1.1501656251,
        [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
        LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)],
        kernel = PolynomialKernel(4))

    The actual exploitation of the additional information about data quality
    can be tested plotting the SV classifier decision function and noting that
    the class boundary is farther from the upper-left example than from the
    other ones:

    ::

        >>> alg.model.plot((0, 1), (0, 1), shading = True)

    REFERENCES:

    [Apolloni et al., 2007] B. Apolloni, D. Malchiodi and L. Natali, A Modified
    SVM Classification Algorithm for Data of Variable Quality, in
    Knowledge-Based Intelligent Information and Engineering Systems 11th
    International Conference, KES 2007, XVII Italian Workshop on Neural
    Networks, Vietri sul Mare, Italy, September 12-14, 2007. Proceedings, Part
    III, Berlin Heidelberg: Springer-Verlag, Lecture Notes in Artificial
    Intelligence 4694 (ISBN 978-3-540-74828-1), 131-139, 2007

    AUTHORS:

    - Dario Malchiodi (2010-04-12)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``SVMVQClassificationAlgorithm`` for full documentation.

        """

        red_sample = [a.example for a in sample]
        LearningAlgorithm.__init__(self, red_sample)
        check_svm_classification_sample(red_sample)
        self.sample = sample
        self.model = None

        try:
            self.c_value = kwargs['c_value']
        except KeyError:
            self.c_value = None

        try:
            self.kernel = kwargs['kernel']
        except KeyError:
            self.kernel = LinearKernel()

        self.solver = CVXOPTVQClassificationSolver()

    def run(self):
        r"""
        Run the variable-quality SVM classification learning algorithm.

        INPUT:

        No input.

        OUTPUT:

        No output. After the invocation the inferred model is available through
        the ``model`` field, in form of a ``SVMClassifier`` instance.

        EXAMPLES:

        Variable-quality SV classification algorithm can be directly applied to
        any problem whose single examples are explicitly associated to a
        numerical evaluation of their quality. For instance, consider the
        following extension of the binary XOR problem, whose upper-left example
        is given higher importance:

        ::

            >>> from yaplf.data import LabeledExample, AccuracyExample
            >>> xor_sample = [AccuracyExample(LabeledExample((1, 1), -1), 0),
            ... AccuracyExample(LabeledExample((0, 0), -1), 0),
            ... AccuracyExample(LabeledExample((0, 1), 1), .3),
            ... AccuracyExample(LabeledExample((1, 0), 1), 0)]

        Training a SV classifier on this sample using a polynomial kernel
        brings to the following model:

        ::

            >>> from yaplf.algorithms.svm.classification \
            ... import SVMVQClassificationAlgorithm
            >>> from yaplf.models.kernel import PolynomialKernel
            >>> alg = SVMVQClassificationAlgorithm(xor_sample,
            ... kernel = PolynomialKernel(4))
            >>> alg.run()
            >>> alg.model
            SVMClassifier([0.092013250246483894, 0.39872408400362103,
            0.25518341380777959, 0.23555392044232534], -1.1501656251,
            [LabeledExample((1, 1), -1.0), LabeledExample((0, 0), -1.0),
            LabeledExample((0, 1), 1.0), LabeledExample((1, 0), 1.0)],
            kernel = PolynomialKernel(4))

        The actual exploitation of the additional information about data
        quality can be tested plotting the SV classifier decision function and
        noting that the class boundary is farther from the upper-left example
        than from the other ones:

        ::

            >>> alg.model.plot((0, 1), (0, 1), shading = True)

        AUTHORS:

        - Dario Malchiodi (2010-04-12)

        """

        alpha = self.solver.solve(self.sample, self.c_value, self.kernel)
        num_examples = len(self.sample)

        if self.c_value == None:
            threshold = mean([self.sample[i].example.label -
                sum([alpha[j] * self.sample[j].example.label *
                self.kernel.compute(self.sample[j].example.pattern,
                self.sample[i].example.pattern)
                for j in range(num_examples)]) for i in range(num_examples)
                if alpha[i] > 0])
        else:
            threshold = mean([self.sample[i].example.label -
                sum([alpha[j] * self.sample[j].example.label *
                self.kernel.compute(self.sample[j].example.pattern,
                self.sample[i].example.pattern) for j in range(num_examples)])
                for i in range(num_examples)
                if alpha[i] > 0 and alpha[i] < self.c_value])

        self.model = SVMClassifier(alpha, threshold, [elem.example \
            for elem in self.sample], kernel=self.kernel)


# Default solvers
SVMClassificationAlgorithm.default_solver = PyMLClassificationSolver()
