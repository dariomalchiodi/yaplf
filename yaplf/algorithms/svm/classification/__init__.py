
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

- Dario Malchiodi (2013-09-04): moved variable-quality classification
  algorithms in a separate module.


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


from numpy import mean

from yaplf.algorithms import LearningAlgorithm
from yaplf.models.kernel import LinearKernel
from yaplf.models.svm import SVMClassifier, check_svm_classification_sample
from yaplf.algorithms.svm.classification.solvers import GurobiClassificationSolver


class SVMClassificationAlgorithm(LearningAlgorithm):
    r"""
    SVM Classification Algorithm.

    INPUT:

    - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``c`` -- float (default: None, amounting to the hard-margin version
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
        >>> from yaplf.models.kernel import PolynomialKernel
        >>> alg = SVMClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(2))

    Running the algorithm and subsequently accessing to its ``model`` field
    allows to get the learnt SV classifier:

    ::

        >>> alg.run()
        >>> alg.model
        SVMClassifier([2.000000000043493, 3.3333333334006983,
        2.6666666667220955, 2.6666666667220955], -1.00000000001,
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

        >>> from yaplf.algorithms.svm.classification.solvers \
        ... import PyMLClassificationSolver
        >>> alg = SVMClassificationAlgorithm(xor_sample,
        ... kernel = PolynomialKernel(2), solver = PyMLClassificationSolver())
        >>> alg.run() # doctest:+ELLIPSIS
        Cpos, Cneg...
        >>> print alg.model
        SVMClassifier([2.000000000030325, 3.3333333333791955,
        2.6666666667061474, 2.666666666703373], -1.00000000001,
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
        >>> alg.run() # doctest: +ELLIPSIS
        Cpos, Cneg...
        >>> print alg.model
        SVMClassifier([2.0000000000434928, 3.333333333400698,
        2.6666666667220946, 2.6666666667220946], -1.00000000001,
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
            self.c = kwargs['c']
        except KeyError:
            self.c = float('inf')

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

            >>> alg.run() # doctest: +ELLIPSIS
            Cpos, Cneg...
            >>> alg.model
            SVMClassifier([2.0000000000434928, 3.333333333400698,
            2.6666666667220946, 2.6666666667220946], -1.00000000001,
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

        alpha = self.solver.solve(self.sample, self.c, self.kernel)

        num_examples = len(self.sample)

        if self.c == float('inf'):
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
                if alpha[i] > 0 and alpha[i] < self.c])

        self.model = SVMClassifier(alpha, threshold, self.sample,
            kernel=self.kernel)




# Default solvers
SVMClassificationAlgorithm.default_solver = GurobiClassificationSolver()
