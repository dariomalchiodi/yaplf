
r"""
Package handling uncertain SV classification learning algorithms in yaplf.

Package yaplf.algorithms.svm.classification.uncertain contains all the classes
handling uncertain SV classification learning algorithms in yaplf.

TODO

- pep8 check
- pylint check

AUTHORS:

- Dario Malchiodi (2013-09-04): initial version.


"""

#*****************************************************************************
#       Copyright (C) 2013 Dario Malchiodi <malchiodi@di.unimi.it>
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


# from numpy import mean

from yaplf.algorithms.svm.classification import SVMClassificationAlgorithm
from yaplf.algorithms.svm.classification.uncertain.solvers import GurobiUncertainClassificationSolver
# from yaplf.models.kernel import LinearKernel
# from yaplf.models.svm import SVMClassifier, check_svm_classification_sample
# from yaplf.algorithms.svm.solvers import PyMLClassificationSolver


class SVMUncertainClassificationAlgorithm(SVMClassificationAlgorithm):
    r"""
    SVM Uncertain Classification Algorithm.

    INPUT:

    - ``labeled_sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``unlabeled_sample`` -- list or tuple of ``Example`` instances

    - ``c_value`` -- float (default: None, amounting to the hard-margin version
      of the algorithm) value for the trade-off constant `C` between steepness
      and accuracy in the soft-margin version of the algorithm.

    - ``kernel`` -- ``Kernel`` (default: ``LinearKernel()``) instance defining
      the kernel to be used.

    - ``solver`` -- ``SVMUncertainClassificationSolver`` (default:
      ``GurobiUncertainClassificationSolver()``, unless differently specified
      through the ``SVMUncertainClassificationAlgorithm.default_solver`` class
      field) solver to be used in order to find the solution of the SV
      uncertain classification optimization problem.

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

    def __init__(self, labeled_sample, unlabeled_sample, **kwargs):
        r"""
        See ``SVMClassificationAlgorithm`` for full documentation.

        """

        SVMClassificationAlgorithm.__init__(self, labeled_sample, **kwargs)

        self.unlabeled_sample = unlabeled_sample

        try:
            self.solver = kwargs['solver']
        except KeyError:
            self.solver = SVMUncertainClassificationAlgorithm.default_solver

    def get_classifier(self, optimal_values, **kwargs):
        alphas, gammas, deltas = optimal_values
        patterns = [e.pattern for e in self.sample]
        labels = [e.label for e in self.sample]

        unlabeled_patterns = self.unlabeled_sample

        if not(len(patterns) == len(labels) == len(alphas)):
            raise ValueError('patterns, labels and optimal alphas have different length')

        bs = numpy.array([labels[i] -
                          sum([alphas[j]*labels[j]*self.kernel.compute(patterns[j],
                                                                       patterns[i])
                               for j in range(len(alphas))]) -
                          sum([(gammas[s]-deltas[s])*self.kernel.compute(
                                                                unlabeled_patterns[s],
                                                                patterns[i])
                               for s in range(len(gammas))])
                          for i in range(len(alphas)) if 0 < alphas[i] < c])

        try:
            b_var_threshold = kwargs['b_var_threshold']
        except KeyError:
            b_var_threshold = 1e-4

        b_mean = bs.mean()
        b_var = bs.var()

        if b_var > b_var_threshold:
            print 'Variance on b values [%s] is %f and exceeds threshold %f' % (', '.join(map(str, bs)), b_var, b_var_threshold)

        def real_classifier(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns):
            return classifier

        def binary_classifier(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns):
            return lambda input: 1 if classifier(input) >= 0 else -1

        def lagrange_values(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns):
            return (alphas, gammas, deltas)

        def epsilon_value(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns):
            alpha_y = [a*y for a, y in zip(alphas, labels)]

            first = numpy.array([-1 * numpy.dot([a*y for a, y in zip(alphas, labels)], map(lambda x: k.compute(x, unlabeled_patterns[s]), patterns)) for s in range(len(gammas)) if gammas[s] > 0])
            second = numpy.array([-1 * numpy.dot(numpy.array(gammas) - numpy.array(deltas), map(lambda x: k.compute(x, unlabeled_patterns[s]), unlabeled_patterns)) for s in range(len(gammas)) if gammas[s] > 0])
            eps_gammas = numpy.array([e - b_mean for e in first-second])

            first = numpy.array([numpy.dot([a*y for a, y in zip(alphas, labels)], map(lambda x: k.compute(x, unlabeled_patterns[s]), patterns)) for s in range(len(deltas)) if deltas[s] > 0])
            second = numpy.array([numpy.dot(numpy.array(gammas) - numpy.array(deltas), map(lambda x: k.compute(x, unlabeled_patterns[s]), unlabeled_patterns)) for s in range(len(deltas)) if deltas[s] > 0])
            eps_deltas = numpy.array([e + b_mean for e in first+second])

            return numpy.concatenate((eps_gammas, eps_deltas)).mean()

        decorator = {'real_classifier': real_classifier, \
                     'binary_classifier': binary_classifier, \
                     'lagrange_values': lagrange_values, 'epsilon_value': epsilon_value}

        try:
            output = kwargs['output']
        except KeyError:
            output = 'real_classifier'

        classifier = lambda input: numpy.dot([alphas[i]*labels[i] for i in range(len(patterns))],[k.compute(p, input) for p in patterns]) + \
            numpy.dot(numpy.array(gammas) - numpy.array(deltas), [k.compute(u, input) for u in unlabeled_patterns]) + \
            b_mean

        output = decorator[output]

        if callable(output):
            return output(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns)
        else:
            return map(lambda o: o(classifier, alphas, gammas, deltas, patterns, labels, unlabeled_patterns), output)



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

        



    def run(self, **kwargs):


        optimal_values = self.solver.solve(self.sample, self.unlabeled_sample, self.c, self.kernel, **kwargs)
        self.model = self.get_classifier(optimal_values, **kwargs)



# Default solvers
SVMUncertainClassificationAlgorithm.default_solver = GurobiUncertainClassificationSolver()


