r"""
Package handling variable-quality SV classification learning algorithms in
yaplf.

Package yaplf.algorithms.svm.classification.vq contains all the classes
handling variable-quality SV classification learning algorithms in yaplf.

- pep8 checked
- pylint score: 9.63

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

from yaplf.algorithms.svm.classification import SVMClassificationAlgorithm
from yaplf.algorithms.svm.classification.VQ.solvers import GurobiVQClassificationSolver

from yaplf.models.svm import SVMClassifier

import numpy

class SVMVQClassificationAlgorithm(SVMClassificationAlgorithm):
    r"""
    SVM Classification Algorithm for data of variable quality, as described in
    [Apolloni et al., 2007].

    INPUT:

    - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
      labels are all set either to `1` or `-1`.

    - ``c`` -- float (default: None, amounting to the hard-margin version
      of the algorithm) value for the trade-off constant `C` between steepness
      and accuracy in the soft-margin version of the algorithm.

    - ``kernel`` -- ``Kernel`` (default: ``LinearKernel()``) instance defining
      the kernel to be used.

    - ``solver`` -- ``SVMVQClassificationSolver`` (default:
      ``GurobiVQClassificationSolver()``, unless differently specified through
      the ``SVMVQClassificationAlgorithm.default_solver`` class field) solver to
      be used in order to find the solution of the SV classification
      optimization problem.

    OUTPUT:

    ``SVMVQClassificationAlgorithm`` instance.

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

    - Dario Malchiodi (2014-03-03)

    """

    def __init__(self, sample, **kwargs):
        r"""
        See ``SVMVQClassificationAlgorithm`` for full documentation.

        """

        standard_sample = [a.example for a in sample]

        SVMClassificationAlgorithm.__init__(self, standard_sample, **kwargs)

        self.sample = sample
        self.model = None

        try:
            self.solver = kwargs['solver']
        except KeyError:
            self.solver = SVMVQClassificationAlgorithm.default_solver



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

        alpha = self.solver.solve(self.sample, self.c, self.kernel)
        
        num_examples = len(self.sample)
        
        accuracy = numpy.array([e.accuracy for e in self.sample])
        label = numpy.array([e.example.label for e in self.sample])
        pattern = numpy.array([e.example.pattern for e in self.sample])
        
        double_sum = sum([alpha[i] * alpha[j] * label[i] * label[j] *
            self.kernel.compute(pattern[i], pattern[j])
            for i in range(num_examples) for j in range(num_examples)])

        if self.c is None:
            indices = [i for i in range(num_examples) if alpha[i] > 0]
        else:
            indices = [i for i in range(num_examples)
                if alpha[i] > 0 and alpha[i] < self.c]

        denominator = 1 + numpy.dot(alpha, accuracy)

        threshold = numpy.mean([
            label[j] -
            sum([alpha[j] * label[j] * self.kernel.compute(pattern[i], pattern[j])
                for i in range(num_examples)])/denominator +
            label[j]*accuracy[j]*double_sum/(2*denominator**2)
        for j in indices])

        self.model = SVMClassifier(alpha/denominator, threshold, [elem.example \
            for elem in self.sample], kernel=self.kernel)


# Default solvers
SVMVQClassificationAlgorithm.default_solver = GurobiVQClassificationSolver()
