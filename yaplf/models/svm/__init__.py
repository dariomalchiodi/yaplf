
r"""
Package handling support vector-based models in yaplf

Package yaplf.models.svm contains all the classes handling SV-based models in
yaplf.

TODO:

- Regression
- Quality-based SVC
- Quality-based SVR
- pep8 checked
- pylint score: 9.22

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


from numpy import sign, dot
#from matplotlib.cm import Greys

from yaplf.models import Classifier
from yaplf.models.kernel import Kernel
from yaplf.data import LabeledExample


def check_svm_classification_sample(sample):
    r"""
    Checks whether the supplied sample is properly formatted in order to use it
    for SVM classification, raising a specialized ValueError otherwise. The
    performed check requires that all patterns have the same dimension, while
    labels should be either set to 1 or to -1.

    INPUT:

    - sample -- iterable containing a sample to be checked.

    OUTPUT:

    No output. Raises a ValueError if the sample is not suitable to build a SVM
    classifier, otherwise returns silently.

    EXAMPLES:

    Any sample given as input to ``check_svm_classification_sample`` is checked
    against the following properties:

    - the sample should be iterable;

    - each sample element should have a ``pattern`` and a ``label`` field, that
      is it should be an instance of the ``LabeledExample`` class;

    - ``pattern`` fields of all sample elements should have the same dimension;

    - ``label`` fields of all sample elements should either be equal to `1` or
      to `-1`.

    Patterns of unequal length in a sample cause a ValueError to be thrown:

    ::
        >>> from yaplf.models.svm import check_svm_classification_sample
        >>> from yaplf.data import LabeledExample
        >>> wrong_sample = (LabeledExample((1, 0), 1),
        ... LabeledExample((1, 0, 1), 1))
        >>> check_svm_classification_sample(wrong_sample)
        Traceback (most recent call last):
            ...
        ValueError: SVM classification patterns should have the same dimension

    The same error is thrown, although with a different message, when any label
    is neither set to `1` or `-1`:

    ::

        >>> xor_sample = (LabeledExample((0., 0.), 0.),
        ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), 0.))
        >>> check_svm_classification_sample(xor_sample)
        Traceback (most recent call last):
            ...
        ValueError: SVM classification labels should be set either to -1 or to
        1

    When all properties are met the function returns silently

    ::

        >>> xor_sample = (LabeledExample((0., 0.), -1),
        ... LabeledExample((1., 0.), 1), LabeledExample((0., 1.), 1),
        ... LabeledExample((1., 1.), -1))
        >>> check_svm_classification_sample(xor_sample)

    Note that equality w.r.t. `1` and `-1` is type insensitive, that is ``1.0``
    and ``-1.0`` are legal values, too:

    ::

        >>> xor_sample = (LabeledExample((0., 0.), -1),
        ... LabeledExample((1., 0.), 1.0), LabeledExample((0., 1.), 1),
        ... LabeledExample((1., 1.), -1.0))
        >>> check_svm_classification_sample(xor_sample)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

    """

    dim = len(sample[0].pattern)

    if abs(sample[0].label) != 1:
        raise ValueError('SVM classification labels should be set either to \
-1 or to 1')

    for elem in sample[1:]:
        if len(elem.pattern) != dim:
            raise ValueError('SVM classification patterns should have the \
same dimension')
        if abs(elem.label) != 1:
            raise ValueError('SVM classification labels should be set either \
to -1 or to 1')


class SVMClassifier(Classifier):
    r"""
    Class implementing the Support Vector Classifier (SVC for short), in the
    version originally introduced by [Cortes and Vapnik, 1995]. This model
    depends inherently on a subset of a given sample `\left\{ (x_1, y_1),
    \dots, (x_m, y_m) \right\} \subset X \cup \{ -1, 1 \}`, where `X` is a
    suitable space, as well as on set of *weights* `\left\{ \alpha_1, \dots,
    \alpha_m \right\}` (one weight for each sample item), on a *threshold `b
    \in \mathbb R`, and on a *kernel function* `k: X^2 \mapsto \mathbb R`.
    Precisely, when presented a generic pattern `x \in X` it outputs the sign
    of `\sum_{i=1}^m \alpha_i y_i k(x_i, x) + b`.

    INPUT:

    - ``alpha`` -- iterable containing the SVC weights.

    - ``threshold`` -- number containing the SVC threshold.

    - ``sample`` -- iterable containing the examples `(x_i, y_i)`.

    - ``kernel`` -- Kernel instance (default: LinearKernel()) SVC kernel
      function.

    OUTPUT:

    Classifier -- a SVMClassifier instance.

    EXAMPLES:

    ``SVMClassifier`` instances can be defined directly through the class
    constructor; for instance it is possible to get such a classifier for the
    binary AND sample:

    ::

        >>> from yaplf.data import LabeledExample
        >>> from yaplf.models.svm import SVMClassifier
        >>> and_sample = (LabeledExample((0., 0.), -1.),
        ... LabeledExample((1., 0.),1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), 1.))
        >>> svc = SVMClassifier((1, 1, 1, 0), -0.5, and_sample)

   To verify how ``svc`` correctly classifies ``and_sample`` it is possible to
   invoke the model's ``classify`` function on every pattern, subsequently
   comparing the result with the original labels:

    ::

        >>> map(svc.compute, [ e.pattern for e in and_sample ])
        [-1.0, 1.0, 1.0, 1.0]
        >>> [e.label for e in and_sample]
        [-1.0..., 1.0..., 1.0..., 1.0...]
        >>> map(svc.compute, [e.pattern for e in and_sample]) == \
        ... [e.label for e in and_sample]
        True

    The same result can be obtained more quickly, directly invoking the
    ``test`` model inherited by the ``Model`` base class:

    ::

        >>> svc.test(and_sample)
        0.0

    Specification of a kernel function allows more flexible SV-classifiers,
    able to correctly classify a more complex sample such as the binary XOR
    one:

    ::

        >>> from yaplf.models.kernel import GaussianKernel
        >>> xor_sample = (LabeledExample((0., 0.), -1.),
        ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
        ... LabeledExample((1., 1.), -1.))
        >>> svc = SVMClassifier([1.52, 2.02, 2.02, 1.52], -0.39, xor_sample,
        ... kernel = GaussianKernel(0.6))
        >>> map(svc.compute, [e.pattern for e in xor_sample])
        [-1.0, 1.0, 1.0, -1.0]
        >>> [e.label for e in xor_sample]
        [-1.0..., 1.0..., 1.0..., -1.0...]
        >>> svc.test(xor_sample)
        0.0

    Another way to figure out how a SV classifier behaves is through a plot of
    its decision function:

    ::

        >>> svc.plot((-0.5, 1.5), (-0.5, 1.5), margin = True, separator = True,
        ... shading = True, margin_color = 'red', margin_width = 7)

    For a more detailed view of how decision function plot can be fine tuned,
    see the documentation for ``plot`` function later on in this class.

    IMPLEMENTATION DEPENDENT ISSUES:

    When used withing sage, ``plot`` outputs a graphic object which is
    directly shown within a netbook or through a helper application if the
    invocation is made in a command-line interface.

    When used in a pure python environment ``plot`` outputs a matplotlib
    figure, which is opened or saved through the standard library functions;
    for instance, the following example draws the same plot of before and saves
    it in a file named ``svc-decision-function.png``:

    ::

        >>> from yaplf.graph import MatplotlibPlotter
        >>> fig = svc.plot((-0.5, 1.5), (-0.5, 1.5), margin = True,
        ... separator = True, shading = True, margin_color = 'red',
        ... margin_width = 7, plotter = MatplotlibPlotter())
        >>> fig.savefig('svc-decision-function.png')

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, alpha, threshold, sample, **kwargs):
        r"""See ``SVMClassifier`` for full documentation.

        """

        Classifier.__init__(self)

        num_patterns = len(sample)
        check_svm_classification_sample(sample)
        self.dim = len(sample[0].pattern)

        if len(alpha) != num_patterns:
            raise ValueError('The supplied sample and multipliers vector do \
not have the same size')

        self.sv_indices = [i for i in range(len(alpha)) if alpha[i] != 0]
        
        self.support_vectors = [sample[i].pattern for i in self.sv_indices]
        self.signed_alphas = [alpha[i] * sample[i].label
            for i in self.sv_indices]
        self.threshold = threshold

        try:
            self.kernel = kwargs['kernel']
        except KeyError:
            self.kernel = Kernel.get_default()

    def __repr__(self):
        alpha = [abs(a) for a in self.signed_alphas]
        # was
        # map(abs, self.signed_alphas)
        patterns = self.support_vectors
        labels = [sign(a) for a in self.signed_alphas]
        # was
        # labels = map(sign, self.signed_alphas)
        sample = [LabeledExample(*pl) for pl in zip(patterns, labels)]
        # was
        # sample = map(lambda x: LabeledExample(*x), zip(patterns, labels))

        result = 'SVMClassifier(' + str(alpha) + ', '
        result += str(self.threshold) + ', '
        result += str(sample.__repr__())
        if self.kernel != Kernel.get_default():
            result += ', kernel = ' + str(self.kernel.__repr__())
        result += ')'
        return result

    def __str__(self):
        return self.__repr__()

    def decision_function(self, pattern):
        r"""
        Returns the decision function associated by the classifier to the
        specified pattern. Its sign determines the class associated to the
        pattern.

        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``pattern`` -- pattern on which the decision function is to be
          computed.

        OUTPUT:

        Number -- the SV classifier decision function value.

        EXAMPLES:

        This function is called in order to get the decision function value
        corresponding to a given pattern:

        ::

            >>> from yaplf.data import LabeledExample
            >>> from yaplf.models.svm import SVMClassifier
            >>> from yaplf.models.kernel import GaussianKernel
            >>> xor_sample = (LabeledExample((0., 0.), -1.),
            ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), -1.))
            >>> svc = SVMClassifier([1.52, 2.02, 2.02, 1.52], -0.39,
            ... xor_sample, kernel = GaussianKernel(0.6))
            >>> svc.decision_function((0, 0))
            -0.99712539305333991
            >>> svc.decision_function((1, 0))
            0.99756586384169432
            >>> svc.decision_function((0.5, 0))
            0.05142629338510496

        It is easy to see that che class in output of the SV classifier equals
        the sign of the corresponding decision function:

        ::

            >>> svc.compute((0, 0))
            -1.0
            >>> svc.compute((1, 0))
            1.0
            >>> svc.compute((0.5, 0))
            1.0

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(pattern) != self.dim:
            raise ValueError('The supplied pattern is incompatible with the \
SVM dimension')

        kernel_values = [self.kernel.compute(x, pattern)
            for x in self.support_vectors]
        # was
        # kernel_values = map(lambda x: self.kernel.compute(x, pattern),
        #     self.support_vectors)

        return dot(self.signed_alphas, kernel_values) + self.threshold

    def compute(self, pattern):
        r"""
        Associates a class (through values -1 and 1) to the specified
        pattern, computing the sign of the corresponding decision function
        value.

        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``pattern`` -- pattern whose corresponding class is to be computed.

        OUTPUT:

        -1 or 1 -- the class associated to the supplied pattern by the SV
        classifier.

        EXAMPLES:

        Consider the following ``SVMClassifier`` instance expressly tailored in
        order to deal with the binary AND sample: cycling through the sample
        items and feeding their patterns to the ``compute`` method one obtains
        the original labels:

        ::

            >>> from yaplf.data import LabeledExample
            >>> from yaplf.models.svm import SVMClassifier
            >>> and_sample = (LabeledExample((0., 0.), -1.),
            ... LabeledExample((1., 0.),1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), 1.))
            >>> svc = SVMClassifier((1, 1, 1, 0), -0.5, and_sample)
            >>> map(svc.compute, [ e.pattern for e in and_sample ])
            [-1.0, 1.0, 1.0, 1.0]

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return sign(self.decision_function(pattern))
