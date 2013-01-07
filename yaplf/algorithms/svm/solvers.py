
r"""
Package handling SV solvers in yaplf.

Package yaplf.algorithms.svm.solvers contains all the classes handling solvers
in SV learning algorithms. A solver is specialized in finding the solution of
one of the peculiar constrained optimization problems rising when dealing with
SV algorithms.

TODO:

- SV regression solvers
- Accuracy SV regression solvers

- pep8 checked
- pylint score: 6.95

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version.

- Dario Malchiodi (2010-04-06): added ``SVMClassificationSolver``,
  ``PyMLClassificationSolver``.

- Dario Malchiodi (2010-04-12): added ``CVXOPTVQClassificationSolver``.

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

import xmlrpclib

from numpy import eye, array, transpose
try:
    from cvxopt import solvers
    from cvxopt.base import matrix as cvxopt_matrix
except ImportError:
    #print "Warning: no cvxopt package"
    pass

try:
    from PyML import VectorDataSet, SVM
except ImportError:
    #print "Warning: no PyML package"
    pass

from yaplf.utility import chop, kronecker_delta


class SVMClassificationSolver:
    r"""
    Base class for classification solvers. Subclasses should implement a
    ``solve`` method having in input a list/tuple of ``LabeledSample``
    instances, a positive float value and a ``Kernel`` subclass instance. This
    method should build the corresponding quadratic constrained optimization
    problem, solve it, and return the optimal solution.

    INPUT

    Each subclass can have different constructor inputs in order to take into
    account specific initialization values.

    OUTPUT

    SVMClassificationSolver instance

    EXAMPLES

    See the examples section for concrete subclasses, such as
    ``CVXOPTClassificationSolver`` in this package.

    AUTHORS

    - Dario Malchiodi (2010-04-06)

    """

    def __init__(self):
        r"""
        See ``ClassificationSolver`` for full documentation.

        """

        pass

    def solve(self, sample, c_value, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C` and the kernel in ``kernel``.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c_value`` -- float or None (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        See the example section for concrete subclasses such as
        ``CVXOPTClassificationSolver``.

        """

        raise NotImplementedError('solve not callable in base class')


class CVXOPTClassificationSolver(SVMClassificationSolver):
    r"""
    SVM Classification solver based on cvxopt. This solver is specialized in
    finding the approximate solution of the optimization problem described in
    [Cortes and Vapnik, 1995], both in its original and soft-margin
    formulation.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``verbose`` -- boolean (default: ``False``) flag triggering verbose mode.

    - ``max_iterations`` -- integer (default: `1000`) maximum number of solver
      iterations.

    - ``solver`` -- string (default: ``'mosek'``) cvxopt solver to be used.

    OUTPUT:

    ``CVXOPTClassificationSolver`` object.

    EXAMPLES:

    Consider the following representation of the AND binary function, and a
    default instantiation for ``CVXOPTClassificationSolver``:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), 1),
        ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
        ... LabeledExample((1, 0), -1)]
        >>> from yaplf.algorithms.svm.solvers import \
        ... CVXOPTClassificationSolver
        >>> s = CVXOPTClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve``function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c_value` and a kernel instance in order
    to get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0.0, 0.99999999360824832, 0.99999999360824821]

    The value for `c_value` can be set to ``None``, in order to build and solve
    the original optimization problem rather than the soft-margin formulation:

    ::

        >>> s.solve(and_sample, None, LinearKernel())
        [4.000000000999421, 0.0, 2.0000000001391336, 2.0000000001391336]

    Note however that this class should never be used directly. It is
    automatically used by ``SVMClassificationAlgorithm``.

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, **kwargs):
        r"""
        See ``CVXOPTClassificationSolver`` for full documentation.

        """

        try:
            solvers.options
        except NameError:
            raise NotImplementedError("cvxopt package not available")

        try:
            self.verbose = kwargs['verbose']
        except KeyError:
            self.verbose = False
        try:
            self.max_iterations = kwargs['max_iterations']
        except KeyError:
            self.max_iterations = 1000
        try:
            self.solver = kwargs['solver']
        except KeyError:
            self.solver = 'mosek'

        SVMClassificationSolver.__init__(self)

    def solve(self, sample, c_value, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c_value`` -- float or None (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        Consider the following representation of the AND binary function, and a
        default instantiation for ``CVXOPTClassificationSolver``:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), 1),
            ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
            ... LabeledExample((1, 0), -1)]
            >>> from yaplf.algorithms.svm.solvers \
            ... import CVXOPTClassificationSolver
            >>> s = CVXOPTClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve``function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `C` and a kernel instance in order to
        get the solution of the corresponding SV classification optimization
        problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> s.solve(and_sample, 2, LinearKernel())
            [2, 0.0, 0.99999999360824832, 0.99999999360824821]

        The value for `C` can be set to ``None``, in order to build and solve
        the original optimization problem rather than the soft-margin
        formulation:

        ::

            >>> s.solve(and_sample, None, LinearKernel())
            [4.000000000999421, 0.0, 2.0000000001391336, 2.0000000001391336]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        solvers.options['show_progress'] = self.verbose
        solvers.options['maxiters'] = self.max_iterations
        solvers.options['solver'] = self.solver

        # cvxopt solves the problem
        # min 1/2 x' Q x + p' x
        # subject to G x >= h and A x = b
        # dict below is mapped to the above symbols as follows:
        # problem["obj_quad"] -> Q
        # problem["obj_lin"] -> p
        # problem["ineq_coeff"] -> G
        # problem["ineq_const"] -> h
        # problem["eq_coeff"] -> A
        # problem["eq_const"] -> b

        num_examples = len(sample)
        problem = {}

        problem["obj_quad"] = cvxopt_matrix([[elem_i.label * elem_j.label *
            kernel.compute(elem_i.pattern, elem_j.pattern)
            for elem_i in sample] for elem_j in sample])
        problem["obj_lin"] = cvxopt_matrix([-1.0] * num_examples)
        if c_value is None:
            problem["ineq_coeff"] = cvxopt_matrix(-1.0 * eye(num_examples))
            problem["ineq_const"] = cvxopt_matrix([0.0] * num_examples)
        else:
            problem["ineq_coeff"] = cvxopt_matrix([
                [-1.0 * kronecker_delta(i, j) for i in range(num_examples)]
                + [kronecker_delta(i, j) for i in range(num_examples)]
                for j in range(num_examples)])
            problem["ineq_const"] = cvxopt_matrix([float(0.0)] * num_examples +
                [float(c_value)] * num_examples)

        # coercion to float in the following assignment is required
        # in order to work with sage notebooks
        problem["eq_coeff"] = cvxopt_matrix([float(elem.label)
            for elem in sample], (1, num_examples))
        problem["eq_const"] = cvxopt_matrix(0.0)
        # was
        # sol = solvers.qp(quad_coeff, lin_coeff, ineq_coeff, ineq_const, \
        #     eq_coeff, eq_const)

        sol = solvers.qp(problem["obj_quad"], problem["obj_lin"], \
            problem["ineq_coeff"], problem["ineq_const"], \
            problem["eq_coeff"], problem["eq_const"])

        if sol["status"] != 'optimal':
            raise ValueError('cvxopt returned status ' + sol.status)

        # was
        # alpha = map(lambda x: chop(x, right = c_value), list(sol['x']))
        alpha = [chop(x, right=c_value) for x in list(sol['x'])]

        return alpha


class PyMLClassificationSolver(SVMClassificationSolver):
    r"""
    SVM Classification solver based on PyML. This solver is specialized in
    finding the approximate solution of the optimization problem described in
    [Cortes and Vapnik, 1995], both in its original and soft-margin
    formulation.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``verbose`` -- boolean (default: ``False``) flag triggering verbose mode.

    OUTPUT:

    ``SVMClassificationSolver`` object.

    EXAMPLES:

    Consider the following representation of the AND binary function, and a
    default instantiation for ``PyMLClassificationSolver``:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), 1),
        ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
        ... LabeledExample((1, 0), -1)]
        >>> from yaplf.algorithms.svm.solvers \
        ... import PyMLClassificationSolver
        >>> s = PyMLClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve`` function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `C` and a kernel instance in order to get
    the solution of the corresponding SV classification optimization problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        ...
        [2.0, 0.0, 1.0, 1.0]

    The value for `C` can be set to ``None``, in order to build and solve the
    original optimization problem rather than the soft-margin formulation:

    ::

        >>> s.solve(and_sample, None, LinearKernel())
        ...
        [3.984375, 0.0, 1.9921875, 1.9921875]

    Note however that this class should never be used directly. It is
    automatically used by ``SVMClassificationAlgorithm``.

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2010-04-06)

    """

    def __init__(self):
        r"""
        See ``PyMLClassificationSolver`` for full documentation.

        """

        try:
            SVM()
        except NameError:
            raise NotImplementedError("PyML package not available")

        SVMClassificationSolver.__init__(self)

    def solve(self, sample, c_value, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c_value`` -- float or None (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        Consider the following representation of the AND binary function, and a
        default instantiation for ``PyMLClassificationSolver``:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), 1),
            ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
            ... LabeledExample((1, 0), -1)]
            >>> from yaplf.algorithms.svm.solvers \
            ... import PyMLClassificationSolver
            >>> s = PyMLClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve``function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `C` and a kernel instance in order to
        get the solution of the corresponding SV classification optimization
        problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> s.solve(and_sample, 2, LinearKernel())
            ...
            [2.0, 0.0, 1.0, 1.0]

        The value for `C` can be set to ``None``, in order to build and solve
        the original optimization problem rather than the soft-margin
        formulation:

        ::

            >>> s.solve(and_sample, None, LinearKernel())
            ...
            [3.984375, 0.0, 1.9921875, 1.9921875]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2010-04-06)

        """

        patterns = array([[float(p) for p in e.pattern] for e in sample])
        # was
        # patterns = array([map(float, e.pattern) for e in sample])
        labels = array([float(e.label) for e in sample])

        data = VectorDataSet(patterns, L=labels)
        if kernel.__class__.__name__ == 'LinearKernel':
            pass
        elif kernel.__class__.__name__ == 'GaussianKernel':
            data.attachKernel('gaussian',
                gamma=float(1.0 / (kernel.sigma ** 2)))
        elif kernel.__class__.__name__ == 'PolynomialKernel':
            data.attachKernel('poly', degree=int(kernel.degree),
                additiveConst=float(1))
        elif kernel.__class__.__name__ == 'HomogeneousPolynomialKernel':
            data.attachKernel('poly', degree=int(kernel.degree),
            additiveConst=float(0))
        else:
            raise NotImplementedError(str(kernel) + 'not implemented in PyML')

        solver = SVM(Cmode='equal')
        solver.C = (float(c_value) if c_value is not None else 100000000.)
        solver.train(data, saveSpace=False)
        alphas = [0.0] * len(sample)
        for index, value in transpose([solver.model.svID, solver.model.alpha]):
            alphas[int(index)] = abs(value)
        return alphas




class CVXOPTVQClassificationSolver(SVMClassificationSolver):
    r"""
    Variable-quality SVM Classification solver based on cvxopt. This solver is
    specialized in finding the approximate solution of the optimization problem
    described in [Apolloni et al., 2007], both in its original
    and soft-margin formulation.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``verbose`` -- boolean (default: ``False``) flag triggering verbose mode.

    - ``max_iterations`` -- integer (default: `1000`) maximum number of solver
      iterations.

    - ``solver`` -- string (default: ``'mosek'``) cvxopt solver to be used.

    - ``epsilon`` -- float (default: 10 ** -6) value used in order to simulate
       greater-than constraint using a greater-or-equal one.


    OUTPUT:

    ``CVXOPTVQClassificationSolver`` object.

    EXAMPLES:

    Consider the following representation of the AND binary function, and a
    default instantiation for ``CVXOPTVQClassificationSolver``:

    ::

        >>> from yaplf.data import LabeledExample, AccuracyExample
        >>> and_sample = [AccuracyExample(LabeledExample((1, 1), 1), 0),
        ... AccuracyExample(LabeledExample((0, 0), -1), 0),
        ... AccuracyExample(LabeledExample((0, 1), -1), 1),
        ... AccuracyExample(LabeledExample((1, 0), -1), 0)]
        >>> from yaplf.algorithms.svm.solvers \
        ... import CVXOPTVQClassificationSolver
        >>> s = CVXOPTVQClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve``function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c_value` and a kernel instance in order to
    get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0.0, 2, 0.0]

    The value for `c_value` can be set to ``None``, in order to build and solve
    the original optimization problem rather than the soft-margin formulation;
    analogously, a different kernel can be used as argument to the solver:

    ::

        >>> from yaplf.models.kernel import PolynomialKernel
        >>> s.solve(and_sample, None, PolynomialKernel(3))
        [0.15135135150351597, 0.0, 0.097297297016552056, 0.054054053943170456]

    Note however that this class should never be used directly. It is
    automatically used by ``SVMVQClassificationAlgorithm``.

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

    def __init__(self, **kwargs):
        r"""
        See ``CVXOPTVQClassificationSolver`` for full documentation.

        """

        try:
            solvers.options
        except NameError:
            raise NotImplementedError("cvxopt package not available")

        try:
            self.verbose = kwargs['verbose']
        except KeyError:
            self.verbose = False
        try:
            self.max_iterations = kwargs['max_iterations']
        except KeyError:
            self.max_iterations = 1000
        try:
            self.solver = kwargs['solver']
        except KeyError:
            self.solver = 'mosek'

        try:
            self.epsilon = kwargs['epsilon']
        except KeyError:
            self.epsilon = 10.0 ** -6

        SVMClassificationSolver.__init__(self)

    def solve(self, sample, c_value, kernel):
        r"""
        Solve the variable-quality SVM classification optimization problem
        corresponding to the supplied sample, according to specified value for
        tradeoff constant `C` and kernel `k`.

        INPUT:

        - ``sample`` -- list or tuple of ``AccuracyExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c_value`` -- float or None (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        Consider the following representation of the AND binary function, and a
        default instantiation for ``CVXOPTVQClassificationSolver``:

        ::

            >>> from yaplf.data import LabeledExample, AccuracyExample
            >>> and_sample = [AccuracyExample(LabeledExample((1, 1), 1), 0),
            ... AccuracyExample(LabeledExample((0, 0), -1), 0),
            ... AccuracyExample(LabeledExample((0, 1), -1), 1),
            ... AccuracyExample(LabeledExample((1, 0), -1), 0)]
            >>> from yaplf.algorithms.svm.solvers \
            ... import CVXOPTVQClassificationSolver
            >>> s = CVXOPTVQClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve``function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `c_value` and a kernel instance in
        order to get the solution of the corresponding SV classification
        optimization problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> s.solve(and_sample, 2, LinearKernel())
            [2, 0.0, 2, 0.0]

        The value for `c_value` can be set to ``None``, in order to build and
        solve the original optimization problem rather than the soft-margin
        formulation; analogously, a different kernel can be used as argument to
        the solver:

        ::

            >>> from yaplf.models.kernel import PolynomialKernel
            >>> s.solve(and_sample, None, PolynomialKernel(3))
            [0.15135135150351597, 0.0, 0.097297297016552056,
            0.054054053943170456]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMVQClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2010-04-12)

        """

        # cvxopt solves the problem
        # min 1/2 x' Q x + p' x
        # subject to G x >= h and A x = b
        # dict below is mapped to the above symbols as follows:
        # problem["obj_quad"] -> Q
        # problem["obj_lin"] -> p
        # problem["ineq_coeff"] -> G
        # problem["ineq_const"] -> h
        # problem["eq_coeff"] -> A
        # problem["eq_const"] -> b

        solvers.options['show_progress'] = self.verbose
        solvers.options['maxiters'] = self.max_iterations
        solvers.options['solver'] = self.solver

        # coercion to float in the following assignment is required
        # in order to work with sage notebook

        num_examples = len(sample)
        problem = {}

        problem["obj_quad"] = cvxopt_matrix([[ \
            float(elem_i.example.label * elem_j.example.label * \
            (kernel.compute(elem_i.example.pattern, elem_j.example.pattern) - \
            elem_i.example.label * elem_j.example.label * (elem_i.accuracy + \
            elem_j.accuracy))) for elem_i in sample] for elem_j in sample])
        problem["obj_lin"] = cvxopt_matrix([-1.0 for i in range(num_examples)])

        if c_value is None:
            problem["ineq_coeff"] = cvxopt_matrix([
                [float(-1.0 * kronecker_delta(i, j))
                for i in range(num_examples)] +
                [float(-1.0 * sample[j].accuracy)]
                for j in range(num_examples)])
            problem["ineq_const"] = cvxopt_matrix(
                [float(0.0)] * num_examples + [float(1.0 - self.epsilon)])
        else:
            problem["ineq_coeff"] = cvxopt_matrix([
                [float(-1.0 * kronecker_delta(i, j))
                for i in range(num_examples)] +
                [float(kronecker_delta(i, j))
                for i in range(num_examples)] +
                [float(-1.0 * sample[j].accuracy)]
                for j in range(num_examples)])
            problem["ineq_const"] = cvxopt_matrix([float(0.0)] * num_examples +
                [float(c_value)] * num_examples + [float(1.0 - self.epsilon)])

        problem["eq_coeff"] = cvxopt_matrix([float(elem.example.label)
            for elem in sample], (1, num_examples))
        problem["eq_const"] = cvxopt_matrix(0.0)
        sol = solvers.qp(problem["obj_quad"], problem["obj_lin"],
            problem["ineq_coeff"], problem["ineq_const"],
            problem["eq_coeff"], problem["eq_const"])

        return [chop(x, right=c_value) for x in list(sol['x'])]
        # was
        # return map(lambda x: chop(x, right = c_value), list(sol['x']))

class NEOSClassificationSolver(SVMClassificationSolver):
    r"""
    SVM Classification solver based on cvxopt. This solver is specialized in
    finding the approximate solution of the optimization problem described in
    [Cortes and Vapnik, 1995], both in its original and soft-margin
    formulation.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``verbose`` -- boolean (default: ``False``) flag triggering verbose mode.

    OUTPUT:

    ``NEOSClassificationSolver`` object.

    EXAMPLES:

    Consider the following representation of the AND binary function, and a
    default instantiation for ``NEOSClassificationSolver``:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), 1),
        ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
        ... LabeledExample((1, 0), -1)]
        >>> from yaplf.algorithms.svm.solvers import \
        ... NEOSClassificationSolver
        >>> s = NEOSClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve`` function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c_value` and a kernel instance in order
    to get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0.0, 0.99999999360824832, 0.99999999360824821]

    The value for `c_value` can be set to ``None``, in order to build and solve
    the original optimization problem rather than the soft-margin formulation:

    ::

        >>> s.solve(and_sample, None, LinearKernel())
        [4.000000000999421, 0.0, 2.0000000001391336, 2.0000000001391336]

    Note however that this class should never be used directly. It is
    automatically used by ``SVMClassificationAlgorithm``.

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2011-02-05)

    """

    def __init__(self, **kwargs):
        r"""
        See ``NEOSClassificationSolver`` for full documentation.

        """

        try:
            self.verbose = kwargs['verbose']
        except KeyError:
            self.verbose = False

        SVMClassificationSolver.__init__(self)

    def solve(self, sample, c_value, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c_value`` -- float or None (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        Consider the following representation of the AND binary function, and a
        default instantiation for ``NEOSClassificationSolver``:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), 1),
            ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
            ... LabeledExample((1, 0), -1)]
            >>> from yaplf.algorithms.svm.solvers \
            ... import NEOSClassificationSolver
            >>> s = NEOSClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve``function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `C` and a kernel instance in order to
        get the solution of the corresponding SV classification optimization
        problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> s.solve(and_sample, 2, LinearKernel())
            [2, 0.0, 1.0, 1.0]

        The value for `C` can be set to ``None``, in order to build and solve
        the original optimization problem rather than the soft-margin
        formulation:

        ::

            >>> s.solve(and_sample, None, LinearKernel())
            [4.000000000999421, 0.0, 2.0000000001391336, 2.0000000001391336]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2011-02-05)

        """

        neos=xmlrpclib.Server("http://%s:%d" % ("www.neos-server.org", 3332))

        num_examples = len(sample)
        input_dimension = len(sample[0].pattern)
        constraint = " <= " + str(c_value) if c_value is not None else ""
        kernel_description = AMPLKernelFactory(kernel).get_kernel_description()
        # that is, something like sum{k in 1..n}(x[i,k]*x[j,k])

        pattern_description = ["param   x:\t"]
        label_description = ["param y:=\n"]

        for component_index in range(input_dimension):
            pattern_description.append(str(component_index+1))
            pattern_description.append("\t")

        pattern_description.append(":=\n")

        example_number = 1
        for example in sample:
            pattern_description.append(str(example_number))
            for component in example.pattern:
                pattern_description.append("\t")
                pattern_description.append(str(component))

            label_description.append(str(example_number))
            label_description.append("\t")
            label_description.append(str(sample[example_number-1].label))

            example_number = example_number + 1
            if example_number > len(sample):
                pattern_description.append(";")
                label_description.append(";")
            pattern_description.append("\n")
            label_description.append("\n")

        xml="""
        <document>
        <category>nco</category>
        <solver>SNOPT</solver>
        <inputMethod>AMPL</inputMethod>
        <model><![CDATA[

        param m integer > 0 default %d; # number of sample points
        param n integer > 0 default %d; # sample space dimension

        param x {1..m,1..n}; # sample points
        param y {1..m}; # sample labels
        param dot{i in 1..m,j in 1..m}:=%s;

        var alpha{1..m} >=0%s;

        maximize quadratic_form:
        sum{i in 1..m} alpha[i]
        -1/2*sum{i in 1..m,j in 1..m}alpha[i]*alpha[j]*y[i]*y[j]*dot[i,j];

        subject to linear_constraint:
        sum{i in 1..m} alpha[i]*y[i]=0;

        ]]></model>

        <data><![CDATA[

        data;

        %s
        
        %s

        ]]></data>

        <commands><![CDATA[

        option solver snopt;

        solve;

        printf: "(";
        printf {i in 1..m-1}:"%%f,",alpha[i];
        printf: "%%f)",alpha[m];

        ]]></commands>

        </document>

        """ % (num_examples, input_dimension, kernel_description, constraint, "".join(pattern_description), "".join(label_description))

        (job_number, password) = neos.submitJob(xml)
        if self.verbose:
            print xml
            print "job number: %s" % job_number

        offset=0
        status = ""
        while status != "Done":
            (msg, offset) = neos.getIntermediateResults(job_number, password, offset)
            if self.verbose:
                print msg.data
            status = neos.getJobStatus(job_number, password)

        msg = neos.getFinalResults(job_number, password).data
        if self.verbose:
            print msg

        begin=0
        while msg[begin] != '(':
            begin = begin + 1

        end = len(msg) -1
        while msg[end] != ')':
            end = end - 1

        return [chop(alpha, right=c_value) for alpha in eval(msg[begin:end+1])]


class AMPLKernelFactory(object):
    r"""
    Factory class used in order to get a string containing the AMPL
    source code description for a given kernel. 
    """

    def __init__(self, kernel):
        self.kernel = kernel

    def get_kernel_description(self):
        if self.kernel.__class__.__name__ == "LinearKernel":
            return "sum{k in 1..n}(x[i,k]*x[j,k])"
        elif self.kernel.__class__.__name__ == "PolynomialKernel":
            return "(sum{k in 1..n}x[i,k]*x[j,k]+1)^" + str(self.kernel.degree)
        elif self.kernel.__class__.__name__ == "HomogeneousPolynomialKernel":
            return "(sum{k in 1..n}x[i,k]*x[j,k])^" + str(self.kernel.degree)
        elif self.kernel.__class__.__name__ == "GaussianKernel":
            return "exp(-1*(sum{k in 1..n}(x[i,k]-x[j,k])^2)/(2*" + str(self.kernel.sigma ** 2) + "))"
        elif self.kernel.__class__.__name__ == "HyperbolicKernel":
            return "tanh(" + str(self.kernel.scale) + " * (sum{k in 1..n}x[i,k]*x[j,k]) + " + str(self.kernel.offset) + ")"
        else:
            raise ValueError(str(self.kernel) + 'not analytically representable')


# Needed in order to use cvxopt within sage
Integer = int
RealNumber = float
