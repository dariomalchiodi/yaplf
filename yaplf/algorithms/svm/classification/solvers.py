
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

- Dario Malchiodi (2010-04-12): added ``CVXOPTVQClassificationSolver``,

- Dario Malchiodi (2014-01-20): added ``GurobiClassificationSolver``.

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

from yaplf.models.kernel import LinearKernel

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

try:
    import gurobipy
except ImportError:
    print 'Warning: no gurobipy package'

from yaplf.utility import chop, kronecker_delta


class SVMClassificationSolver(object):
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

    def solve(self, sample, c, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C` and the kernel in ``kernel``.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c`` -- float or None (the former choice selects the
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


class GurobiClassificationSolver(SVMClassificationSolver):
    r"""
    SVM Classification solver based on gurobi. This solver is specialized in
    finding the approximate solution of the optimization problem described in
    [Cortes and Vapnik, 1995], both in its original and soft-margin
    formulation.

    INPUT:

    - ``self`` -- object on which the function is invoked.

    - ``verbose`` -- boolean (default: ``False``) flag triggering verbose mode.

    OUTPUT:

    ``GurobiClassificationSolver`` object.

    EXAMPLES:

    Consider the following representation of the AND binary function, and a
    default instantiation for ``GurobiClassificationSolver``:

    ::

        >>> from yaplf.data import LabeledExample
        >>> and_sample = [LabeledExample((1, 1), 1),
        ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
        ... LabeledExample((1, 0), -1)]
        >>> from yaplf.algorithms.svm.classification.solvers import \
        ... GurobiClassificationSolver
        >>> s = GurobiClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve``function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c` and a kernel instance in order
    to get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0, 0.999999999992222, 0.999999999992222]

    The value for `c` can be set to ``float('inf')``, in order to build and
    solve the original optimization problem rather than the soft-margin
    formulation:

    ::

        >>> s.solve(and_sample, float('inf'), LinearKernel())
        [4.00000000000204, 0, 1.999999999976717, 1.99999999997672]

    Note however that this class should never be used directly. It is
    automatically used by ``SVMClassificationAlgorithm``.

    REFERENCES:

    [Cortes and Vapnik, 1995] Corinna Cortes and Vladimir Vapnik,
    Support-Vector Networks, Machine Learning 20 (1995), 273--297.

    AUTHORS:

    - Dario Malchiodi (2014-01-20)

    """

    def __init__(self, verbose=False):
        r"""
        See ``GurobiClassificationSolver`` for full documentation.

        """

        try:
            gurobipy.os
        except NameError:
            raise NotImplementedError("gurobipy package not available")

        SVMClassificationSolver.__init__(self)
        self.verbose = verbose

    def solve(self, sample, c=float('inf'), kernel=LinearKernel(),
              tolerance=1e-6):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c`` -- float value for the tradeoff constant `C`.
          ``float('inf')`` selects the soft-margin version of the algorithm)

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used.

        - ``tolerance`` -- tolerance to be used when clipping values to the
          extremes of an interval.

        OUTPUT:

        list of float values -- optimal values for the optimization problem.

        EXAMPLES:

        Consider the following representation of the AND binary function, and a
        default instantiation for ``GurobiClassificationSolver``:

        ::

            >>> from yaplf.data import LabeledExample
            >>> and_sample = [LabeledExample((1, 1), 1),
            ... LabeledExample((0, 0), -1), LabeledExample((0, 1), -1),
            ... LabeledExample((1, 0), -1)]
            >>> from yaplf.algorithms.svm.classification.solvers \
            ... import GurobiClassificationSolver
            >>> s = GurobiClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve`` function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `C` and a kernel instance in order to
        get the solution of the corresponding SV classification optimization
        problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> s.solve(and_sample, 2, LinearKernel())
            [2, 0, 0.999999999992222, 0.999999999992222]

        The value for `C` can be set to ``float('inf')`` (which is also its
        default value), in order to build and solve the original optimization
        problem rather than the soft-margin formulation:

        ::

            >>> s.solve(and_sample, float('inf'), LinearKernel())
            [4.00000000000204, 0, 1.999999999976717, 1.99999999997672]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2014-01-20)

        """

        m = len(sample)
        patterns = [e.pattern for e in sample]
        labels = [e.label for e in sample]

        model = gurobipy.Model('classify')

        for i in range(m):
            if c == float('inf'):
                model.addVar(name='alpha_%d' % i, lb=0,
                             vtype=gurobipy.GRB.CONTINUOUS)
            else:
                model.addVar(name='alpha_%d' % i, lb=0, ub=c,
                             vtype=gurobipy.GRB.CONTINUOUS)

        model.update()

        alphas = model.getVars()
        obj = gurobipy.QuadExpr() + sum(alphas)
        map(lambda (i, j):
            obj.add(alphas[i] * alphas[j] * labels[i] * labels[j] *
                    kernel.compute(patterns[i], patterns[j]), -0.5),
            [(i, j) for i in xrange(m) for j in xrange(m)])

        model.setObjective(obj, gurobipy.GRB.MAXIMIZE)

        constEqual = gurobipy.LinExpr()
        map(lambda x: constEqual.add(x, 1.0),
            [a*l for a, l in zip(alphas, labels)])
        model.addConstr(constEqual, gurobipy.GRB.EQUAL, 0)

        if not self.verbose:
            model.setParam('OutputFlag', False)

        model.optimize()

        alphas_opt = [chop(a.x, right=c, tolerance=tolerance) for a in alphas]

        return alphas_opt


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
        >>> from yaplf.algorithms.svm.classification.solvers import \
        ... CVXOPTClassificationSolver
        >>> s = CVXOPTClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve``function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c` and a kernel instance in order
    to get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0, 0.9999998669645057, 0.9999998669645057]

    The value for `c` can be set to ``float('inf')`` (the default
    value), in order to build and solve the original optimization problem
    rather than the soft-margin formulation:

    ::

        >>> s.solve(and_sample, float('inf'), LinearKernel())
        [4.000001003300218, 0, 2.000000364577095, 2.000000364577095]

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

    def solve(self, sample, c=float('inf'), kernel=LinearKernel()):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c`` -- float or ``float('inf')`` (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used
          (default value: ``LinearKernel()``, using a linear kernel)

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
            >>> from yaplf.algorithms.svm.classification.solvers \
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
            [2, 0, 0.9999998669645057, 0.9999998669645057]

        The value for `C` can be set to ``float('inf')``, in order to build
        and solve the original optimization problem rather than the
        soft-margin formulation:

        ::

            >>> s.solve(and_sample, float('inf'), LinearKernel())
            [4.000001003300218, 0, 2.000000364577095, 2.000000364577095]

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

        problem["obj_quad"] = cvxopt_matrix(
            [[elem_i.label * elem_j.label *
              kernel.compute(elem_i.pattern, elem_j.pattern)
              for elem_i in sample] for elem_j in sample])
        problem["obj_lin"] = cvxopt_matrix([-1.0] * num_examples)
        if c == float('inf'):
            problem["ineq_coeff"] = cvxopt_matrix(-1.0 * eye(num_examples))
            problem["ineq_const"] = cvxopt_matrix([0.0] * num_examples)
        else:
            problem["ineq_coeff"] = cvxopt_matrix([
                [-1.0 * kronecker_delta(i, j) for i in range(num_examples)]
                + [kronecker_delta(i, j) for i in range(num_examples)]
                for j in range(num_examples)])
            problem["ineq_const"] = cvxopt_matrix(
                [float(0.0)] * num_examples + [float(c)] * num_examples
            )

        # coercion to float in the following assignment is required
        # in order to work with sage notebooks
        problem["eq_coeff"] = cvxopt_matrix(
            [float(elem.label) for elem in sample], (1, num_examples)
        )
        problem["eq_const"] = cvxopt_matrix(0.0)
        # was
        # sol = solvers.qp(quad_coeff, lin_coeff, ineq_coeff, ineq_const, \
        #     eq_coeff, eq_const)

        sol = solvers.qp(problem["obj_quad"], problem["obj_lin"],
                         problem["ineq_coeff"], problem["ineq_const"],
                         problem["eq_coeff"], problem["eq_const"])

        if sol["status"] != 'optimal':
            raise ValueError('cvxopt returned status ' + sol["status"])

        # was
        # alpha = map(lambda x: chop(x, right = c), list(sol['x']))
        alpha = [chop(x, right=c) for x in list(sol['x'])]

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
        >>> from yaplf.algorithms.svm.classification.solvers \
        ... import PyMLClassificationSolver
        >>> s = PyMLClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve`` function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `C` and a kernel instance in order to get
    the solution of the corresponding SV classification optimization problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> alphas = s.solve(and_sample, 2, LinearKernel()) # doctest:+ELLIPSIS
        Cpos, Cneg...
        >>> print alphas
        [2.0, 0.0, 1.0, 1.0]

    The value for `C` can be set to ``None``, in order to build and solve the
    original optimization problem rather than the soft-margin formulation:

    ::

        >>> alphas = s.solve(and_sample, None, LinearKernel()) # doctest:+ELLIPSIS
        Cpos, Cneg...
        >>> print alphas
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

    def solve(self, sample, c, kernel):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c`` -- float or None (the former choice selects the
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
            >>> from yaplf.algorithms.svm.classification.solvers \
            ... import PyMLClassificationSolver
            >>> s = PyMLClassificationSolver()

        Once the solver instance is available, it is possible to invoke its
        ``solve``function, specifying a labeled sample such as ``and_sample``,
        a positive value for the constant `C` and a kernel instance in order to
        get the solution of the corresponding SV classification optimization
        problem:

        ::

            >>> from yaplf.models.kernel import LinearKernel
            >>> alphas = s.solve(and_sample, 2, LinearKernel()) # doctest:+ELLIPSIS
            Cpos, Cneg...
            >>> print alphas
            [2.0, 0.0, 1.0, 1.0]

        The value for `C` can be set to ``None``, in order to build and solve
        the original optimization problem rather than the soft-margin
        formulation:

        ::

            >>> alphas = s.solve(and_sample, None, LinearKernel()) # doctest:+ELLIPSIS
            Cpos, Cneg...
            >>> print alphas
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
        solver.C = (float(c) if c is not None else 100000000.)
        solver.train(data, saveSpace=False)
        alphas = [0.0] * len(sample)
        for index, value in transpose([solver.model.svID, solver.model.alpha]):
            alphas[int(index)] = abs(value)
        return alphas


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
        >>> from yaplf.algorithms.svm.classification.solvers import \
        ... NEOSClassificationSolver
        >>> s = NEOSClassificationSolver()

    Once the solver instance is available, it is possible to invoke its
    ``solve`` function, specifying a labeled sample such as ``and_sample``, a
    positive value for the constant `c` and a kernel instance in order
    to get the solution of the corresponding SV classification optimization
    problem:

    ::

        >>> from yaplf.models.kernel import LinearKernel
        >>> s.solve(and_sample, 2, LinearKernel())
        [2, 0, 1.0, 1.0]

    The value for `c` can be set to ``float('inf')``, in order to build and
    solve the original optimization problem rather than the soft-margin
    formulation:

    ::

        >>> alphas = s.solve(and_sample, float('inf'), LinearKernel()) # doctest:+ELLIPSIS
        ...
        >>> print alphas
        [4.0, 0, 2.0, 2.0]

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

    def solve(self, sample, c=float('inf'), kernel=LinearKernel()):
        r"""
        Solve the SVM classification optimization problem corresponding
        to the supplied sample, according to specified value for the tradeoff
        constant `C`.

        INPUT:

        - ``sample`` -- list or tuple of ``LabeledExample`` instances whose
          labels are all set either to `1` or `-1`.

        - ``c`` -- float or ``float('inf')`` (the former choice selects the
          soft-margin version of the algorithm) value for the tradeoff constant
          `C`.

        - ``kernel`` -- ``Kernel`` instance defining the kernel to be used
          (default: ``LinearKernel()``, accounting for a linear kernel).

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
            >>> from yaplf.algorithms.svm.classification.solvers \
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
            [2, 0, 1.0, 1.0]

        The value for `C` can be set to ``float('inf')``, in order to build
        and solve the original optimization problem rather than the
        soft-margin formulation:

        ::

            >>> s.solve(and_sample, float('inf'), LinearKernel())
            [4.0, 0, 2.0, 2.0]

        Note however that this class should never be used directly. It is
        automatically used by ``SVMClassificationAlgorithm``.

        AUTHORS:

        - Dario Malchiodi (2011-02-05)

        """

        neos = xmlrpclib.Server("http://%s:%d" % ("www.neos-server.org", 3332))

        num_examples = len(sample)
        input_dimension = len(sample[0].pattern)
        constraint = " <= " + str(c) if c != float('inf') else ""
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

        xml = """
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

        """ % (num_examples, input_dimension, kernel_description, constraint,
               "".join(pattern_description), "".join(label_description))

        (job_number, password) = neos.submitJob(xml)
        if self.verbose:
            print xml
            print "job number: %s" % job_number

        offset = 0
        status = ""
        while status != "Done":
            (msg, offset) = neos.getIntermediateResults(job_number, password,
                                                        offset)
            if self.verbose:
                print msg.data
            status = neos.getJobStatus(job_number, password)

        msg = neos.getFinalResults(job_number, password).data
        if self.verbose:
            print msg

        begin = 0
        while msg[begin] != '(':
            begin = begin + 1

        end = len(msg) - 1
        while msg[end] != ')':
            end = end - 1

        return [chop(alpha, right=c) for alpha in eval(msg[begin:end+1])]


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
            return "exp(-1*(sum{k in 1..n}(x[i,k]-x[j,k])^2)/(2*" + \
                str(self.kernel.sigma ** 2) + "))"
        elif self.kernel.__class__.__name__ == "HyperbolicKernel":
            return "tanh(" + str(self.kernel.scale) + \
                " * (sum{k in 1..n}x[i,k]*x[j,k]) + " + \
                str(self.kernel.offset) + ")"
        else:
            raise ValueError(str(self.kernel)
                             + 'not analytically representable')


# Needed in order to use cvxopt within sage
Integer = int
RealNumber = float
