
r"""
Package handling variable-quality SV solvers in yaplf.

Package yaplf.algorithms.svm.vq.solvers contains all the classes handling
solvers in variable-quality SV learning algorithms. A solver is specialized in
finding the solution of one of the peculiar constrained optimization problems
rising when dealing with variable-quality SV algorithms.

TODO:


- pep8 check
- pylint check

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version.

- Dario Malchiodi (2010-04-06): added ``SVMClassificationSolver``,
  ``PyMLClassificationSolver``.

- Dario Malchiodi (2010-04-12): added ``CVXOPTVQClassificationSolver``.

- Dario Malchiodi (2013-09-04): moved into a separate package for
  variable-quality learning.

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
