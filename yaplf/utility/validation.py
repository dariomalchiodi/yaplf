
r"""
Module handling utility classes and functions in yaplf

Module :mod:`yaplf.utility.validation` contains validation utilities in yaplf.

AUTHORS:

- Dario Malchiodi (2010-12-30): initial version, factored out from
  :mod:`yaplf.utility`, containing the standard cross validation procedure

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from yaplf.utility import mean, filter_arguments, flatten, split, \
    cartesian_product

from yaplf.utility.error import MSE

from numpy import array


def train_and_test(learning_algorithm, train, test, parameters, **kwargs):
    r"""
    Train a model using :obj:`learning_algorithm` on the sample described in
    :obj:`train` with parameters as in :obj:`parameters`, and subsequently test
    the obtained model on the examples in :obj:`test` according to the
    criterion expressed in :obj:`error_model`.

    :param learning_algorithm: learning algorithm to be used.

    :type learning_algorithm: :class:`LearningAlgorithm`

    :param train: sample to be used for training.

    :type train: list or tuple of :class:`Example`

    :param test: sample to be used for testing.

    :type test: list or tuple of :class:`Example`

    :param parameters: parameters to be passed to the learning algorithm.

    :type parameters: dictionary with keys set to parameters name

    :param run_parameters: parameters for the :meth:`run` method of the
      learning algorithm.

    :type run_parameters: dictionary with keys set to parameters name,
      default: {}

    :param error_model: error model to be used to measure the test performance
      of the induced model.

    :type error_model: :class:`yaplf.utility.error.ErrorModel`,
      default: :class:`yaplf.utility.error.MSE`

    :returns: error of the inferred model on the test sample.

    :rtype: float

    EXAMPLES:

    Consider a learning problem based on the binary AND function. If three out
    of the four examples are used as training set as follows, leaving the
    remaining one as test set, a perceptron can be successfully trained using
    the Rosenblatt algorithm [Rosenblatt, 1958]. Indeed, invoking the
    :meth:`train_and_test` method with suitable learning algorithm parameters
    one obtains a null test error:

    >>> from yaplf.utility.stopping import FixedIterationsStoppingCriterion
    >>> from yaplf.utility.validation import train_and_test
    >>> from yaplf.data import LabeledExample
    >>> from yaplf.algorithms.neural import RosenblattPerceptronAlgorithm
    >>> train_sample = (LabeledExample((0, 0,), (0,)), \
    ... LabeledExample((0, 1), (1,)), LabeledExample((1, 0), (1,)))
    >>> test_sample = (LabeledExample((1, 1), (1,)),)
    >>> sc = FixedIterationsStoppingCriterion(5000)
    >>> train_and_test(RosenblattPerceptronAlgorithm, train_sample,
    ... test_sample, {'threshold': True, 'weight_bound': 0.1, 'beta': 0.8},
    ... run_parameters = {'stopping_criterion': sc})
    0.0

    REFERENCES:

    [Rosenblatt, 1958] Frank Rosenblatt, The Perceptron: A Probabilistic Model
    for Information Storage and Organization in the Brain, Psychological
    Review, v65, No. 6, pp. 386-408, doi:10.1037/h0042519.

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    try:
        run_parameters = kwargs['run_parameters']
    except KeyError:
        run_parameters = {}

    try:
        error_model = kwargs['error_model']
    except KeyError:
        error_model = MSE()

    algorithm = learning_algorithm(train, **parameters)
    algorithm.run(**run_parameters)
    return algorithm.model.test(test, error_model)


def cross_validation_step(learning_algorithm, parameters, split_sample,
    **kwargs):
    r"""
    Perform one step of cross validation using :obj:`learning_algorithm` as
    algorithm and :obj:`parameters` as parameters. For each sample chunk in
    :obj:`split_sample`, the algorithm is run using the remaining chunks in
    order to assemble training set and validating the result on the excluded
    chunk. The operation is cycled on all chunks and the obtained errors are
    averaged in order to assess the overall performance.

    :param learning_algorithm: learning algorithm to be used for training.

    :type learning_algorithm: :class:`yaplf.algorithms.LearningAlgorithm`

    :param parameters: parameters and values to be fed to the learning
      algorithm

    :type parameters: dictionary with parameters name as keys (note that
      typically these parameters have different value at each invocation
      of :meth:`cross_validation_step`).

    :param split_sample: partition of the available sample in chuncks
      (approximately) having the same size.

    :type split_sample: list or tuple composed by lists or tuples of
      :class:`yaplf.data.Example`

    :param fixed_parameters: -- assignments to parameters of the learning
      algorithm whose value does not change in the various cross validation
      steps.

    :type fixed_parameters: dictionary with parameters name as keys, default:
      {}

    :param error_measure: function to be used in order to average test errors
      on the various sample chunks.

    :type error_measure: function taking a list/tuple as argument and returning
      a float, default: numpy.mean

    :param run_parameters: assignments to parameters to be passed to the
      :meth:`run` method of the learning algorithm (forwarded to
      :meth:`train_and_test`).

    :type run_parameters: dictionary with parameters name as keys, default:
      {}

    :param error_model: error model to be used in order to evaluate the test
      error of a single chunk (forwarded to :meth:`train_and_test`).

    :type error_model: :class:`yaplf.utility.error.ErrorModel`, default:
      :class:`yaplf.utility.error.MSE`

    :returns: averaged performance of the induced models.

    :rtype: float

    EXAMPLES:

    Starting from two data sets, the following instructions train a perceptron
    using the Rosenblatt algorithm [Rosenblatt, 1958] on one of them and
    subsequently test the inferred perceptron on the remaining set. The
    procedure is then repeated after exchanging train and test set, and the two
    test errors are averaged:

    >>> from yaplf.data import LabeledExample
    >>> from yaplf.algorithms.neural import RosenblattPerceptronAlgorithm
    >>> from yaplf.utility.validation import cross_validation_step
    >>> split_sample = ((LabeledExample((0, 0), (0,)),
    ... LabeledExample((0, 1), (1,))), (LabeledExample((1, 0), (1,)),
    ... LabeledExample((1, 1), (1,))))
    >>> parameters = {'threshold': True}
    >>> cross_validation_step(RosenblattPerceptronAlgorithm, parameters,
    ... split_sample, fixed_parameters = {'num_steps': 500})
    0.75

    REFERENCES

    [Rosenblatt, 1958] Frank Rosenblatt, The Perceptron: A Probabilistic Model
    for Information Storage and Organization in the Brain, Psychological
    Review, v65, No. 6, pp. 386-408, doi:10.1037/h0042519.

    AUTHORS

    - Dario Malchiodi (2010-02-22)

    """

    try:
        fixed_parameters = kwargs['fixed_parameters']
    except KeyError:
        fixed_parameters = {}

    try:
        error_measure = kwargs['error_measure']
    except KeyError:
        error_measure = mean

    filtered_args = filter_arguments(kwargs, \
        ('fixed_parameters', 'error_measure'))

    parameters.update(fixed_parameters)
    errors = [train_and_test(learning_algorithm,
        flatten(split_sample[:i] + split_sample[i + 1:]),
        split_sample[i], parameters, **filtered_args)
        for i in range(len(split_sample))]
    return error_measure(errors)


def cross_validation(learning_algorithm, sample, parameters_description, \
    **kwargs):
    r"""
    Perform cross validation on a given sample using a fixed learning
    algorithm and a set of possible named parameter values.

    :param learning_algorithm: -- learning algorithm to be used.

    :type learning_algorithm: :class:`yaplf.algorithms.LearningAlgorithm`

    :param sample: sample to be cross validated.

    :type sample: list or tuple of :class:`yaplf.data.Example`

    :param parameters_description: candidate values for parameters of the
      learning algorithm

    :type parameters_description: dictionary whose entries have as key a
      string describing a paramter's name and as value a list or tuple
      enclosing candidate values.

    :param num_folds: number of folds of the provided sample.

    :type num_folds: integer, default: 5)

    :param verbose: flag triggering verbose output.

    :type verbose: boolean, default: ``False``

    :param fixed_parameters: parameters of the learning algorithm whose value
      does not change in the various cross validation steps.

    :type fixed_parameters: dictionary with parameters name as keys, default:
      {}

    :param error_measure: function to be used in order to average test errors
      on the various sample chunks.

    :type error_measure: function taking a list/tuple as argument and returning
       a float, default: numpy.mean

    :param run_parameters: parameters to be passed to the :meth:`run` method of
      the learning algorithm (forwarded to :meth:`train_and_test`).

    :type run_parameters: dictionary with parameters name as keys, default:
      {} 

    :param error_model: error model to be used in order to evaluate the test
      error of a single chunk (forwarded to :meth:`train_and_test`).

    :type error_model: :class:`yaplf.utility.error.ErrorModel`, default:
      :class:`yaplf.utility.error.MSE`

    :returns: model trained on all data using the parameters optimizing the
      cross validation performance

    :rtype: :class:`yaplf.models.Model`

    EXAMPLES:

    The following instructions perform a cross validation on a given sample in
    the aim of selecting the best combination for two parameters' values, where
    each can be chosen in two ways. As :class:`yaplf.algorithms.IdiotAlgorithm`
    has no plasticity, there is no true learning process. Anyway, setting the
    verbosity flag in :obj:`fixed_parameters` and through the corresponding
    named argument in :meth:`cross_validation` allows to see how the whole
    selection process go through examining the various parameters' choices, and
    for each of those how the sample chunks are generated and processed:

    >>> from yaplf.data import LabeledExample
    >>> from yaplf.algorithms import IdiotAlgorithm
    >>> from yaplf.utility.validation import cross_validation
    >>> sample = (LabeledExample((0, 0), (0,)),
    ... LabeledExample((1, 1), (1,)), LabeledExample((2, 2), (1,)),
    ... LabeledExample((3, 3), (1,)), LabeledExample((4, 4), (0,)),
    ... LabeledExample((5, 5), (1,)), LabeledExample((6, 6), (1,)),
    ... LabeledExample((7, 7), (1,)), LabeledExample((8, 8), (0,)),
    ... LabeledExample((9, 9), (1,)))
    >>> cross_validation(IdiotAlgorithm, sample,
    ... {'c': (1, 10), 'sigma': (.1, .01)},
    ... fixed_parameters = {'verbose': False},
    ... run_parameters = {'num_iters': 100}, num_folds = 3,
    ... verbose = False)
    ConstantModel(0)

    It is important to point out that cross validation works only on named
    parameters, thus any implementation of
    :class:`yaplf.algorithms.LearningAlgorithm` subclasses should be designed
    with this requirement in mind.


    AUTHORS

    - Dario Malchiodi (2010-02-22)

    """

    try:
        num_folds = kwargs['num_folds']
        del kwargs['num_folds']
    except KeyError:
        num_folds = 5

    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    split_sample = split(sample, \
        num_folds * (1.0 / num_folds,), random=False)

    parameters_candidate = cartesian_product(*parameters_description.values())

    errors = [cross_validation_step(learning_algorithm, \
        dict(zip(parameters_description.keys(), params)), split_sample, \
        **kwargs) for params in parameters_candidate]

    if verbose:
        print 'Errors: ' + str(errors)

    min_index = array(errors).argmin()
    best_parameters = parameters_candidate[min_index]

    if verbose:
        print 'Minimum error in position ' + str(min_index)
        print 'Selected parameters ' + str(best_parameters)

    try:
        fixed_parameters = kwargs['fixed_parameters']
    except KeyError:
        fixed_parameters = {}

    try:
        run_parameters = kwargs['run_parameters']
    except KeyError:
        run_parameters = {}

    final_parameters = dict(zip(parameters_description.keys(), \
        best_parameters))
    final_parameters.update(fixed_parameters)

    return __final_learning(sample, learning_algorithm, final_parameters, \
        run_parameters)


def __final_learning(sample, learning_algorithm, final_parameters, \
    run_parameters):
    r"""
    Private function to be called in order to get the model in output of the
    cross-validation procedure.
    """

    algorithm = learning_algorithm(sample, **final_parameters)
    algorithm.run(**run_parameters)
    return algorithm.model 