
r"""
Package handling the plotting of support vector-based models in yaplf

Package yaplf.models.svm.plot contains all the classes handling the plot
generation for SV-based models in yaplf.

AUTHORS:

- Dario Malchiodi (2011-01-04): initial version, factored out from
  yaplf.models.svm

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************


from matplotlib.cm import Greys

from yaplf.models import Classifier
from yaplf.graph import Plot


class SVMClassifierDecisionFunctionPlot(Plot):
    r"""
    """

    def __init__(self, classifier):
       r"""
       """

       self.classifier = classifier

    def plot(self, *args, **kwargs):
        r"""
        Returns the plot SV classifier decision function. Depending on the
        environment within which the function is called, the plot is returned
        as a matplotlib figure or as a sage graphics. Raises a ValueError if
        invoked on classifiers not having exactly two or three input units.

        The SV classifier decision function plot can contain: i) the plot of
        the curve/surface separating positive and negative patterns, ii) the
        plot of the curve/surface corresponding to SV margin, and iii) a color
        gradient describing the decision function values. The appearance of
        all these ingredients is customizable through the ``plot`` function
        named arguments.

        INPUT:

        - ``self`` -- SVMClassifier object on which the function is invoked.

        - ``args`` -- list of two or three visualization ranges for the
          involved variables, where each range is in turn a two-elements list
          or tuple containing respectively the lower and upper extreme.

        - ``separator`` -- boolean (default value: True) flag triggering the
          visualization of the curve/surface separating positive and negative
          patterns.

        - ``separator_color`` -- color (default value: 'black') color to be
          used in order to draw the curve/surface separating positive and
          negative patterns.

        - ``separator_style`` -- line style (default value: '-') style to be
          used in order to draw the curve/surface separating positive and
          negative patterns.

        - ``separator_width`` -- number (default value: 1) line width to be
          used in order to draw the curve/surface separating positive and
          negative patterns.

        - ``margin`` -- boolean (default value: False) flag triggering the
          visualization of the curve/surface showing the SV margin.

        - ``margin_color`` -- color (default value: 'gray') color to be used
          in order to draw the curve/surface showing the SV margin.

        - ``margin_style`` -- line style (default value: '-') style to be used
          in order to draw the curve/surface showing the SV margin.

        - ``margin_width`` -- number (default value: 1) line width to be used
          in order to draw the curve/surface showing the SV margin.

        - ``shading`` -- boolean (default value: False) flag triggering the
          decision function visualization in form of a color gradient.

        - ``shading_color`` -- colormap (default value: Greys) colormap to be
          used in order to show decision function values.

        - ``plotter`` -- Plotter object (default: SagePlotter() when the code
          is run within sage and MatplotlibPlotter() otherwise) to be used in
          order to render graphics.

        OUTPUT:

        graphic object -- the decision function plot, in form of a matplotlib
        figure or of a sage graphic.

        EXAMPLES:

        Consider the following ``SVMClassifier`` instance expressly tailored in
        order to deal with the binary AND sample:

        ::

            >>> from yaplf.data import LabeledExample
            >>> from yaplf.models.svm import SVMClassifier
            >>> and_sample = [LabeledExample((1., 1.), 1.),
            ... LabeledExample((0., 0.), -1.), LabeledExample((0, 1), -1.),
            ... LabeledExample((1, 0), -1.)]
            >>> svc = SVMClassifier([3, 1, 1, 1], -3, and_sample)

        The first arguments to ``plot`` are the visualization ranges. In this
        case two such ranges are needed, for the SV classifier has two inputs.
        In the following example, the visualization range is `(0, 1.7)^2` and
        the plot will contain a red separator and show the SV margin (colored
        in gray as this is the default choice):

        ::
            >>> svc.plot((0., 1.7), (0., 1.7), separator_color = 'red',
            ... margin = True)

        The above function call automatically detects the environment it is
        invoked into, returning either a sage graphics or a matplotlib figure.
        It is anyway possible to force a specific return value through
        specification of the ``plotter`` named argument. For instance, the
        following instruction explicitly require to use matplotlib, and this
        allows for specific line styles not (yet?) supported in the default
        sage graphics library:

        ::

            >>> from yaplf.graph import MatplotlibPlotter
            >>> fig = svc.plot((0., 1.7), (0., 1.7),
            ... plotter = MatplotlibPlotter(), separator_color = 'red',
            ...  separator_style = '--', margin = True, margin_width = 3,
            ... color_bar = True)
            >>> fig.savefig('and-SVC.png')

        It is important to point out that the above function call can be used
        within sage, too. Moreover, when using the notebook facility instead
        of the command line interface, invocation of the ``savefig`` function
        has the effect of showing up the plot in the notebook. The matplotlib
        plotter is used within sage when particular features not (yet?)
        implemented are needed in a plot, such as particular line styles.

        Using a specific kernel function instead of the default one
        corresponding to a standard dot product in the original patterns space
        requires to refer to the named argument ``kernel``, whose valid values
        are specific subinstances of the ``Kernel`` class defined in package
        yaplf.utility. For instance, the following instructions build an
        instance of ``SVClassifier`` expressly tailored in order to correctly
        classify a binary XOR sample through exploitation of a Gaussian kernel,
        subsequently plotting its decision function in `(0, 2)^2` and showing
        both separator and margin:

        ::

            >>> from yaplf.models.kernel import GaussianKernel
            >>> xor_sample = [LabeledExample((1., 1.), -1.),
            ... LabeledExample((0., 0.), -1.), LabeledExample((0, 1), 1.),
            ... LabeledExample((1, 0), 1.)]
            >>> svc = SVMClassifier([6.21, 6.21, 6.71, 6.71], -0.65,
            ... xor_sample, kernel = GaussianKernel(1))
            >>> svc.plot((0., 2.), (0., 2.), margin = True)

        The same result can be obtained through specific plotters, such as
        those based on matplotlib:

        ::
            >>> fig = svc.plot((0., 2.), (0., 2.), margin = True,
            ... plotter = MatplotlibPlotter())
            >>> fig.savefig('mpl-gauss-svm.png')

        Plots can be combined easily in sage through the graph concatenation
        operator `+`. The following example shows how to plot a data set
        together with the corresponding SV classifier decision function:

        ::

            >>> from yaplf.data import classification_data_plot
            >>> cf = lambda x: ('white' if x.label == 1 else 'red')
            >>> fig_xor_sample = classification_data_plot(xor_sample,
            ... color_function = cf)
            >>> svc = SVMClassifier([1.52, 2.02, 2.02, 1.52], -0.39,
            ... xor_sample, kernel = GaussianKernel(0.6))
            >>> fig_xor_model = svc.plot((-1, 2), (-1, 2), margin = True,
            ... separator = True, shading = True, margin_color = 'red')
            >>> fig_xor_model + fig_xor_sample

        A similar result in matplotlib requires the use of ``base`` named
        argument:

        ::

            >>> fig_mpl = classification_data_plot(xor_sample,
            ... color_function = cf, plotter = MatplotlibPlotter())
            >>> fig_mpl = svc.plot((-1, 2), (-1, 2), margin = True,
            ... separator = True, shading = True, margin_color = 'red',
            ... margin_style = ":", plotter = MatplotlibPlotter(),
            ... base = fig_mpl)
            >>> fig_mpl.savefig('mpl-svm-xor.png')

        The figure produced by ``plot`` can be fine tuned. For instance, the
        following instructions produce a decision function plot of another SV
        classifier for the binary XOR function, where the curve highlighting
        the SV margin is rendered through a red dashed style with fixed line
        width, while the gradient is colored in blue shades:

        ::

            >>> from matplotlib.cm import Blues
            >>> svc = SVMClassifier([0.5, 1, 1, 0.5], -0.76, xor_sample,
            ... kernel = GaussianKernel(0.5))
            >>> fig_xor_model_color = svc.plot((-.9, 1.9), (-.9, 1.9),
            ... margin = True, separator = True, shading = True,
            ... margin_color = 'red', margin_width = 1, margin_style = '--',
            ... shading_color = Blues)
            >>> fig_xor_model_color + fig_xor_sample

        A similar result can be obtained through matplotlib exploiting the
        same technique previously explained:

        ::
            >>> fig_mpl_2 = classification_data_plot(xor_sample,
            ... color_function = cf, plotter = MatplotlibPlotter())
            >>> fig_mpl_2 = svc.plot((-.9, 1.9), (-.9, 1.9), margin = True,
            ... separator = True, shading = True, margin_color = 'red',
            ... margin_width = 1, margin_style = '--', shading_color = Blues,
            ... plotter = MatplotlibPlotter(), base = fig_mpl_2,
            ... color_bar = True)
            >>> fig_mpl_2.savefig('mpl-svm-xor-color.png')

        Decision function plots are available also for SV classifiers having
        three inputs, although with a limited number of features:

        ::
            >>> td_sample = [LabeledExample((1., 1., 1.), 1),
            ... LabeledExample((0., 0., 1.), -1),
            ... LabeledExample((0., 1, 0.), -1),
            ... LabeledExample((1., 0., 0.), -1),
            ... LabeledExample((0., 0., 0.), 1)]
            >>> p0 = classification_data_plot(td_sample,
            ... color_function = lambda x: ('green' if x.label == 1 else
            ... 'yellow'))
            ... svc  = SVMClassifier([11.04, 10.42, 10.42, 10.42, 21.24],
            ... -1.14, td_sample, kernel = GaussianKernel(1.4))
            >>> p1 = svc.plot((-.5, 3.), (-1., 3.), (-1., 3.))
            >>> p0 + p1

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            separator = kwargs['separator']
            del kwargs['separator']
        except KeyError:
            separator = True
        try:
            separator_color = kwargs['separator_color']
            del kwargs['separator_color']
        except KeyError:
            separator_color = 'black'
        try:
            separator_style = kwargs['separator_style']
            del kwargs['separator_style']
        except KeyError:
            separator_style = '-'
        try:
            separator_width = kwargs['separator_width']
            del kwargs['separator_width']
        except KeyError:
            separator_width = 1

        try:
            margin = kwargs['margin']
            del kwargs['margin']
        except KeyError:
            margin = False
        try:
            margin_color = kwargs['margin_color']
            del kwargs['margin_color']
        except KeyError:
            margin_color = 'gray'
        try:
            margin_style = kwargs['margin_style']
            del kwargs['margin_style']
        except KeyError:
            margin_style = '-'
        try:
            margin_width = kwargs['margin_width']
            del kwargs['margin_width']
        except KeyError:
            margin_width = 1

        try:
            shading = kwargs['shading']
            del kwargs['shading']
        except KeyError:
            shading = False
        try:
            shading_color = kwargs['shading_color']
            del kwargs['shading_color']
        except KeyError:
            shading_color = Greys

        levels = []
        levels_color = []
        levels_style = []
        levels_width = []

        if margin:
            levels.append(-1.)
            levels_color.append(margin_color)
            levels_style.append(margin_style)
            levels_width.append(margin_width)

        if separator:
            levels.append(0.)
            levels_color.append(separator_color)
            levels_style.append(separator_style)
            levels_width.append(separator_width)

        if margin:
            levels.append(1.)
            levels_color.append(margin_color)
            levels_style.append(margin_style)
            levels_width.append(margin_width)

        kwargs['contours'] = levels
        kwargs['contour_color'] = levels_color
        kwargs['contour_style'] = levels_style
        kwargs['contour_width'] = levels_width
        kwargs['gradient'] = shading
        kwargs['gradient_color'] = shading_color

        return Classifier.plot(self.classifier, *args, **kwargs)    