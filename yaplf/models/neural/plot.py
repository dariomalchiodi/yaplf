
r"""
Package handling neural models plots in yaplf

Package yaplf.models.neural.plot contains all the classes providing plots of
neural models in yaplf.

TODO:

AUTHORS:

- Dario Malchiodi (2011-01-04): initial version

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

from yaplf.graph import Plot, PlotterFactory 
from yaplf.models import Classifier
from matplotlib.cm import Greys

class PerceptronDecisionFunctionPlot(Plot):
    r"""
    Class wrapping a :class:`yaplf.models.neural.Perceptron` object and
    providing a :meth:`plot` method drawing a plot of its decision function.
    """

    def __init__(self, perceptron):
        r"""
        See :class:`PerceptronDecisionFunctionPlot` for full documentation.
        """

        self.perceptron = perceptron

    def plot(self, *args, **kwargs):
        r"""
        Returns a graphic containing the plot of the perceptron output. Raises
        a ValueError if invoked on perceptrons not having two or three input
        units. The graphic is either a bi- or three-dimensional plot according
        to the following invocation syntax:

        - ``plot(x_range, y_range)`` returns a 2D plot.

        - ``plot(x_range, y_range, z_range)`` returns a 3D plot.

        INPUT:

        - ``self`` -- ``Perceptron`` object on which the function is invoked.

        - ``x_range`` -- bidimensional iterable containing the range of x
          variable.

        - ``y_range`` -- bidimensional iterable containing the range of y
          variable.

        - ``z_range`` -- bidimensional iterable containing the range of z
          variable.

        - ``shading`` -- boolean (default value: False) flag triggering the
          perceptron output visualization in form of a color gradient.

        - ``shading_color`` -- colormap (default value: Greys) colormap to be
          used in order to show perceptron output.

        - ``x_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in x range.

        - ``y_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in y range.

        - ``z_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in z range.

        - ``output`` -- integer (default: unused) output to be selected when
          drawing the decision function if the perceptron has several output
          units, not specified when the perceptron has one output unit.

        - ``base`` -- matplotlib figure (default: a new figure) figure to be
          used to draw the decision function.

        - ``contours``-- iterable of numeric values (default: empty tuple)
          perceptron output values to be highlighted through contours.

        - ``contour_color`` -- iterable of colors or single color value
          (default: 'gray') colors of the drawn contours; if a single value is
          supplied, it refers to all contours.

        - ``contour_width`` -- iterable of numberic values or single numeric
          value (default: 1) width of the drawn contours; if a single numeric
          value is supplied, it refers to all contours.

        - ``contour_style`` -- iterable or single value of a valid style
          (default: '-') style of the drawn contours; if a single value is
          supplied, it refers to all contours.

        - ``color_bar`` -- boolean (default: False) flag setting the
          visualization of a color legend.

        - ``plotter`` -- Plotter object (default: SagePlotter() when the code
          is run within sage and MatplotlibPlotter() otherwise) to be used in
          order to render graphics.

        OUTPUT:

        graphic object -- the perceptron output values plot in function of
        possible input values, in form of a matplotlib figure or of a sage
        graphic.

        EXAMPLES:

        Another way to visualize a perceptron's behaviour is through the
        ``plot`` function, generating a graphic object summarizing the outputs
        for a given range of possible inputs:

        ::

            >>> from yaplf.utility.activation import SigmoidActivationFunction
            >>> from yaplf.models.neural import Perceptron
            >>> p = Perceptron(((4, 4),), threshold = (6,),
            ... activation = SigmoidActivationFunction(0.8))
            >>> p.plot((-5, 5), (-5, 5), plot_points = 100,
            ... contours = (0.1, 0.5, 0.9),
            ... contour_color = ('red', 'green', 'blue'), shading = True)

        Here the first two arguments represent the ranges for the possible
        values for the two perceptron input units, and the obtained graph
        contains a colored gradient shading from white to black in order to
        visualize how the perceptron output varies w.r.t. the possible input
        values (named argument ``shading``), highlighting through colored
        curves specific output values (where named arguments ``contours`` and
        ``contour_color`` specify these values and the color of the
        corresponding curves, while ``plot_points`` refers to the precision to
        be used in order to approximate those curves through a set of
        successive segments).

        Only perceptrons having two or three inputs allow invocation of the
        ``plot`` function. In the second case it will be necessary to specify
        three input value ranges, and the result will be a 3D graph:

        ::

            >>> p = Perceptron(((.3, 9.56, .2),), threshold=(1.7,),
            ... activation = SigmoidActivationFunction(beta = .1))
            >>> p.plot((-5, 5), (-5, 5), (-5, 5), plot_points = 20,
            ... contours=(0.1, 0.5, 0.9), contour_color = ('red', 'green',
            ... 'blue'), shading = True)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if not 1 < self.perceptron.get_num_inputs() < 4:
            raise ValueError('plot only works for 2-inputs and 3-inputs \
                perceptrons.')

        if len(args) != self.perceptron.get_num_inputs():
            raise ValueError('plot ranges incompatible with perceptron \
                inputs.')

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

        kwargs['gradient'] = shading
        kwargs['gradient_color'] = shading_color

        try:
            output_unit = kwargs['output']
            def classify(*args):
                """Classification function created on-the-fly."""

                return self.perceptron.decision_function(args)[output_unit]
        except KeyError:
            def classify(*args):
                """Classification function created on-the-fly."""

                return self.perceptron.decision_function(args)

        try:
            plotter = kwargs['plotter']
            del kwargs['plotter']
        except KeyError:
            plotter = PlotterFactory.get_plotter()

        return plotter.decision_function_plot(args, classify, **kwargs)

        #return Classifier.plot(self.perceptron, *args, **kwargs)


class PerceptronStatePlot(Plot):
    r"""
    """

    def __init__(self, perceptron):
        r"""
        """

        self.perceptron = perceptron

    def plot(self, *args, **kwargs):
        r"""
        Returns a matplotlib object containing a visual representation of
        the object.

        :param frame: (optional) flag setting the frame visualization (default
        value False)
        :type frame: boolean

        """

        fig = pyplot.figure()

        try:
            frame = kwargs['frame']
        except KeyError:
            frame = False

        if frame:
            axes = fig.add_subplot(111)
            axes.axis('equal')
        else:
            axes = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.0)
            axes.xaxis.set_visible(False)
            axes.yaxis.set_visible(False)

        num_input = self.perceptron.get_num_inputs()
        num_unit = self.perceptron.get_num_outputs()

        rad = 0.4
        half_distance = (num_input - num_unit) / 2.

        for i in range(num_unit):
            unit = Circle((half_distance + i + 1, 0.5), rad)
            axes.add_patch(unit)
            if self.perceptron.has_threshold:
                axes.text(half_distance + i + 1, 0.5, self.weights[i][-1],
                horizontalalignment='center', verticalalignment='center')

        for j in range(num_input):
            input_unit = Circle((j + 1, 2.5), rad, alpha=0.5, color='red')
            axes.add_patch(input_unit)

        for j in range(num_input):
            for i in range(num_unit):
                cos_theta = 2 / sqrt(4 + (j - half_distance - i) ** 2)
                sin_theta = sin(arccos(cos_theta))
                sgn = sign(half_distance + i - j)

                conn = Arrow(j + 1 + 1.05 * rad * sin_theta * sgn,
                    2.5 - 1.05 * rad * cos_theta,
                    half_distance + i - j - 2 * 1.05 * rad * sgn * sin_theta,
                    -2 + 2 * 1.05 * rad * cos_theta, 0.1)
                axes.add_patch(conn)

                alpha = .25
                axes.text(alpha * (j + 1) + (1 - alpha) * \
                    (half_distance + i + 1) - .07,
                    alpha * 2.5 + (1 - alpha) * 0.5, self.perceptron.weights[i][j],
                    horizontalalignment='right')

        axes.set_xlim(0, (num_input + num_unit) / 2 + 1)
        axes.set_ylim(0, 3)
        return fig


class MultilayerPerceptronDecisionFunctionPlot(Plot):
    r"""
    """

    def __init__(self, ml_perceptron):
        r"""
        """

        self.ml_perceptron = ml_perceptron

    def plot(self, *args, **kwargs):
        r"""
        Returns a graphic containing the plot of the perceptron output. Raises
        a ValueError if invoked on perceptrons not having two or three input
        units. The graphic is either a bi- or three-dimensional plot according
        to the following invocation syntax:

        - ``plot(x_range, y_range)`` returns a 2D plot.

        - ``plot(x_range, y_range, z_range)`` returns a 3D plot.

        INPUT:

        - ``self`` -- ``Perceptron`` object on which the function is invoked.

        - ``x_range`` -- bidimensional iterable containing the range of x
          variable.

        - ``y_range`` -- bidimensional iterable containing the range of y
          variable.

        - ``z_range`` -- bidimensional iterable containing the range of z
          variable.

        - ``shading`` -- boolean (default value: False) flag triggering the
          perceptron output visualization in form of a color gradient.

        - ``shading_color`` -- colormap (default value: Greys) colormap to be
          used in order to show perceptron output.

        - ``x_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in x range.

        - ``y_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in y range.

        - ``z_points`` -- integer (default: 50 in 2D, 35 in 3D) number of
          samples in z range.

        - ``output`` -- integer (default: unused) output to be selected when
          drawing the decision function if the perceptron has several output
          units, not specified when the perceptron has one output unit.

        - ``base`` -- matplotlib figure (default: a new figure) figure to be
          used to draw the decision function.

        - ``contours``-- iterable of numeric values (default: empty tuple)
          perceptron output values to be highlighted through contours.

        - ``contour_color`` -- iterable of colors or single color value
          (default: 'gray') colors of the drawn contours; if a single value is
          supplied, it refers to all contours.

        - ``contour_width`` -- iterable of numberic values or single numeric
          value (default: 1) width of the drawn contours; if a single numeric
          value is supplied, it refers to all contours.

        - ``contour_style`` -- iterable or single value of a valid style
          (default: '-') style of the drawn contours; if a single value is
          supplied, it refers to all contours.

        - ``color_bar`` -- boolean (default: False) flag setting the
          visualization of a color legend.

        - ``plotter`` -- Plotter object (default: SagePlotter() when the code
          is run within sage and MatplotlibPlotter() otherwise) to be used in
          order to render graphics.

        OUTPUT:

        graphic object -- the perceptron output values plot in function of
        possible input values, in form of a matplotlib figure or of a sage
        graphic.

        EXAMPLES:

        The ``plot`` function generates a graphic object summarizing the
        multilayer perceptron outputs for a given range of possible inputs.
        For instance, the following Heaviside-activated perceptron computes
        the binary XOR function, as shown by visual inspection of the plot:

        ::

            >>> from yaplf.models.neural import MuyltilayerPerceptron
            >>> p = MultilayerPerceptron([2, 2, 1], [[(1, -1), (-1, 1)],
            ... [(1, 1)]], thresholds = [(-1, -1), (-1,)])
            >>> p.plot((-0.1, 1.1), (-0.1, 1.1), plot_points = 100,
            ... shading = True)

        Here the first two arguments represent the ranges for the possible
        values for the two perceptron input units, and the obtained graph
        contains two black zones corresponding to the set of inputs mapped to
        `0`, while the remaining black zone corresponds to the output `1`.
        Visualization of these zones is activated by the ``shading`` named
        argument, while ``plot_points`` refers to the precision to be used in
        order to draw the above mentioned zones.

        ::

            >>> from yaplf.utility.activation import SigmoidActivationFunction
            >>> p = MultilayerPerceptron([2, 2, 1], [[(1, -1), (-1, 1)],
            ... [(1, 1)]], thresholds = [(-1, -1), (-1,)],
            ... activations = SigmoidActivationFunction(beta = 2))
            >>> p.plot((-0.1, 1.1), (-0.1, 1.1), shading=True,
            ... contours = (0.2, ), contour_color = ('red',))

        Only perceptrons having two or three inputs allow invocation of the
        ``plot`` function. In the second case it will be necessary to specify
        three input value ranges, and the result will be a 3D graph:

        ::

            >>> p = MultilayerPerceptron([3, 2, 1],
            ... [[(1, -1, -.5), (-1, 1, 1.3)], [(1, 1)]],
            ... thresholds = [(-1, -1), (-1,)],
            ... activations = SigmoidActivationFunction(beta = 2))
            >>> p.plot((-0.1, 1.1), (-0.1, 1.1), (0, 1),
            ... contours=(0.1, 0.5, 0.9),
            ... contour_color=('red', 'green', 'blue'))

        AUTHORS:

        - Dario Malchiodi (2010-03-22)

        """

        if not 1 < self.ml_perceptron.dimensions[0] < 4:
            raise ValueError('plot only works for 2-inputs and 3-inputs \
                perceptrons.')

        if len(args) != self.ml_perceptron.dimensions[0]:
            raise ValueError('plot ranges incompatible with perceptron \
                inputs.')

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

        kwargs['gradient'] = shading
        kwargs['gradient_color'] = shading_color

        return Classifier.plot(self.ml_perceptron, *args, **kwargs)

