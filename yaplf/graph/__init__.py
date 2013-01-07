
r"""
Package handling graphics in yaplf

Package yaplf.graph contains classes handling graphic generation in yaplf.

TODO:

- pep8 checked
- pylint score: 9.20

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@dsi.unimi.it>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************


from copy import deepcopy

from numpy import transpose, array, arange, shape

try:
    from sage.all import var
    from sage.plot.point import point
    from sage.plot.line import line
    from sage.plot.contour_plot import contour_plot
    from sage.plot.density_plot import density_plot
    from sage.plot.plot3d.implicit_plot3d import implicit_plot3d
except ImportError:
    try:
        from matplotlib import pyplot
        from matplotlib.axes3d import Axes3D
    except ImportError:
        raise ImportError("Neither sage or matplotlib found")



class Plot(object):
    r"""
    """

    def plot(self, *args, **kwargs):
        r"""
        """

        pass


class Plotter(object):
    r"""
    Base class for concrete plotters. Each subclass should implement the
    ``example_plot``, ``list_plot`` and ``decision_funcion_plot``, to be
    called in order to obtain a graph describing a data set, a generic list
    plot, and a decision function plot, respectively.

    Concrete subclasses handle the details of graphic creations in specific
    libraries. yaplf currently supports the subclasses SagePlotter and
    MatplotlibPlotter, respectively dealing with the sage environment [SAGE,
    Stein and Joyner, 2005] and the matplotlib library [Hunter, 2007].

    ``Plotter`` is basically a utility class, containing exclusively class
    methods. Moreover, those methods are not callable in the base class
    itself, as a ``ValueError`` is otherwise thrown.

    EXAMPLES:

    A default plotter is obtained through the ``PlotterFactory`` class:

    ::

        >>> from yaplf.graph import PlotterFactory
        >>> p = PlotterFactory.get_plotter()
        >>> p.list_plot(((1, 2), (-9, 6)))

    REFERENCES:

    [Hunter, 2007] John D. Hunter, Matplotlib: A 2D Graphics Environment,
    Computing in Science and Engineering, vol. 9, no. 3, pp. 90-95, May/June
    2007, doi:10.1109/MCSE.2007.55

    [SAGE] SAGE Mathematics Software, Version 4.3, http://www.sagemath.org/

    [Stein and Joyner, 2005] William Stein and David Joyner, SAGE: System for
    Algebra and Geometry Experimentation, Communications in Computer Algebra
    (SIGSAM Bulletin), July 2005

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    @classmethod
    def example_plot(cls, sample, **kwargs):
        r"""
        Returns a figure containing a colored scatter plot of the labeled
        sample passed as argument. Each pattern is represented through a
        bullet, colored according to the corresponding label. The function
        forwards to ``scatter`` named parameters with the exception of
        ``color_function``.

        ``plotter.example_plot(sample, ...)``

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``sample`` -- a sequence of ``Example`` objects.

        The following optional inputs can be specified through named arguments:

        - ``color_function`` -- function (default: function associating the
          color ``'black'`` whenever an example label is ``1``, ``'gray'``
          otherwise); function having as input a generic ``Example`` object and
          as value the corresponding colour.

        - ``size_function`` -- function (default: function associating the
           value``20`` independently of its argument); function having as input
          a generic ``Example`` object and as value the corresponding bullet
          size, expressed in pts^2.

        - ``alpha_function``-- function (default: function associating the
          value 1 independently of its argument, which means no transparency)
          function having as input a generic ``Example``object and as value the
          corresponding opacity.

        OUTPUT:

        figure -- a graph containing the example plot.

        EXAMPLES:

        Simple plot of the XOR binary function:

        ::

            >>> from yaplf.graph import PlotterFactory
            >>> p = PlotterFactory.get_plotter()
            >>> from yaplf.data import LabeledExample
            >>> xor_sample = (LabeledExample((0., 0.), 0.),
            ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), 0.))
            >>> p.example_plot(xor_sample)

        The same graph using a more sophisticated style, assigning green opaque
        bullets to examples having label set to ``1`` and yellow transparent
        bullets to the remaining examples, the bullet size being chosen
        according to its relative position w.r.t. `(\frac{1}{2}, \frac{1}{2})`:

        ::

            >>> cf = lambda x: ('green' if x.label == 1 else 'yellow')
            >>> sf = lambda x: (90 if x.pattern < (.5, .5) else 20)
            >>> af = lambda x: (.4 if x.label == 1 else .8)
            >>> p.example_plot(xor_sample, color_function = cf,
            ... size_function = sf, alpha_function = af)

        A 3D sample plot:

        ::

            >>> patterns = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), \
            ... (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
            >>> parity = lambda x: sum(x) % 2
            >>> parity_sample = [LabeledExample(e, parity(e)) \
            ... for e in patterns]
            >>> sf = lambda x: (20 if x.pattern[1] == 1 else 40)
            >>> af = lambda x: (.2 if x.pattern[0] == 1 else 1)
            >>> p.example_plot(parity_sample, color_function = cf,
            ... alpha_function = af, size_function = sf)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """
        raise NotImplementedError('example_plot() not callable in base class.')

    @classmethod
    def list_plot(cls, points, **kwargs):
        r"""
        Returns a sage figure containing a list plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``points`` -- list or tuple of 2D or 3D points to be plotted.

        - ``joined`` -- boolean (default: False) flag triggering the production
          of a graph whose points are joined instead of a scatter plot.

        - ``alpha`` -- number (default: not used) opacity value of the points
          (or lines) in the produced graph.

        - ``size`` -- integer (default: not used) size of the points (or lines)
          in the produced graph.

        Other named arguments affecting the graphic style are forwarded to
        matplotlib's ``plot`` or ``scatter``.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        The following instructions generate and show a figure showing three
        points:

        ::

            >>> from yaplf.graph import PlotterFactory
            >>> p = PlotterFactory.get_plotter()
            >>> points = ((1, 1), (3, -1), (7, 2))
            >>> p.list_plot(points)

        The same graph can be obtained joining the single points:

        ::

            >>> p.list_plot(points, joined = True)

        When ``joined`` is set to ``True``, the ``size``, ``color``, and
        ``alpha`` arguments affect respectively the line size, color, and
        opacity:

        ::

            >>> p.list_plot(points, joined = True, size = 3,
            ... alpha = .2)

        When the first argument of ``list_plot`` is a list or tuple of
        three-sized list or tuples, the result is a 3D graph:

        ::
            >>> points = ((1, 3, -4), (2, 1, 2), (1, 6, 5))
            >>> p.list_plot(points)


        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError('list_plot() not callable in base class.')

    @classmethod
    def decision_function_plot(cls, ranges, classify, **kwargs):
        r"""
        Returns a figure containing a decision function plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``ranges`` -- tuple of two or three ranges for the plot variables.

        - ``classify`` -- function to be used in order to get the decision
          function corresponding to a particular choice for the independent
          variables.

        - ``points`` -- number (default: not used) opacity value of the points
          (or lines) in the produced graph.

        - ``gradient`` -- boolean (default: ``False``) flag triggering colored
          visualization of the decision function.

        - ``gradient_color`` -- colormap or list/tuple of colors (default: hue
          colormap) set of colors to be used for the decision function colored
          visualization.

        - ``contours`` -- list or tuple of numbers (default: ()) decision
          function value to be highlighted through contours.

        - ``contour_color`` -- colormap or list/tuple of colors (default:
          Grays) set of colors to be used for contours.

        Other named arguments affecting the graphic style are forwarded to
        the lower-level functions within sage or matplotlib.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        All the default value of named arguments imply an empty graph:

        ::

            >>> cl = lambda x, y: x + y
            >>> from yaplf.graph import PlotterFactory
            >>> p = PlotterFactory.get_plotter()
            >>> p.decision_function_plot(((0, 1), (0, 1)), cl)
            0

        The ``gradient`` named argument triggers colored visualization:

        ::

            >>> p.decision_function_plot(((0, 1), (0, 1)), cl, gradient = True)

        Standard coloring can be overridden through specification of a color
        map or of a color tuple trhough the ``gradient_color`` named argument:

        ::

            >>> from matplotlib.cm import Blues
            >>> p.decision_function_plot(((0, 1), (0, 1)), cl,
            ... gradient = True, gradient_color = Blues)

        Likewise, contours can be obtained through the ``contours`` named
        argument:

        ::

            >>> p.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9))

        Note how previous figure contains three contours altough only two of
        them are visible, for the last one is colored in white (as the default
        coloring scheme for contours uses a graylevel gradient). This choice
        can either be overridden through specification of a tuple of colors to
        be circularly applied to all contours, or using a colormap. In both
        cases the named argument to be assigned is ``contour_color``:

        ::

            >>> p.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = ('gray',))
            >>> p.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = Blues)

        Clearly, gradient and contours can coexist:

        ::

            >>> p.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = ('gray',),
            ... gradient = True)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError(
            'decision_function_plot not callable in base class.')


class SagePlotter(Plotter):
    r"""
    Concrete plotter based on sage graphics.

    """

    @classmethod
    def example_plot(cls, sample, **kwargs):
        r"""
        Returns a sage figure containing a colored scatter plot of the labeled
        sample passed as argument. Each pattern is represented through a
        bullet, colored according to the corresponding label. The function
        forwards to ``scatter`` named parameters with the exception of
        ``color_function``.

        ``plotter.example_plot(sample, ...)``

        INPUT:

        - ``sample`` -- a sequence of ``Example`` objects.

        The following optional inputs can be specified through named arguments:

        - ``color_function`` -- function (default: function associating the
          color ``'black'`` whenever an example label is ``1``, ``'gray'``
          otherwise); function having as input a generic ``Example`` object and
          as value the corresponding colour.

        - ``size_function`` -- function (default: function associating the
          value ``20`` independently of its argument); function having as input
          a generic ``Example`` object and as value the corresponding bullet
          size, expressed in pts^2.

        - ``alpha_function``-- function (default: function associating the
          value 1 independently of its argument, which means no transparency)
          function having as input a generic ``Example``object and as value the
          corresponding opacity.

        OUTPUT:

        figure -- a graph containing the example plot.

        EXAMPLES:

        Simple plot of the XOR binary function:

        ::

            >>> from yaplf.data import LabeledExample
            >>> xor_sample = (LabeledExample((0., 0.), 0.),
            ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), 0.))
            >>> from yaplf.graph import SagePlotter
            >>> SagePlotter.example_plot(xor_sample)

        The same graph using a more sophisticated style, assigning green opaque
        bullets to examples having label set to ``1`` and yellow transparent
        bullets to the remaining examples, the bullet size being chosen
        according to its relative position w.r.t. `(\frac{1}{2}, \frac{1}{2})`:

        ::

            >>> cf = lambda x: ('green' if x.label == 1 else 'yellow')
            >>> sf = lambda x: (90 if x.pattern < (.5, .5) else 20)
            >>> af = lambda x: (.4 if x.label == 1 else .8)
            >>> SagePlotter.example_plot(xor_sample, color_function = cf, \
            ... size_function = sf, alpha_function = af)

        A 3D sample plot:

        ::

            >>> patterns = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), \
            ... (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
            >>> parity = lambda x: sum(x) % 2
            >>> parity_sample = [LabeledExample(e, parity(e)) \
            ... for e in patterns]
            >>> sf = lambda x: (20 if x.pattern[1] == 1 else 40)
            >>> af = lambda x: (.2 if x.pattern[0] == 1 else 1)
            >>> SagePlotter.example_plot(parity_sample, color_function = cf, \
            ... alpha_function = af, size_function = sf)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(sample[0].pattern) == 2:
            size_name = 'pointsize'
            alpha_name = 'alpha'
        elif len(sample[0].pattern) == 3:
            size_name = 'size'
            alpha_name = 'opacity'
        else:
            raise ValueError(
                'data plots only generated for 2D and 3D patterns.')

        try:
            color_function = kwargs['color_function']
            del kwargs['color_function']
        except KeyError:
            color_function = lambda e: ('black' if e.label == 1 else 'gray')

        try:
            size_function = kwargs['size_function']
            del kwargs['size_function']
        except KeyError:
            size_function = lambda e: 20

        try:
            alpha_function = kwargs['alpha_function']
            del kwargs['alpha_function']
        except KeyError:
            alpha_function = lambda e: 1

        def style_function(example):
            """Style function to be applied to points in scatter plot"""
            style = {size_name: size_function(example),
                alpha_name: alpha_function(example),
                'color': color_function(example)}

            return style

        points = []
        for elem in sample:
            kwargs.update(style_function(elem))
            points.append(point(elem.pattern, **kwargs))

        return sum(points)

    @classmethod
    def list_plot(cls, points, **kwargs):
        r"""
        Returns a sage figure containing a list plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``points`` -- list or tuple of 2D or 3D points to be plotted.

        - ``joined`` -- boolean (default: False) flag triggering the production
          of a graph whose points are joined instead of a scatter plot.

        - ``alpha`` -- number (default: not used) opacity value of the points
          (or lines) in the produced graph.

        - ``size`` -- integer (default: not used) size of the points (or lines)
          in the produced graph.

        Other named arguments affecting the graphic style are forwarded to
        matplotlib's ``plot`` or ``scatter``.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        The following instructions generate and show a figure showing three
        points:

        ::

            >>> points = ((1, 1), (3, -1), (7, 2))
            >>> from yaplf.graph import SagePlotter
            >>> SagePlotter.list_plot(points)

        The same graph can be obtained joining the single points:

        ::

            >>> SagePlotter.list_plot(points, joined = True)

        When ``joined`` is set to ``True``, the ``size``, ``color``, and
        ``alpha`` arguments affect respectively the line size, color, and
        opacity:

        ::

            >>> SagePlotter.list_plot(points, joined = True, size = 3,
            ... alpha = .2)

        When the first argument of ``list_plot`` is a list or tuple of
        three-sized list or tuples, the result is a 3D graph:

        ::
            >>> points = ((1, 3, -4), (2, 1, 2), (1, 6, 5))
            >>> SagePlotter.list_plot(points)


        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            joined = kwargs['joined']
            del kwargs['joined']
        except KeyError:
            joined = False

        if len(shape(points)) == 1:
            points = zip(range(len(points)), points)

        if len(points[0]) == 2:
            try:
                size = kwargs['size']
                del kwargs['size']
                if joined:
                    kwargs['thickness'] = size
                else:
                    kwargs['pointsize'] = size
            except KeyError:
                pass
        elif len(points[0]) == 3:
            try:
                alpha = kwargs['alpha']
                del kwargs['alpha']
                kwargs['opacity'] = alpha
                if joined:
                    size = kwargs['size']
                    del kwargs['size']
                    kwargs['thickness'] = size
            except KeyError:
                pass
        else:
            raise ValueError('scatter() only available for 2D and 3D points')

        if joined:
            return line(points, **kwargs)
        else:
            return point(points, **kwargs)

    @classmethod
    def decision_function_plot(cls, ranges, classify, **kwargs):
        r"""
        Returns a sage figure containing a decision function plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``ranges`` -- tuple of two or three ranges for the plot variables.

        - ``classify`` -- function to be used in order to get the decision
          function corresponding to a particular choice for the independent
          variables.

        - ``points`` -- number (default: not used) opacity value of the points
          (or lines) in the produced graph.

        - ``gradient`` -- boolean (default: ``False``) flag triggering colored
          visualization of the decision function.

        - ``gradient_color`` -- colormap or list/tuple of colors (default: hue
          colormap) set of colors to be used for the decision function colored
          visualization.

        - ``contours`` -- list or tuple of numbers (default: ()) decision
          function value to be highlighted through contours.

        - ``contour_color`` -- colormap or list/tuple of colors (default:
          Grays) set of colors to be used for contours.

        Other named arguments affecting the graphic style are forwarded to
        ``density_plot``, ``contour_plot`` and ``implicit_plot3d``.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        All the default value of named arguments imply an empty graph:

        ::

            >>> cl = lambda x, y: x + y
            >>> from yaplf.graph import SagePlotter
            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl)
            0

        The ``gradient`` named argument triggers colored visualization:

        ::

            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... gradient = True)

        Standard coloring can be overridden through specification of a color
        map or of a color tuple trhough the ``gradient_color`` named argument:

        ::

            >>> from matplotlib.cm import Blues
            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... gradient = True, gradient_color = Blues)

        Likewise, contours can be obtained through the ``contours`` named
        argument:

        ::

            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9))

        Note how previous figure contains three contours altough only two of
        them are visible, for the last one is colored in white (as the default
        coloring scheme for contours uses a graylevel gradient). This choice
        can either be overridden through specification of a tuple of colors to
        be circularly applied to all contours, or using a colormap. In both
        cases the named argument to be assigned is ``contour_color``:

        ::

            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = ('gray',))
            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = Blues)

        Clearly, gradient and contours can coexist:

        ::

            >>> SagePlotter.decision_function_plot(((0, 1), (0, 1)), cl,
            ... contours = (.2, .5, .9), contour_color = ('gray',),
            ... gradient = True)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(ranges) < 2 or len(ranges) > 3:
            raise ValueError(
                'decision_function_plot only works for 2D or 3D data')

        x_range = ranges[0]
        y_range = ranges[1]

        try:
            gradient = kwargs['gradient']
            del kwargs['gradient']
        except KeyError:
            gradient = False

        try:
            gradient_color = kwargs['gradient_color']
            del kwargs['gradient_color']
        except KeyError:
            gradient_color = False

        try:
            contour_value = kwargs['contours']
            del kwargs['contours']
        except KeyError:
            contour_value = ()

        try:
            contour_color = kwargs['contour_color']
            del kwargs['contour_color']
        except KeyError:
            contour_color = 'gray'

        #ignores contour_style and contour_width options as the corresponding
        #styles are not (yet) supported in sage
        #ignores base as it is only intended for matlib
        try:
            del kwargs['contour_style']
            del kwargs['contour_width']
            del kwargs['base']
        except KeyError:
            pass

        plots = []

        if len(ranges) == 2:
            if gradient:
                plots.append(density_plot(classify, x_range, y_range,
                    cmap=gradient_color, **kwargs))

            if contour_value:
                plots.append(contour_plot(classify, x_range, y_range,
                    contours=contour_value, cmap=contour_color,
                    fill=False, **kwargs))
        else:
            z_range = ranges[2]
            x_var = var('x')
            y_var = var('y')
            z_var = var('z')

            plots = [implicit_plot3d(classify(x_var, y_var, z_var), \
                (x_var,) + x_range, (y_var,) + y_range, (z_var,) + z_range,
                contour=level, color=c, **kwargs)
                for level, c in zip(contour_value, contour_color)]

        return sum(plots)


class MatplotlibPlotter(Plotter):
    """Concrete plotter based on matplotlib
    """

    @classmethod
    def example_plot(cls, sample, **kwargs):
        r"""
        Returns a matplotlib figure containing a colored scatter plot of the
        labeled sample passed as argument. Each pattern is represented through
        a bullet, colored according to the corresponding label. The function
        forwards to ``scatter`` named parameters with the exception of
        ``color_function``.

        ``plotter.example_plot(sample, ...)``

        INPUT:

        - ``sample`` -- a sequence of ``Example`` objects.

        The following optional inputs can be specified through named arguments:

        - ``color_function`` -- function (default: function associating the
          color ``'black'`` whenever an example label is ``1``, ``'gray'``
          otherwise); function having as input a generic ``Example`` object and
          as value the corresponding colour.

        - ``size_function`` -- function (default: function associating the
          value ``20`` independently of its argument); function having as input
          a generic ``Example`` object and as value the corresponding bullet
          size, expressed in pts^2.

        - ``alpha_function``-- function (default: function associating the
          value 1 independently of its argument, which means no transparency)
          function having as input a generic ``Example``object and as value the
          corresponding opacity.

        - ``base`` -- matplotlib figure (default: a newly generated matplotlib
          figure) figure to be used to draw the scatter plot onto.

        OUTPUT:

        figure -- a graph containing the example plot.

        EXAMPLES:

        Simple plot of the XOR binary function:

        ::

            >>> from yaplf.data import LabeledExample
            >>> xor_sample = (LabeledExample((0., 0.), 0.),
            ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.),
            ... LabeledExample((1., 1.), 0.))
            >>> from yaplf.graph import MatplotlibPlotter
            >>> f = MatplotlibPlotter.example_plot(xor_sample)
            >>> f.savefig('plot-1.png')

        The same graph using a more sophisticated style, assigning green opaque
        bullets to examples having label set to ``1`` and yellow transparent
        bullets to the remaining examples, the bullet size being chosen
        according to its relative position w.r.t. `(\frac{1}{2}, \frac{1}{2})`:

        ::

            >>> cf = lambda x: ('green' if x.label == 1 else 'yellow')
            >>> sf = lambda x: (90 if x.pattern < (.5, .5) else 20)
            >>> af = lambda x: (.4 if x.label == 1 else .8)
            >>> f = MatplotlibPlotter.example_plot(xor_sample,
            ... color_function = cf, size_function = sf, alpha_function = af)
            >>> f.savefig('plot-2.png')

        A 3D sample plot:

        ::

            >>> patterns = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), \
            ... (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
            >>> parity = lambda x: sum(x) % 2
            >>> parity_sample = [LabeledExample(e, parity(e)) \
            ... for e in patterns]
            >>> sf = lambda x: (20 if x.pattern[1] == 1 else 40)
            >>> af = lambda x: (.2 if x.pattern[0] == 1 else 1)
            >>> f = MatplotlibPlotter.example_plot(parity_sample,
            ... color_function = cf, alpha_function = af, size_function = sf)
            >>> f.savefig('plot-3.png')

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        patterns = [elem for elem in sample]

        try:
            color_function = kwargs['color_function']
            del kwargs['color_function']
        except KeyError:
            color_function = lambda x: ('black' if x.label == 1 else 'gray')
        color_values = [color_function(p) for p in patterns]
        # was
        # color_values = map(color_function, patterns)
        kwargs['c'] = color_values

        try:
            size_function = kwargs['size_function']
            del kwargs['size_function']
            size_values = [size_function(p) for p in patterns]
            # was
            # size_values = map(size_function, patterns)
            kwargs['s'] = size_values
        except KeyError:
            pass

        try:
            alpha_function = kwargs['alpha_function']
            del kwargs['alpha_function']
            # alpha_function argument can only be used in order to set a *same*
            # transparency value for all examples, as matplotlib's scatter()
            # do not allow multiple transparency values; the selected
            # transparency valuewill be that returned by alpha_function on the
            # first example
            kwargs['alpha'] = alpha_function(sample[0])

        except KeyError:
            pass

        try:
            fig = kwargs['base']
            del kwargs['base']
        except KeyError:
            fig = pyplot.figure()
        if len(sample[0].pattern) == 2:
            axes = fig.add_subplot(111)
        elif len(sample[0].pattern) == 3:
            axes = Axes3D(fig)
        else:
            raise ValueError('Classification data plot only available for \
                bi- and three-dimensional data.')

        values = [[elem.pattern[i] for elem in sample]
            for i in range(len(sample[0].pattern))]
        axes.scatter(*values, **kwargs)

        return fig

    @classmethod
    def list_plot(cls, points, **kwargs):
        r"""
        Returns a matplotlib figure containing a list plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``points`` -- list or tuple of 2D or 3D points to be plotted.

        - ``base`` -- matplotlib figure (default: a newly generated matplotlib
          figure) figure to be used to draw the scatter plot onto.

        - ``joined`` -- boolean (default: False) flag triggering the production
          of a graph whose points are joined instead of a scatter plot.

        - ``size`` -- integer (default: not used) size of the points (or lines)
          in the produced graph.

        Other named arguments affecting the graphic style are forwarded to
        matplotlib's ``plot`` or ``scatter``.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        The following instructions generate and save a figure showing three
        points:

        ::

            >>> points = ((1, 1), (3, -1), (7, 2))
            >>> from yaplf.graph import MatplotlibPlotter
            >>> f = MatplotlibPlotter.list_plot(points)
            >>> f.savefig('plot-1.png')

        The same graph can be obtained joining the single points:

        ::

            >>> f = MatplotlibPlotter.list_plot(points, joined = True)
            >>> f.savefig('plot-2.png')

        When ``joined`` is set to ``True``, the ``size`` and ``color``
        arguments affect respectively the line size and color:

        ::

            >>> f = MatplotlibPlotter.list_plot(points, joined = True,
            ... size = 3, color = 'red')
            >>> f.savefig('plot-3.png')

        When the first argument of ``list_plot`` is a list or tuple of
        three-sized list or tuples, the result is a 3D graph:

        ::
            >>> points = ((1, 3, -4), (2, 1, 2), (1, 6, 5))
            >>> f = MatplotlibPlotter.list_plot(points, joined = True)
            >>> f.savefig('plot-4.png')


        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        try:
            fig = kwargs['base']
            del kwargs['base']
        except KeyError:
            fig = pyplot.figure()

        try:
            joined = kwargs['joined']
            del kwargs['joined']
        except KeyError:
            joined = False

        try:
            size = kwargs['size']
            del kwargs['size']
            if joined:
                kwargs['linewidth'] = size
            else:
                kwargs['s'] = size
        except KeyError:
            pass

        coords = transpose(points)

        if len(coords) == 2:
            axes = fig.add_subplot(111)
            if joined:
                axes.plot(*coords, **kwargs)
            else:
                axes.scatter(*coords, **kwargs)
        elif len(coords) == 3:
            axes = Axes3D(fig)
            if joined:
                axes.plot(*coords, **kwargs)
            else:
                axes.scatter(*coords, **kwargs)
        return fig

    @classmethod
    def decision_function_plot(cls, ranges, classify, **kwargs):
        """Returns a matplotlib object containing the plot of the model's
        decision function. Each subclass of model shoud check that a
        bidimensional plot can be produced, and use the returned matplotlib
        figure as base decision function. If a model outputs a number, that
        number is used in order to build the decision function. If the output
        is array-like, the named argument 'output' should be used in order to
        select wich component in the array has to be chosen for building the
        decision function.

        Parameters are specified as in plot().

        Returns a sage figure containing a decision function plot.

        INPUT:

        - ``cls`` -- class on which the function is invoked.

        - ``ranges`` -- tuple of two or three ranges for the plot variables.

        - ``classify`` -- function to be used in order to get the decision
          function corresponding to a particular choice for the independent
          variables.

        - ``points`` -- number (default: not used) opacity value of the points
          (or lines) in the produced graph.

        - ``gradient`` -- boolean (default: ``False``) flag triggering colored
          visualization of the decision function.

        - ``gradient_color`` -- colormap or list/tuple of colors (default: hue
          colormap) set of colors to be used for the decision function colored
          visualization.

        - ``contours`` -- list or tuple of numbers (default: ()) decision
          function value to be highlighted through contours.

        - ``contour_color`` -- colormap or list/tuple of colors (default:
          Grays) set of colors to be used for contours.

        Other named arguments affecting the graphic style, e.g. ``colopr_bar``,
        are forwarded to
        ``imshow``, ``contour`` and ``plot3d``.

        OUTPUT:

        figure containing a list plot.

        EXAMPLES:

        All the default value of named arguments imply an empty graph:

        ::

            >>> cl = lambda x, y: x + y
            >>> from yaplf.graph import MatplotlibPlotter
            >>> fig = MatplotlibPlotter.decision_function_plot(((0, 1),
            ... (0, 1)), cl)
            >>> fig.savefig('df-1.png')

        The ``gradient`` named argument triggers colored visualization. In
        this case a specific coloring should be specified through assignment
        of a color map or a color tuple to the ``gradient_color`` named
        argument:

        ::

            >>> from matplotlib.cm import Blues
            >>> fig = MatplotlibPlotter.decision_function_plot(((0, 1),
            ... (0, 1)), cl, gradient = True, gradient_color = Blues)
            >>> fig.savefig('df-2.png')

        Likewise, contours can be obtained through the ``contours`` named
        argument:

        ::

            >>> fig = MatplotlibPlotter.decision_function_plot(((0, 1),
            ... (0, 1)), cl, contours = (.2, .5, .9))
            >>> fig.savefig('df-3.png')

        The contour coloring can be overridden assignint to the named argument
         ``contour_color`` a tuple of colors to be circularly applied to all
         contours:

        ::

            >>> fig = MatplotlibPlotter.decision_function_plot(((0, 1),
            ... (0, 1)), cl, contours = (.2, .5, .9), contour_color = ('red',))
            >>> fig.savefig('df-4.png')

        Clearly, gradient and contours can coexist:

        ::

            >>> fig = MatplotlibPlotter.decision_function_plot(((0, 1),
            ... (0, 1)), cl, contours = (.2, .5, .9),
            ... contour_color = ('gray',), gradient = True,
            ... gradient_color = Blues)

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        if len(ranges) < 2 or len(ranges) > 3:
            raise ValueError(
                'decision_function_plot only usable with 2D or 3D data')

        x_range = ranges[0]
        y_range = ranges[1]

        try:
            points = kwargs['points']
            del kwargs['points']
            if type(points) is not type(()) and type(points) is not type([]):
                points = (points,) * len(ranges)
            if len(points) != len(ranges):
                raise ValueError('Ranges and point number specifications \
                    refer to different dimensions')
        except KeyError:
            points = (40,) * len(ranges)

        x_points = points[0]
        y_points = points[1]

        x_start, x_end = x_range
        y_start, y_end = y_range

        x_values = arange(x_start, x_end, 1.0 * (x_end - x_start) / x_points)
        y_values = arange(y_start, y_end, 1.0 * (y_end - y_start) / y_points)

        if len(ranges) == 3:
            z_range = ranges[2]
            z_points = points[2]
            z_start, z_end = z_range
            z_values = arange(z_start, z_end,
                1.0 * (z_end - z_start) / z_points)

        try:
            fig = kwargs['base']
            del kwargs['base']
        except KeyError:
            fig = pyplot.figure()

        try:
            gradient = kwargs['gradient']
            del kwargs['gradient']
        except KeyError:
            gradient = False

        try:
            gradient_color = kwargs['gradient_color']
            del kwargs['gradient_color']
        except KeyError:
            gradient_color = False

        try:
            contour_value = kwargs['contours']
            del kwargs['contours']
        except KeyError:
            contour_value = ()

        try:
            contour_color = kwargs['contour_color']
            del kwargs['contour_color']
        except KeyError:
            contour_color = 'gray'

        if len(ranges) == 2:
            axes = fig.add_subplot(111)
            z_values = array([[classify(x, y) for x in x_values]
                for y in y_values])
            filtered_args = deepcopy(kwargs)
            try:
                del filtered_args['contour_style']
                del filtered_args['contour_width']
                del filtered_args['color_bar']
            except KeyError:
                pass

            if gradient:
                grad = axes.imshow(z_values, origin='lower',
                    cmap=gradient_color,
                    extent=(x_start, x_end, y_start, y_end),
                    **filtered_args)

            try:
                color_bar = kwargs['color_bar']
            except KeyError:
                color_bar = False

            try:
                contour_style = kwargs['contour_style']
                if type(contour_style) != type(()) and \
                    type(contour_style) != type([]):
                    contour_style = (contour_style,) * len(contour_value)
            except KeyError:
                contour_style = ('-',) * len(contour_value)
            try:
                contour_width = kwargs['contour_width']
                if type(contour_width) != type(()) and \
                    type(contour_width) != type([]):
                    contour_width = (contour_width,) * len(contour_value)
            except KeyError:
                contour_width = (1,) * len(contour_value)

            if contour_value:
                axes.contour(z_values, contour_value,
                    extent=(x_start, x_end, y_start, y_end),
                    colors=contour_color, linestyles=contour_style,
                    linewidths=contour_width, **filtered_args)

            if color_bar and gradient:
                #fig.colorbar(cs, shrink = 0.8, extend = 'both')
                fig.colorbar(grad, orientation='vertical', **filtered_args)
        else:
            axs = Axes3D(fig)
            if contour_value:
                for level, col in zip(contour_value, contour_color):
                    contour = transpose([(x, y, z)
                        for x in x_values for y in y_values for z in z_values
                        if level - 0.1 < classify(x, y, z) < level + 0.1])
                    axs.plot3D(contour[0], contour[1], contour[2], color=col,
                        alpha=0.5)
        return fig


class PlotterFactory(object):
    r"""
    Factory allowing to automatically get a default ``Plotter`` instance in
    function of the environment within which the code is executed.

    INPUT:

    - ``cls`` -- class on which the function is executed.

    OUTPT:

    ``Plotter`` subclass object specialized for the used environment.

    EXAMPLES:

    A default plotter is obtained through the ``get_plotter`` function:

    ::

        >>> from yaplf.graph import PlotterFactory
        >>> p = PlotterFactory.get_plotter()

    Subsequently, this plotter can be used in order to actually build figures,
    such as in the following instruction:

    ::
            >>> p.list_plot(((1, 1), (3, -1), (7, 2)))

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    @classmethod
    def get_plotter(cls):
        r"""
        See ``PlotterFactory`` for full documentation.

        """

        try:
            import sage.all
            return SagePlotter
        except ImportError:
            return MatplotlibPlotter


def classification_data_plot(sample, **kwargs):
    r"""
    Generate a plot describing a sample for classification problems.

    Return a figure containing a colored scatter plot of the
    labeled sample passed as argument. Each example is represented through a
    bullet, whose position is identified by the example's pattern, while its
    style depend in general on the whole example, although it is typically
    colored according to the label.

    The function detects whether it has been invoked in a pure python
    environment or within sage, and returns accordingly a matplotlib figure
    or a graphic sage object. As an alternative, the kind of returned object
    can be explicitely set through the ``plotter`` named argument.

    ``classification_data_plot(sample, ...)``

    INPUT:

    - ``sample`` -- a sequence of ``Example`` objects

    The following optional inputs can be specified through named arguments:

    - ``color_function`` -- function (default: function associating the color
      ``'black'`` whenever an example label is ``1``, ``'gray'`` otherwise);
      function having as input a generic ``Example`` object and as value the
      corresponding colour;

    - ``size_function`` -- function (default: function associating the value
      ``20`` independently of its argument); function having as input a
      generic ``Example`` object and as value the corresponding bullet size,
      expressed in pts^2;

    - ``alpha_function`` -- function (default: function associating the value
      ``1`` independently of its argument, which means no transparency);
      function having as input a generic ``Example`` object and as value the
      corresponding bullet opacity;

    - ``plotter`` -- ``Plotter`` class (default: ``SagePlotter`` or
      ``MatplotlibPlotter`` class according to the execution environment --
      Sage or a pure python environment); plotter utility class to which the
      actual graphic generation is to be delegated.

    OUTPUT:

    figure -- classification data plot.

    EXAMPLES:

    Simple plot of the XOR binary function:

    ::

        >>> from yaplf.data import LabeledExample
        >>> from yaplf.graph import classification_data_plot
        >>> xor_sample = (LabeledExample((0., 0.), 0.), \
        ... LabeledExample((1., 0.), 1.), LabeledExample((0., 1.), 1.), \
        ... LabeledExample((1., 1.), 0.))
        >>> classification_data_plot(xor_sample)

    When executed within sage, the above command generates and displays a
    graphic object, either in a notebook or through a helper application. In
    the remaining cases, a matplotlib figure object is returned::

    The same graph using a more sophisticated style, assigning green opaque
    bullets to examples having label set to ``1`` and yellow transparent
    bullets to the remaining examples, the bullet size being chosen according
    to its relative position w.r.t. `(\frac{1}{2}, \frac{1}{2})`:

    ::

        >>> cf = lambda x: ('green' if x.label==1 else 'yellow')
        >>> sf = lambda x: (90 if x.pattern<(.5, .5) else 20)
        >>> af = lambda x: (.4 if x.label == 1 else .8)
        >>> classification_data_plot(xor_sample, color_function = cf, \
        ... size_function = sf, alpha_function = af)

    A 3D sample plot:

    ::

        >>> patterns = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), \
        ... (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
        >>> parity = lambda x: sum(x) % 2
        >>> parity_sample = [LabeledExample(e, parity(e)) for e in patterns]
        >>> sf = lambda x: (20 if x.pattern[1] == 1 else 40)
        >>> af = lambda x: (.2 if x.pattern[0] == 1 else 1)
        >>> classification_data_plot(parity_sample, color_function = cf, \
        ... alpha_function = af, size_function = sf)

    The first graph explicitly exploiting matplotlib:

    ::

        >>> from yaplf.graph import MatplotlibPlotter
        >>> fig = classification_data_plot(xor_sample, \
        ... plotter = MatplotlibPlotter)
        >>> fig.savefig('xor-plot.png')

    IMPLEMENTATION-DEPENDENT ISSUES:

    When using ``MatplotlibPlotter`` the following issues should be
    considered:

    - the alpha function is applied to the first example in order to get all
      bullets opacity. This parameter behaves differently than its base class
      as matplotlib's ``scatter()`` doesn't allow multiple transparency
      values;

    - the additional ``base``named parameter allow to specify a base figure to
      be used to draw the scatter plot;

    - the function delegates to matplotlib's ``scatter()`` additional named
      parameters, e.g. ``marker``.

    ::

        >>> fig = classification_data_plot(xor_sample, color_function = cf, \
        ... size_function = sf, plotter = MatplotlibPlotter, marker = 'v')
        >>> fig.savefig('mpl_dataplot.png')

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    try:
        plotter = kwargs['plotter']
        del kwargs['plotter']
    except KeyError:
        plotter = PlotterFactory.get_plotter()

    return plotter.example_plot(sample, **kwargs)