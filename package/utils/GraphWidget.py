from __future__ import unicode_literals
import sys
import os
import random
from collections import Counter
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class GraphWidget(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100, graph="main"):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # FigureCanvas.__init__(self, fig)
        super(GraphWidget, self).__init__(fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def chartSingleClassFrequency(self, data):
        """Display a bar chart of frequencies per label
            # Arguments
                data: list, List of integer values corresponding to the actual
                question score.
        """
        def getNumClasses(labels):
            """Helper function to return the number of available labels
                # Throws
                    ValueError: if less than 2 classes or there are no samples with 
                    a given class.
            """
            num_classes = max(labels) + 1
            missing_classes = [i for i in range(num_classes) if i not in labels]
            if len(missing_classes):
                raise ValueError('Missing samples with label value(s) '
                                '{missing_classes}. Please make sure you have '
                                'at least one sample for every label value '
                                'in the range(0, {max_class})'.format(
                                    missing_classes=missing_classes,
                                    max_class=num_classes - 1))

            if num_classes <= 1:
                raise ValueError('Invalid number of labels: {num_classes}.'
                                'Please make sure there are at least two classes '
                                'of samples'.format(num_classes=num_classes))
            return num_classes
        num_classes = getNumClasses(data)
        count_map = Counter(data)
        counts = [count_map[i] for i in range(num_classes)]
        idx = np.arange(num_classes)
        self.axes.cla()
        self.axes.bar(idx, counts)
        self.axes.set_xlabel('Class')
        self.axes.set_ylabel('Number of Samples')
        self.axes.set_xticks(idx)
        self.draw()



# class MyDynamicMplCanvas(MyMplCanvas):
#     """A canvas that updates itself every second with a new plot."""

#     def __init__(self, *args, **kwargs):
#         MyMplCanvas.__init__(self, *args, **kwargs)
#         timer = QtCore.QTimer(self)
#         timer.timeout.connect(self.update_figure)
#         timer.start(1000)

#     def compute_initial_figure(self):
#         self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

#     def update_figure(self):
#         # Build a list of 4 random integers between 0 and 10 (both inclusive)
#         l = [random.randint(0, 10) for i in range(4)]
#         self.axes.cla()
#         self.axes.plot([0, 1, 2, 3], l, 'r')
#         self.draw()


