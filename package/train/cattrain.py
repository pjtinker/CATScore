'''
CATTrain module contains the all functionality associated with the training
 and tuning of machine-learning models.

@author pjtinker
'''

import sys
import argparse
import pandas as pd

from PySide2.QtCore import (Qt, Slot)
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QHeaderView,
                               QMainWindow, QSizePolicy, QProgressBar, QWidget)

from package.train.TrainWidget import TrainWidget

class CatTrain(QMainWindow):
    """ The central widget for the training component of CATScore
        Most of the functionality is contained in this class
    """
    def __init__(self, parent=None):
        super(CatTrain, self).__init__(parent)
        self.title = 'CAT Train'
        self.setWindowTitle(self.title)
        geometry = QApplication.desktop().availableGeometry(self)
        self.setGeometry(0, 0, geometry.width() * 0.45, geometry.height() * 0.5)
        self.statusBar().showMessage('Cut me, Mick!')
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)
        # self.progressBar.setValue(50)
        self.statusBar().addPermanentWidget(self.progressBar)
        self.train_widget = TrainWidget(self)
        self.train_widget.data_loader.update_progressbar.connect(self.updateStatusbar)
        self.setCentralWidget(self.train_widget)

    def closeEvent(self, event):
        print("closeEvent fired")

    @Slot(int, bool)
    def updateStatusbar(self, val, pulse):
        if pulse:
            self.progressBar.setRange(0,0)
        else:
            self.progressBar.setRange(0, 1)

        self.progressBar.setValue(val)


if __name__ == "__main__":
    import sys
    # Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CatTrain()
    window.show()
    sys.exit(app.exec_())