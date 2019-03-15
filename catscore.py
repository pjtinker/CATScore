"""CAT Score is a machine-learning based scoring system developed for the 
Critical-thinking Assessment Test

@author pjtinker
"""

import argparse
import pandas as pd

from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, Slot)
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QAction, QApplication, QHBoxLayout, QVBoxLayout, QHeaderView,
                               QMainWindow, QSizePolicy, QTableView, QWidget, QPushButton, QTabWidget)
from PySide2.QtCharts import QtCharts

from package.train.cattrain import CatTrain

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.current_file = ''
        self.title = 'CAT Score'
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 400
        self.setWindowTitle(self.title)
        geometry = app.desktop().availableGeometry(self)
        self.setGeometry(0, 0, geometry.width() * 0.2, geometry.height() * 0.2)

        self.cat_train = CatTrain(self)

        self.main_layout = QVBoxLayout()
        self.open_cat_train_btn = QPushButton('CAT Train', self)
        self.open_cat_train_btn.clicked.connect(lambda: self.cat_train.show())
        self.main_layout.addWidget(self.open_cat_train_btn)

        self.setLayout(self.main_layout)

if __name__ == "__main__":
    import sys
    # Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())