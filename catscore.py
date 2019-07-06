"""CAT Score is a machine-learning based scoring system developed for the 
Critical-thinking Assessment Test

@author pjtinker
"""

import argparse
import pandas as pd
import logging
import logging.handlers

# logging.basicConfig(filename='cat.log', format=logFormatter, level=logging.DEBUG)
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, 
                                QMainWindow, QSizePolicy, QWidget, 
                                QPushButton, QTabWidget, QMenuBar)

from package.train.cattrain import CatTrain

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.current_file = ''
        self.title = 'CAT Score'
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 400
        self.setWindowTitle(self.title)
        geometry = app.desktop().availableGeometry(self)
        self.setGeometry(10, 50, geometry.width() * 0.2, geometry.height() * 0.2)

        self.cat_train = CatTrain(self)

        self.main_layout = QVBoxLayout()
        self.open_cat_train_btn = QPushButton('CAT Train', self)
        self.open_cat_train_btn.clicked.connect(lambda: self.cat_train.show())
        self.main_layout.addWidget(self.open_cat_train_btn)


        self.setLayout(self.main_layout)

if __name__ == "__main__":
    import sys
    import logging
    import logging.handlers 
    """Setup logger for logging"""
    handler = logging.handlers.TimedRotatingFileHandler('cat.log', when='d', interval=1)
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(logFormatter)
    logging.basicConfig( handlers=[handler], format=logFormatter, level=logging.DEBUG)

    # Qt Application
    app = QApplication(sys.argv)
    # app.processEvents()
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())