"""CAT Score is a machine-learning based scoring system developed for the 
Critical-thinking Assessment Test

@author pjtinker
"""

import argparse
import pandas as pd
import logging
import logging.handlers
from dask.distributed import Client

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, 
                                QMainWindow, QSizePolicy, QWidget, QGridLayout,
                                QPushButton, QTabWidget, QMenuBar, QSplashScreen)
from PyQt5.QtGui import QPixmap, QMovie, QIcon
from PyQt5.QtCore import Qt, pyqtSlot, QEventLoop

from multiprocessing import Pool

# from PyQt5 import Qt.WindowStaysOnTopHint

from package.train.cattrain import CatTrain
from package.evaluate.catevaluate import CatEvaluate

class SplashScreen(QSplashScreen):
    def __init__(self, animation, flags):
        QSplashScreen.__init__(self, QPixmap(), flags)
        self.movie = QMovie(animation)
        self.movie.frameChanged.connect(self.onNextFrame)
        self.movie.start()

    @pyqtSlot()
    def onNextFrame(self):
        pixmap = self.movie.currentPixmap()
        self.setPixmap(pixmap)
        self.setMask(pixmap.mask())


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.current_file = ''
        self.title = 'CAT Score'
        self.setWindowIcon(QIcon('./icons/cat-silhouette.jpg'))
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 400
        self.setWindowTitle(self.title)
        geometry = app.desktop().availableGeometry(self)
        self.setGeometry(10, 50, geometry.width() * 0.2, geometry.height() * 0.2)
        self.catscore = QWidget()
        self.main_layout = QVBoxLayout()
        self.setCentralWidget(self.catscore)
        self.catscore.setLayout(self.main_layout)
        self.cat_train = CatTrain(self)
        self.cat_evaluate = CatEvaluate(self)

        self.cat_train.version_widget.version_created.connect(self.cat_evaluate.score_widget.data_predictor.add_new_version)
        self.cat_train.train_widget.model_widget.comms.version_change.connect(self.cat_evaluate.score_widget.data_predictor.update_version)
        self.button_grid = QGridLayout()

        self.open_cat_train_btn = QPushButton('CAT &Train', self)
        self.open_cat_train_btn.clicked.connect(lambda: self.cat_train.show())
        self.button_grid.addWidget(self.open_cat_train_btn, 0, 0)

        self.open_cat_evaluate_btn = QPushButton('CAT &Evaluate', self)
        self.open_cat_evaluate_btn.clicked.connect(lambda: self.cat_evaluate.show())
        self.button_grid.addWidget(self.open_cat_evaluate_btn, 1, 0)

        self.main_layout.addLayout(self.button_grid)


def initialize_everything():
    handler = logging.handlers.RotatingFileHandler('cat.log', mode='a', maxBytes=1000)
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(logFormatter)
    logging.basicConfig( handlers=[handler], format=logFormatter, level=logging.DEBUG)
    # dask client for paralellization
    client = Client()
    return 0

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    splash = QSplashScreen(QPixmap('./icons/busycat.gif'), Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    # init_loop = QEventLoop()
    # pool = Pool(processes=1)
    # pool.apply_async(initialize_everything, [2], callback=lambda exitCode: init_loop.exit(exitCode))
    # init_loop.exec_()

    import logging
    import logging.handlers 
    
    """Setup logger for logging"""
    handler = logging.handlers.TimedRotatingFileHandler('cat.log', when='W6', interval=1)
    # handler = logging.FileHandler('cat.log')
    # handler = logging.handlers.RotatingFileHandler('cat.log', mode='a', maxBytes=1000)
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(logFormatter)
    logging.basicConfig( handlers=[handler], format=logFormatter, level=logging.DEBUG)
    logging.captureWarnings(True)
    # dask client for paralellization
    # Qt Application
    # splash_pix = QPixmap('./icons/busycat.gif', Qt.WindowStaysOnTopHint)
    client = Client()
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())