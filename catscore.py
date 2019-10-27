"""CAT Score is a machine-learning based scoring system developed for the 
Critical-thinking Assessment Test

@author pjtinker
"""

import argparse
import pandas as pd
import logging
import logging.handlers

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QProgressBar,
                                QMainWindow, QSizePolicy, QWidget, QGridLayout,
                                QPushButton, QTabWidget, QMenuBar, QSplashScreen)
from PyQt5.QtGui import QPixmap, QMovie, QIcon, QImage, QPalette, QBrush
from PyQt5.QtCore import Qt, pyqtSlot, QEventLoop, QSize

# from multiprocessing import Pool
# from dask.distributed import Client

# from package.train.cattrain import CatTrain
# from package.evaluate.catevaluate import CatEvaluate

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
        # background_img = QImage('./icons/cat-silhouette.jpg')
        # background_img_scaled = background_img.scaled(QSize(240, 280))
        # palette = QPalette()
        # palette.setBrush(10, QBrush(background_img_scaled))
        # self.setPalette(palette)
        self.left = 0
        self.top = 0
        self.width = 250
        self.height = 300
        self.setWindowTitle(self.title)
        geometry = app.desktop().availableGeometry(self)
        self.setGeometry(10, 50, geometry.width() * 0.15, geometry.height() * 0.15)
        # self.setGeometry(10, 50, self.width, self.height)
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
        self.open_cat_train_btn.setMinimumHeight(45)
        # self.open_cat_train_btn.setMaximumWidth(150)
        self.button_grid.addWidget(self.open_cat_train_btn, 0, 0)

        self.open_cat_evaluate_btn = QPushButton('CAT &Evaluate', self)
        self.open_cat_evaluate_btn.clicked.connect(lambda: self.cat_evaluate.show())
        self.open_cat_evaluate_btn.setMinimumHeight(45)
        # self.open_cat_evaluate_btn.setMaximumWidth(150)
        self.button_grid.addWidget(self.open_cat_evaluate_btn, 1, 0)
        # self.main_layout.addStretch()
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

    # splash = QSplashScreen(QPixmap('./icons/busycat.gif'), Qt.WindowStaysOnTopHint)
    splash_pix = QPixmap('./icons/cat-silhouette.jpg')
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setEnabled(False)
    #! FIXME: Progressbar is not animating properly.  Need to put this in a thread
    progressBar = QProgressBar(splash)
    progressBar.setTextVisible(False)
    progressBar.setMaximum(8)
    progressBar.setGeometry(0, splash_pix.height()-20, splash_pix.width(), 20)
    percent_loaded = 0
    splash.show()
    splash.showMessage("<h1>CATScore</h1>", Qt.AlignTop | Qt.AlignCenter, Qt.black)
    app.processEvents()

    # init_loop = QEventLoop()
    # pool = Pool(processes=1)
    # pool.apply_async(initialize_everything, [2], callback=lambda exitCode: init_loop.exit(exitCode))
    # init_loop.exec_()
    import logging
    progressBar.setValue(1)
    import logging.handlers 
    progressBar.setValue(2)
    """Setup logger for logging"""
    handler = logging.handlers.TimedRotatingFileHandler('cat.log', when='W6', interval=1)
    # handler = logging.FileHandler('cat.log')
    # handler = logging.handlers.RotatingFileHandler('cat.log', mode='a', maxBytes=1000)
    logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(logFormatter)
    logging.basicConfig( handlers=[handler], format=logFormatter, level=logging.DEBUG)
    logging.captureWarnings(True)
    progressBar.setValue(3)
    from multiprocessing import Pool
    progressBar.setValue(4)

    from dask.distributed import Client
    progressBar.setValue(5)

    from package.train.cattrain import CatTrain
    progressBar.setValue(6)

    from package.evaluate.catevaluate import CatEvaluate
    progressBar.setValue(7)

    # dask client for paralellization
    # Qt Application
    # splash_pix = QPixmap('./icons/busycat.gif', Qt.WindowStaysOnTopHint)
    client = Client()
    progressBar.setValue(8)

    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())