from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, 
                               QGridLayout, QHeaderView, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.DataframeTableModel import DataframeTableModel

class DataImporter(QWidget):
    def __init__(self, parent=None):
        super(DataImporter, self).__init__(parent)
        self.question_checkboxes = []
        self.open_file_button = QPushButton('Open', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())

        self.busy_movie = QMovie("../icons/busycat.gif")
        self.movie_screen = QLabel(self)
        self.movie_screen.setScaledContents(True)

        self.movie_screen.setAlignment(Qt.AlignCenter)
        self.busy_movie.setCacheMode(QMovie.CacheAll)
        self.movie_screen.setMovie(self.busy_movie)
        self.busy_movie.start()

        self.checkbox_group = QGroupBox("Select Questions")
        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.checkbox_grid = QGridLayout()
        self.checkbox_group.setLayout(self.checkbox_grid)

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.checkbox_group)
        self.right_column.addWidget(self.movie_screen)
        self.main_layout.addLayout(self.left_column)
        self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)
        # self.main_layout.addWidget(self.movie_screen)
        
        # Initialize preview, first five rows only.
        self.answer_preview_view = QTableView()
        self.answer_preview_model = DataframeTableModel()
        self.answer_preview_view.setModel(self.answer_preview_model)
        self.left_column.addWidget(self.answer_preview_view)
        self.left_column.addStretch()
        self.setLayout(self.main_layout)

    def openFile(self):
        """Open file chooser for user to select the CSV file containing their data
        Only CSV files are allowed at this time.
        """
        self.question_checkboxes = []
        file_name, filter = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.loadFile(file_name)
            # if self.question_checkboxes:
                # self.question_checkboxes = []

    def loadFile(self, f_name):
        """Load data from a CSV file to the workspace.
        chardet attempts to determine encoding prior to opening the file.
            # Attributes
                f_name: String, The filename selected via openFile
        """
        print("Attempting to open {}".format(f_name))
        detector = UniversalDetector()
        try:
            for line in open(f_name, 'rb'):
                detector.feed(line)
                if detector.done: break
            detector.close()
        except IOException as ioe:
            exceptionWarning("IO Exception occured while opening file.", exception=ioe)
        except Exception as e:
            exceptionWarning("Unexpected error occured!", exception=e)

        print("chardet determined encoding type to be {}".format(detector.result['encoding']))
        try:
            full_data = pd.read_csv(f_name, encoding=detector.result['encoding'])
            columns = full_data.columns
            self.question_checkboxes = []
            clearLayout(self.checkbox_grid)
            self.answer_preview_model.loadData(full_data.head())
            col = -1
            for i, column in enumerate(columns):
                if(i%8 == 0): 
                    col += 1
                    row = 0
                chkbox = QCheckBox(column)
                chkbox.stateChanged.connect(lambda state, x=i: self.changeColumnVisibility(state, x))
                self.question_checkboxes.append(chkbox)
                print("column number: {}".format(col))
                self.checkbox_grid.addWidget(self.question_checkboxes[i], row, col)
                row += 1
                # Hide all columns on load
                self.answer_preview_view.setColumnHidden(i, True)
            # Stretch pushes checkboxes to top of layout
            # self.checkbox_grid.addStretch()
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(exceptionTitle='Empty Data Error occured!\n', exception=ede)
        except Exception as e:
            exceptionWarning("Unexpected error occured!", exception=e)

    def changeColumnVisibility(self, state, col_index):
        self.answer_preview_view.setColumnHidden(col_index, not state)



        
            

        