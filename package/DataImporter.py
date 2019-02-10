from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, 
                               QGridLayout, QHeaderView, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.preprocess_text import processText
from package.DataframeTableModel import DataframeTableModel

class DataImporter(QWidget):
    def __init__(self, parent=None):
        super(DataImporter, self).__init__(parent)
        self.question_checkboxes = []
        self.full_data = pd.DataFrame()
        self.open_file_button = QPushButton('Open CSV', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())

        self.question_checkbox_group = QGroupBox("Select Questions")
        self.preprocess_checkbox_group = QGroupBox("Select Preprocessing Steps")
        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.question_checkbox_grid = QGridLayout()
        self.preprocess_checkbox_grid = QGridLayout()
        self.question_checkbox_group.setLayout(self.question_checkbox_grid)
        self.preprocess_checkbox_group.setLayout(self.preprocess_checkbox_grid)

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.question_checkbox_group)
        self.right_column.addWidget(self.preprocess_checkbox_group)

        self.main_layout.addLayout(self.left_column)
        self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)
        
        self._setupPreprocessCheckboxes()

        self.right_column.addStretch()
        # Initialize preview, first five rows only.
        self.answer_preview_view = QTableView()
        self.answer_preview_model = DataframeTableModel()
        self.answer_preview_view.setModel(self.answer_preview_model)

        self.load_data_btn = QPushButton('Load Data', self)
        self.load_data_btn.clicked.connect(lambda: self.getSelectedData())

        self.left_column.addWidget(self.answer_preview_view)
        self.left_column.addWidget(self.load_data_btn)
        self.left_column.addStretch()
        self.setLayout(self.main_layout)

    def getSelectedData(self):
        """Return dataframe with only selected columns
        # Returns
            A dataframe consisting of those columns selected by the user.
        """
        self.selected_data = self.full_data[self.getSelectedQuestions()]
        self.preprocess_checkbox_group.setEnabled(True)

    def openFile(self):
        """Open file chooser for user to select the CSV file containing their data
        Only CSV files are allowed at this time.
        """
        self.preprocess_checkbox_group.setEnabled(False)
        self.question_checkboxes = []
        file_name, filter = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.loadFile(file_name)
            # if self.question_checkboxes:
                # self.question_checkboxes = []

    def loadFile(self, f_path):
        """Load data from a CSV file to the workspace.
        chardet attempts to determine encoding if file is not utf-8.
            # Attributes
                f_path: String, The filename selected via openFile
        """
        print("Attempting to open {}".format(f_path))
        try:
            self.full_data = pd.read_csv(f_path, encoding='utf-8')
        except UnicodeDecodeError as ude:
            print("UnicodeDecodeError caught.  File is not UTF-8.  Attempting to determine file encoding...")
            detector = UniversalDetector()
            for line in open(f_path, 'rb'):
                detector.feed(line)
                if detector.done: break
            detector.close()
            print("chardet determined encoding type to be {}".format(detector.result['encoding']))
            self.full_data = pd.read_csv(f_path, encoding=detector.result['encoding'])
        except IOError as ioe:
            exceptionWarning("IO Exception occured while opening file.", exception=ioe)
        except Exception as e:
            exceptionWarning("Error occured opening file.", exception=e)

        try:
            columns = self.full_data.columns
            self.question_checkboxes = []
            clearLayout(self.question_checkbox_grid)
            self.answer_preview_model.loadData(self.full_data.head())
            col = -1
            for i, column in enumerate(columns):
                if(i%8 == 0): 
                    col += 1
                    row = 0
                chkbox = QCheckBox(column)
                chkbox.stateChanged.connect(lambda state, x=i: self.changeColumnVisibility(state, x))
                self.question_checkboxes.append(chkbox)
                self.question_checkbox_grid.addWidget(self.question_checkboxes[i], row, col)
                row += 1
                # Hide all columns on load
                self.answer_preview_view.setColumnHidden(i, True)
            
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(exceptionTitle='Empty Data Error occured!\n', exception=ede)
        except Exception as e:
            exceptionWarning("Unexpected error occured!", exception=e)

    def changeColumnVisibility(self, state, col_index):
        self.answer_preview_view.setColumnHidden(col_index, not state)
        # self.answer_preview_view.resizeColumnsToContents()
        self.answer_preview_view.setWordWrap(True)
        # self.answer_preview_view.resizeRowsToContents()

    def getSelectedQuestions(self):
        selected_questions = []
        for question in self.question_checkboxes:
            if question.isChecked():
                selected_questions.append(question.text())
        return selected_questions

    def _setupPreprocessCheckboxes(self):
        preprocess_types = ['Lowercase','Expand Contractions', 'Remove Punctuation', 'Remove Stopwords', 'Lemmatize', 'Stem']
        preprocess_
        self.preprocess_checkboxes = []
        for i, p in enumerate(preprocess_types):
            chkbox = QCheckBox(p)
            self.preprocess_checkboxes.append(chkbox)
            self.preprocess_checkbox_grid.addWidget(self.preprocess_checkboxes[i], i, 0)
        self.preprocess_btn = QPushButton('Apply preprocessing', self)
        self.preprocess_btn.clicked.connect(lambda: self.applyPreprocessing())
        self.preprocess_checkbox_grid.addWidget(self.preprocess_btn)
        self.preprocess_checkbox_group.setEnabled(False)
        
    def applyPreprocessing(self):
        print("Apply preprocessing fired")


        
            

        