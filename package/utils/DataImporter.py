from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, 
                               QGridLayout, QHeaderView, QScrollArea, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.preprocess_text import processText
from package.utils.DataframeTableModel import DataframeTableModel
"""DataImporter imports CSV file and returns a dataframe with the appropriate columns.
For training data, DI will carry along the n column as training data and n+1 as ground truth.
CSV files must be formatted accordingly.
"""
class DataImporter(QWidget):
    def __init__(self, parent=None):
        super(DataImporter, self).__init__(parent)
        self.column_checkboxes = []
        self.selected_column_targets = []

        self.full_data = pd.DataFrame()
        self.open_file_button = QPushButton('Import CSV', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())

        self.column_select_groupbox = QGroupBox("Select Questions")
        # self.preprocess_checkbox_group = QGroupBox("Select Preprocessing Steps")
        # self.main_layout = QHBoxLayout()
        self.main_layout = QVBoxLayout()
        # self.right_column = QVBoxLayout()
        self.column_select_grid = QGridLayout()

        self.column_select_scrollarea = QScrollArea()
        self.column_select_scrollarea.setWidgetResizable(True)
        self.column_select_scrollarea.setMinimumHeight(100)
        # self.column_select_scrollarea.setMaximumHeight(300)
        self.column_select_scrollarea.setMaximumWidth(200)

        self.column_select_groupbox.setLayout(self.column_select_grid)
        self.column_select_scrollarea.setWidget(self.column_select_groupbox)
        # self.preprocess_checkbox_group.setLayout(self.preprocess_checkbox_grid)

        self.main_layout.addWidget(self.open_file_button)
        self.main_layout.addWidget(self.column_select_scrollarea)
        # self.right_column.addWidget(self.preprocess_checkbox_group)

        # self.main_layout.addLayout(self.main_layout)
        # self.main_layout.addStretch()
        # self.main_layout.addLayout(self.right_column)
        
        # self._setupPreprocessCheckboxes()

        # self.right_column.addStretch()
        # Initialize preview, first five rows only.
        # self.answer_preview_view = QTableView()
        # self.answer_preview_model = DataframeTableModel()
        # self.answer_preview_view.setModel(self.answer_preview_model)

        # self.load_data_btn = QPushButton('Load Data', self)
        # self.load_data_btn.clicked.connect(lambda: self.getSelectedData())

        # self.main_layout.addWidget(self.answer_preview_view)
        # self.main_layout.addWidget(self.load_data_btn)
        # self.main_layout.addStretch()
        self.setLayout(self.main_layout)

    def getSelectedData(self):
        """Return columns selected from dataframe by user
        Enables the selection of preprocessing options.  
        # Returns
            list; column names selected by user
        """
        self.selected_columns = self.getSelectedQuestions()
        self.preprocess_checkbox_group.setEnabled(True)

    def openFile(self):
        """Open file chooser for user to select the CSV file containing their data
        Only CSV files are allowed at this time.
        """
        # self.preprocess_checkbox_group.setEnabled(False)
        self.column_checkboxes = []
        self.selected_column_targets = []
        file_name, filter = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.loadFile(file_name)

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
            # Fill empty values with zero.  Will be a string for questions?
            na_dict = dict.fromkeys(columns, 0)
            self.full_data.fillna(na_dict, inplace=True)
            self.column_checkboxes = []
            clearLayout(self.column_select_grid)
            row = 0
            for column in columns:
                if column.endswith("Text"):
                    chkbox = QCheckBox(column, self)
                    # chkbox.stateChanged.connect(lambda state, x=i: self.changeColumnVisibility(state, x))
                    self.column_checkboxes.append(chkbox)
                    self.selected_column_targets.append(self.full_data.columns[self.full_data.columns.get_loc(column) + 1])
                    self.column_select_grid.addWidget(self.column_checkboxes[row], row, 0)
                    self.column_select_grid.addWidget(QLabel(self._getTargetLabel(self.full_data[self.selected_column_targets[row]])), row, 1)
                    row += 1
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(exceptionTitle='Empty Data Error.\n', exception=ede)
        except Exception as e:
            exceptionWarning("Unexpected error occured!", exception=e)

    def _getTargetLabel(self, column):
        return column.name + ' [' + str(int(column.min(skipna=True))) + ' ' + str(int(column.max())) + ']'

    def changeColumnVisibility(self, state, col_index):
        pass
        # self.answer_preview_view.setColumnHidden(col_index, not state)
        # self.answer_preview_view.resizeColumnsToContents()
        # self.answer_preview_view.setWordWrap(True)
        # self.answer_preview_view.resizeRowsToContents()

    def getSelectedQuestions(self):
        selected_questions = []
        for question in self.column_checkboxes:
            if question.isChecked():
                selected_questions.append(question.text())
        return selected_questions

    def _setupPreprocessCheckboxes(self):
        preprocess_types = ['Lowercase','Expand Contractions', 'Remove Punctuation', 'Remove Stopwords', 'Lemmatize', 'Stem']
        self.preprocess_checkboxes = []
        for i, p in enumerate(preprocess_types):
            chkbox = QCheckBox(p)
            self.preprocess_checkboxes.append(chkbox)
            self.preprocess_checkbox_grid.addWidget(self.preprocess_checkboxes[i], i, 0)

        self.preprocess_btn = QPushButton('Apply preprocessing', self)
        self.preprocess_btn.clicked.connect(lambda: self.applyPreprocessing())
        self.preprocess_checkbox_grid.addWidget(self.preprocess_btn)
        self.preprocess_checkbox_group.setEnabled(False)
        
    def _addPreprocessItem(self, state, preprocess_type):
        kwargs = {

        }

    def applyPreprocessing(self):
        self.full_data[self.selected_columns] = self.full_data[self.selected_columns].applymap(lambda x: processText(str(x)))
        print(self.full_data[self.selected_columns].head())
        print("Shape of full_data: ", self.full_data.shape)
        


        
            

        