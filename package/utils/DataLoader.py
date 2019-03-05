from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Signal)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout,
                               QGridLayout, QHeaderView, QScrollArea, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.preprocess_text import processText, get_avg_words_per_sample
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.AttributeTableModel import AttributeTableModel
from package.utils.GraphWidget import GraphWidget
"""DataLoader imports CSV file and returns a dataframe with the appropriate columns.
For training data, DI will consider the nth column as training data and nth+1 as ground truth.
CSV files must be formatted accordingly.
"""

class DataLoader(QWidget):
    """TODO: Refactor this monstrosity into functions to setup UI
    """
    data_load = Signal(pd.DataFrame)
    def __init__(self, parent=None):
        super(DataLoader, self).__init__(parent)
        self.parent = parent
        self.column_checkboxes = []
        self.selected_column_targets = []

        self.full_data = pd.DataFrame()
        self.open_file_button = QPushButton('Import CSV', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.full_text_hbox = QHBoxLayout()

        # Text details
        self.text_stats_groupbox = QGroupBox("Selected Question")
        self.text_stats_grid = QGridLayout()

        self.full_text_count_label = QLabel("Total samples")
        self.full_text_count = QLabel()

        self.current_question_count_label = QLabel("No. samples")
        self.current_question_count = QLabel()
        self.current_question_avg_word_label = QLabel("Avg. words per sample")
        self.current_question_avg_word = QLabel()

        self.full_text_hbox.addWidget(self.full_text_count_label)
        self.full_text_hbox.addWidget(self.full_text_count)
        self.text_stats_grid.addWidget(self.current_question_count_label, 1, 0)
        self.text_stats_grid.addWidget(self.current_question_count, 1, 1)
        self.text_stats_grid.addWidget(
            self.current_question_avg_word_label, 2, 0)
        self.text_stats_grid.addWidget(self.current_question_avg_word, 2, 1)
        self.full_text_count.setText("None")
        self.text_stats_groupbox.setLayout(self.text_stats_grid)

        #~ Available question column view
        self.available_column_view = QTableView()
        self.available_column_view.setMinimumHeight(330)
        self.available_column_view.setMaximumWidth(220)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = AttributeTableModel()
        self.available_column_view.setModel(self.available_column_model)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(lambda x: self.displaySelectedRow(x))

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.available_column_view)

        self.load_data_btn = QPushButton('Load Data', self)
        self.load_data_btn.clicked.connect(lambda: self.getSelectedData())

        self.left_column.addWidget(self.load_data_btn)
        self.left_column.addStretch()
        self.right_column.addLayout(self.full_text_hbox)
        self.right_column.addWidget(self.text_stats_groupbox)
        self.graph = GraphWidget(self, width=5, height=4, dpi=100)
        self.right_column.addWidget(self.graph)
        self.main_layout.addLayout(self.left_column)
        self.main_layout.addLayout(self.right_column)
        self.setLayout(self.main_layout)

    def getSelectedData(self):
        """Return columns selected from dataframe by user
        # Returns
            list; column names selected by user
        """
        selected_columns = self.available_column_model.getChecklist()
        if len(selected_columns) == 0:
            exceptionWarning('No questions selected')
            return
        data = self.full_data[selected_columns]
        self.data_load.emit(data)
        #TODO: delete full data?
        
    def openFile(self):
        """Open file chooser for user to select the CSV file containing their data
        Only CSV files are allowed at this time.
        """
        self.column_checkboxes = []
        self.selected_column_targets = []
        file_name, filter = QFileDialog.getOpenFileName(
            self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.loadFile(file_name)

    def loadFile(self, f_path):
        """Load data from a CSV file to the workspace.
        chardet attempts to determine encoding if file is not utf-8.
            # Attributes
                f_path: String, The filename selected via openFile
        """
        #FIXME: Reset status bar when new data is loaded.
        # self.parent.parent.statusBar.showMessage("Attempting to open {}".format(f_path))
        try:
            self.full_data = pd.read_csv(f_path, encoding='utf-8', index_col=0)

        except UnicodeDecodeError as ude:
            print("UnicodeDecodeError caught.  File is not UTF-8 encoded. \
                   Attempting to determine file encoding...")
            detector = UniversalDetector()
            try:
                for line in open(f_path, 'rb'):
                    detector.feed(line)
                    if detector.done:
                        break
                detector.close()
                print("chardet determined encoding type to be {}".format(
                    detector.result['encoding']))
                self.full_data = pd.read_csv(
                    f_path, encoding=detector.result['encoding'], index_col=0)
            except Exception as e:
                exceptionWarning("Exception has occured.", exception=e)
        except IOError as ioe:
            exceptionWarning(
                "IO Exception occured while opening file.", exception=ioe)
        except Exception as e:
            exceptionWarning("Error occured opening file.", exception=e)

        try:
            columns = self.full_data.columns
            self.available_columns = []
            for column in columns:
                if column.endswith("Text"):
                    self.available_columns.append(column)
                    self.available_columns.append(
                        columns[columns.get_loc(column) + 1])
            self.available_column_model.loadData(self.available_columns)
            self.full_text_count.setText(str(self.full_data.shape[0]))
            self.displaySelectedRow(None)
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(
                exceptionTitle='Empty Data Error.\n', exception=ede)
        except Exception as e:
            exceptionWarning("Unexpected error occured!", exception=e)

    def displaySelectedRow(self, selection=None):
        """Updates the stats and label distro plot when a question is selected.
            # Attributes
                selection: QItemSelectionModel, item currently selected by user.
        """
        if selection:
            idx = selection.indexes()[0]
        else:
            self.available_column_view.selectRow(0)
            self.available_column_view.setFocus()
            idx = QModelIndex(self.available_column_model.index(0, 0))
        offset = idx.row() * 2
        col_name = self.full_data.columns[offset]
        self.text_stats_groupbox.setTitle(col_name)
        question_data = self.full_data[self.full_data.columns[offset]].dropna(
            how='any')
        avg_num_words = get_avg_words_per_sample(question_data.values)
        self.current_question_count.setText(str(question_data.shape[0]))
        self.current_question_avg_word.setText("%.2f" % avg_num_words)

        self.graph.chartSingleClassFrequency(
            self.full_data[self.full_data.columns[offset + 1]].values)
