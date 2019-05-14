from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Signal, Slot, QThread)
from PySide2.QtGui import QMovie, QIcon, QPixmap
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout,
                               QGridLayout, QHeaderView, QProgressBar, QScrollArea, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import logging
from functools import partial

import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.preprocess_text import processText, get_avg_words_per_sample
from package.utils.spellcheck import SpellCheck
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.AttributeTableModel import AttributeTableModel
from package.utils.GraphWidget import GraphWidget
"""DataLoader imports CSV file and returns a dataframe with the appropriate columns.
For training data, DI will consider the nth column as a training sample
and nth+1 as ground truth.
CSV files must be formatted accordingly.
"""


class DataLoader(QWidget):
    """TODO: Refactor this monstrosity into functions to setup UI
    """
    data_load = Signal(int, bool)
    update_statusbar = Signal(str)
    update_progressbar = Signal(int, bool)

    def __init__(self, parent=None):
        super(DataLoader, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.parent = parent
        self.column_checkboxes = []
        self.selected_column_targets = []
        self.text_preprocessing_checkboxes = []

        self.full_data = pd.DataFrame()
        self.selected_data = pd.DataFrame()
        self.open_file_button = QPushButton('Import CSV', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.full_text_hbox = QHBoxLayout()

        # Column selection and basic stats
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

        # ~ Available question column view
        self.available_column_view = QTableView()
        self.available_column_view.setMinimumHeight(330)
        self.available_column_view.setMaximumWidth(220)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = AttributeTableModel()
        self.available_column_view.setModel(self.available_column_model)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(
            lambda x: self.displaySelectedRow(x))

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.available_column_view)

        self.load_data_btn = QPushButton('Load Data', self)
        self.load_data_btn.clicked.connect(lambda: self.getSelectedData())
        self.select_all_btn = QPushButton('Select All', self)
        self.select_all_btn.clicked.connect(
            lambda: self.available_column_model.setCheckboxes(True))
        self.select_all_btn.setEnabled(False)
        self.deselect_all_btn = QPushButton('Remove All', self)
        self.deselect_all_btn.clicked.connect(
            lambda: self.available_column_model.setCheckboxes(False))
        self.deselect_all_btn.setEnabled(False)

        self.selection_button_layout = QHBoxLayout()
        self.selection_button_layout.addWidget(self.select_all_btn)
        self.selection_button_layout.addWidget(self.deselect_all_btn)

        self.left_column.addLayout(self.selection_button_layout)
        self.left_column.addWidget(self.load_data_btn)
        # self.left_column.addStretch()

        # Text preprocessing options
        self.text_proc_groupbox = QGroupBox("Text Preprocessing Options")
        self.text_proc_groupbox.setEnabled(False)
        self.text_proc_grid = QGridLayout()
        self.text_proc_groupbox.setLayout(self.text_proc_grid)
        self.data_load.connect(self.setProcState)
        self.preprocess_text_btn = QPushButton('Preprocess Text', self)
        self.preprocess_text_btn.clicked.connect(
            lambda: self.applyPreprocessing())
        self.preprocess_text_btn.setEnabled(False)

        self.left_column.addWidget(self.text_proc_groupbox)
        self.left_column.addWidget(self.preprocess_text_btn)
        self.left_column.addStretch()

        # Data subset save button
        self.save_dataset_btn = QPushButton('&Save Dataset', self)
        icon = QIcon()
        icon.addPixmap(QPixmap('icons/Programming-Save-icon.png'))
        self.save_dataset_btn.setIcon(icon)
        self.save_dataset_btn.setEnabled(False)
        self.save_dataset_btn.clicked.connect(lambda: self.saveData())
        self.save_dataset_btn.resize(32, 32)
        self.left_column.addWidget(self.save_dataset_btn)

        self.right_column.addLayout(self.full_text_hbox)
        self.right_column.addWidget(self.text_stats_groupbox)
        self.graph = GraphWidget(self, width=6, height=4, dpi=100)
        self.right_column.addWidget(self.graph)

        # Text DataframeTableModel view for text preview
        self.text_table_view = QTableView()
        self.text_table_view.setSelectionMode(QTableView.SingleSelection)
        self.text_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.text_table_model = DataframeTableModel()
        self.text_table_view.setModel(self.text_table_model)
        self.right_column.addWidget(self.text_table_view)

        self.main_layout.addLayout(self.left_column)
        # self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)

        self.setupTextPreprocessingOptions()

        self.setLayout(self.main_layout)

    def getSelectedData(self):
        """Return columns selected from dataframe by user.
            # Returns
                list: column names selected by user
        """
        self.selected_columns = self.available_column_model.getChecklist()
        if len(self.selected_columns) == 0:
            self.text_proc_groupbox.setEnabled(False)
            self.preprocess_text_btn.setEnabled(False)
            self.selected_data = pd.DataFrame()
            self.text_table_model.loadData(None)
            self.data_load.emit(1, False)
            # exceptionWarning('No questions selected')
        else:
            self.selected_data = self.full_data[self.selected_columns]
            self.text_table_model.loadData(self.selected_data.head())
            self.data_load.emit(1, True)

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
        """
        Load data from a CSV file to the workspace.\n
        Column 0 is used for the index column.\n
        chardet attempts to determine encoding if file is not utf-8.
            # Attributes
                f_path: String, The filename selected via openFile
        """
        # FIXME: Reset status bar when new data is loaded.
        try:
            self.full_data = pd.read_csv(f_path, encoding='utf-8', index_col=0)
        except UnicodeDecodeError as ude:
            self.logger.warning(
                "UnicodeDecode error opening file", exc_info=True)
            print("UnicodeDecodeError caught.  File is not UTF-8 encoded. \
                   Attempting to determine file encoding...")
            self.update_statusbar.emit(
                "UnicodeDecodeError caught.  File is not UTF-8 encoded. Attempting to determine file encoding...")
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
                self.logger.error("Error detecing encoding", exc_info=True)
                exceptionWarning("Exception has occured.", exception=e)
        except IOError as ioe:
            self.logger.error("IOError detecting encoding", exc_info=True)
            exceptionWarning(
                "IO Exception occured while opening file.", exception=ioe)
        except Exception as e:
            self.logger.error("Error detecting encoding", exc_info=True)
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
            self.select_all_btn.setEnabled(True)
            self.deselect_all_btn.setEnabled(True)

            self.update_statusbar.emit("CSV loaded.")
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(
                exceptionTitle='Empty Data Error.\n', exception=ede)
        except Exception as e:
            self.logger.error("Error loading dataframe", exc_info=True)
            exceptionWarning("Unexpected error occured!", exception=e)

    def displaySelectedRow(self, selection=None):
        """
        Updates the stats and label distro plot when a question is selected.
            # Attributes
                selection: QItemSelectionModel, item currently selected by user.
        """
        if selection:
            idx = selection.indexes()[0]
        else:
            # If no question selected, select the first in the list
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

    def saveData(self):
        if self.selected_data.empty:
            exceptionWarning('No data selected')
            return
        file_name, filter = QFileDialog.getSaveFileName(
            self, 'Save to CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.selected_data.to_csv(
                file_name, index_label='testnum', quoting=1, encoding='utf-8')
            self.update_statusbar.emit("Data saved successfully.")

    def setupTextPreprocessingOptions(self):
        """
        Generate necessary UI and backend data structures for text preprocessing option
        selection.
        """
        self.preprocessing_options = {
            "lower_case": True,
            "remove_punctuation": True,
            "expand_contractions": True,
            "remove_stopwords": False,
            "lemmatize": False,
            "spell_correction": False
        }
        proc_labels = [
            'Convert to lowercase', 'Remove punctuation',
            'Expand contractions', 'Remove stopwords',
            'Lemmatize'
        ]
        row = 0
        for k, v in self.preprocessing_options.items():
            chkbox = QCheckBox(k)
            chkbox.setChecked(v)
            chkbox.stateChanged.connect(lambda state, o=k, :
                                        self._updateTextPreprocessingOptions(
                                            o, state)
                                        )
            self.text_proc_grid.addWidget(chkbox, row, 0)
            self.text_preprocessing_checkboxes.append(chkbox)
            row = row + 1

    def _updateTextPreprocessingOptions(self, option, state):
        """
        Updates the selected text preprocessing options.
            # Attributes
                option: String, The text preprocessing option to update
                state: QtCore.Qt.CheckState, The state of the checkbox currently.  This value
                is either 0 or 2, (true, false) so we convert the 2 into a 1 if True.
        """
        truth = 0
        if state == 2:
            truth = 1
        self.preprocessing_options[option] = truth

    @Slot(int, bool)
    def setProcState(self, tab, state):
        """
        Slot for determining if text preprocessing options are selectable.  Based on
        if data has been successfully loaded and selected.  Reusing the Signal that
        enables the Model Selection tab, thus the tab attribute is irrelevant.
            # Attributes:
                tab: int, Tab to enable.  Irrelevant in this case.
                state: bool, The state of the preprocess options.
        """
        if not state:
            self.text_proc_groupbox.setEnabled(False)
            self.preprocess_text_btn.setEnabled(False)
            self.save_dataset_btn.setEnabled(False)
        else:
            self.text_proc_groupbox.setEnabled(True)
            self.preprocess_text_btn.setEnabled(True)
            self.save_dataset_btn.setEnabled(True)

    @Slot(pd.DataFrame)
    def updateData(self, data):
        self.load_data_btn.setEnabled(True)
        self.text_proc_groupbox.setEnabled(True)
        self.preprocess_text_btn.setEnabled(True)
        self.save_dataset_btn.setEnabled(True)

        self.update_statusbar.emit("Text preprocessing complete.")
        self.update_progressbar.emit(0, False)
        self.selected_data = data
        self.text_table_model.loadData(self.selected_data.head())

    def applyPreprocessing(self):
        """
        Spins up a thread to apply user selected preprocessing to input data.
        Data is stored as self.selected_data.
        """
        self.load_data_btn.setEnabled(False)
        self.text_proc_groupbox.setEnabled(False)
        self.preprocess_text_btn.setEnabled(False)
        self.save_dataset_btn.setEnabled(False)
        self.update_progressbar.emit(0, True)
        self.preproc_thread = PreprocessingThread(self.full_data[self.selected_columns],
                                                  self.preprocessing_options)
        self.preproc_thread.preprocessing_complete.connect(self.updateData)
        self.update_statusbar.emit(
            'Preprocessing text.  This may take several minutes.')
        self.preproc_thread.start()


class PreprocessingThread(QThread):
    """
    QThread to handle all text data preprocessing.
    This can be an expensive operation, especially if spell_correction is requested.
    """
    preprocessing_complete = Signal(pd.DataFrame)

    def __init__(self, data, kwargs):
        """
        Make a new thread instance to apply text preprocessing to 
        the selected data.
            # Attributes:
                data: pandas.DataFrame, user selected columns for preprocessing
                kwargs: dict, Text preprocessing options
        """
        super(self.__class__, self).__init__()
        self.data = data
        self.kwargs = kwargs

    def _apply_preprocessing(self, unique_count=5000):
        """
        Apply preprocessing steps to provided data and emit signal when complete.  
        FIXME: Using _Text to denote our text data.  Should this be customizable?
        """
        apply_cols = [
            col for col in self.data.columns if col.endswith('_Text')]
        self.data[apply_cols] = self.data[apply_cols].applymap(
            lambda x: processText(str(x), **self.kwargs))
        if self.kwargs['spell_correction']:
            sentences = self.data[apply_cols].applymap(
                lambda x: str(x).split()).values
            sc = SpellCheck(sentences, 5000)
            self.data[apply_cols] = self.data[apply_cols].applymap(
                lambda x: sc.correct_spelling(x))

        self.preprocessing_complete.emit(self.data)

    def run(self):
        apply_cols = [
            col for col in self.data.columns if col.endswith('_Text')]
        self.data[apply_cols] = self.data[apply_cols].applymap(
            lambda x: processText(str(x), **self.kwargs))
        if self.kwargs['spell_correction']:
            print()
            sentences = self.data[apply_cols].applymap(
                lambda x: str(x).split()).values
            sc = SpellCheck(sentences, unique_count)
            self.data[apply_cols] = self.data[apply_cols].applymap(
                lambda x: sc.correct_spelling(x))

        self.preprocessing_complete.emit(self.data)