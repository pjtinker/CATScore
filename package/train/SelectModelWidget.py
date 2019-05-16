from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot, Signal, SIGNAL, QObject)
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, 
                                QTabWidget, QComboBox, QFormLayout,
                                QApplication, QLabel, QFileDialog, QHBoxLayout, 
                                QVBoxLayout, QGridLayout, QHeaderView, QScrollArea, 
                                QSizePolicy, QTableView, QWidget, QPushButton)
import os
import json
from collections import OrderedDict
import pkg_resources
import traceback 
from functools import partial
import logging 

import pandas as pd
# from addict import Dict

from package.train.models.SkModelDialog import SkModelDialog
from package.train.models.TfModelDialog import TfModelDialog
from package.utils.catutils import exceptionWarning

BASE_SK_MODEL_DIR = "./package/data/base_models"
BASE_TF_MODEL_DIR = "./package/data/tensorflow_models"
BASE_TFIDF_DIR = "./package/data/feature_extractors/TfidfVectorizer.json"
BASE_FS_DIR = "./package/data/feature_selection/FeatureSelection.json"

class Communicate(QObject):
    version_change = Signal(str)

class SelectModelWidget(QTabWidget):
    """QTabWidget that holds all of the selectable models and the accompanying ModelDialog for each.
    """
    update_statusbar = Signal(str)
    update_progressbar = Signal(int, bool)
    def __init__(self, parent=None):
        super(SelectModelWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.selected_version = 'experimental'
        self.comms = Communicate()

        self.selected_models = {}
        self.model_checkboxes = []
        
        self.sklearn_model_dialogs = []
        self.sklearn_model_dialog_btns = []

        self.tf_model_dialogs = []
        self.tf_model_dialog_btns = []


        self.main_layout = QVBoxLayout()
        self.form_grid = QGridLayout()
        # self.left_column = QVBoxLayout()
        # self.right_column = QVBoxLayout()
        # self.version_hbox = QHBoxLayout()

        self.skmodel_groupbox = QGroupBox("Sklearn Models")
        self.tf_model_groupbox = QGroupBox("Tensorflow Models")
        self.fs_groupbox = QGroupBox("Feature Selection")
        self.tf_model_grid = QGridLayout()
        self.sklearn_model_grid = QGridLayout()
        self.feature_selection_param_form = QFormLayout()
        self.skmodel_groupbox.setLayout(self.sklearn_model_grid)
        self.tf_model_groupbox.setLayout(self.tf_model_grid)
        self.fs_groupbox.setLayout(self.feature_selection_param_form)

        self.setupUi()
        # self._build_fs_ui()
        self.form_grid.addWidget(self.skmodel_groupbox, 1, 0)
        self.form_grid.addWidget(self.tf_model_groupbox, 1, 1)
        self.form_grid.addWidget(self.fs_groupbox, 1, 2)
        # self.left_column.addLayout(self.version_hbox)
        # self.version_hbox.addStretch()
        # self.left_column.addWidget(self.skmodel_groupbox)
        # self.left_column.addWidget(self.tf_model_groupbox)
        # self.left_column.addStretch()
        # self.right_column.addStretch()
        # self.main_layout.addLayout(self.left_column)
        # self.main_layout.addStretch()
        # self.main_layout.addLayout(self.right_column)
        # self.main_layout.addStretch()
        self.main_layout.addLayout(self.form_grid)
        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

    def setupUi(self):
        self.version_selection = QComboBox(objectName='version_select')
        available_versions = os.listdir(".\\package\\data\\versions")
        for version in available_versions:
            self.version_selection.addItem(version, os.path.join('.\\package\\data\\versions', version))
        self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection: 
                                                            self._signal_update(y.currentData())
                                                            )
        
        self.form_grid.addWidget(self.version_selection, 0, 0)
        # Load base TF-IDF and feature selection data
        try:
            with open(BASE_TFIDF_DIR, 'r') as f:
                tfidf_data = json.load(f)
        except IOError as ioe:
            self.logger.error("Error loading base TFIDF params", exc_info=True)
            exceptionWarning('Error occurred while loading base TFIDF parameters.', repr(ioe))
        try:
            with open(BASE_FS_DIR, 'r') as f:
                self.fs_params = json.load(f)
        except IOError as ioe:
            self.logger.error("Error loading base feature selector params", exc_info=True)
            exceptionWarning('Error occurred while loading base feature selector parameters.', repr(ioe))
        
        try:
            row = 0
            for filename in os.listdir(BASE_SK_MODEL_DIR):
                with open(os.path.join(BASE_SK_MODEL_DIR, filename), 'r') as f:
                    print("Loading model:", filename)
                    model_data = json.load(f)
                    model_dialog = SkModelDialog(self, model_data, tfidf_data)
                    self.comms.version_change.connect(model_dialog.update_version)
                    model = model_data['model_class']
                    # Initialize model as unselected
                    self.selected_models[model] = False
                    btn = QPushButton(model)
                    # Partial allows the connection of dynamically generated QObjects
                    btn.clicked.connect(partial(self.openDialog, model_dialog))
                    chkbox = QCheckBox()
                    chkbox.stateChanged.connect(lambda state, x=model :
                                            self._update_selected_models(x, state))
                    self.sklearn_model_grid.addWidget(chkbox, row, 0)
                    self.sklearn_model_grid.addWidget(btn, row, 1)
                    self.sklearn_model_dialogs.append(model_dialog)
                    self.sklearn_model_dialog_btns.append(btn)
                    self.model_checkboxes.append(chkbox)
                    row += 1
        except OSError as ose:
            self.logger.error("OSError opening Scikit model config files", exc_info=True)
            exceptionWarning('OSError opening Scikit model config files!', ose)
        except Exception as e:
            self.logger.error("Error opening Scikit model config files", exc_info=True)
            exceptionWarning('Error occured.', e)
            tb = traceback.format_exc()
            print(tb)

        try:
            row = 0
            for filename in os.listdir(BASE_TF_MODEL_DIR):
                with open(os.path.join(BASE_TF_MODEL_DIR, filename), 'r') as f:
                    print("Loading model:", filename)
                    model_data = json.load(f)
                    model_dialog = TfModelDialog(self, model_data)
                    self.comms.version_change.connect(model_dialog.update_version)
                    model = model_data['model_class']
                    # Intialize model as unselected
                    self.selected_models[model] = False
                    btn = QPushButton(model)
                    btn.clicked.connect(partial(self.openDialog, model_dialog))
                    chkbox = QCheckBox()
                    chkbox.stateChanged.connect(lambda state, x=model :
                                            self._update_selected_models(x, state))
                    self.tf_model_grid.addWidget(chkbox, row, 0)
                    self.tf_model_grid.addWidget(btn, row, 1)
                    self.tf_model_dialogs.append(model_dialog)
                    self.tf_model_dialog_btns.append(btn)
                    self.model_checkboxes.append(chkbox)
                    row += 1
        except OSError as ose:
            self.logger.error("OSError opening Tensorflow model config files", exc_info=True)
            exceptionWarning('Error opening Tensorflow model config files!', ose)
        except Exception as e:
            self.logger.error("Error opening Scikit model config files", exc_info=True)
            exceptionWarning('Error occured.', e)
            tb = traceback.format_exc()
            print(tb)
        

        # Trigger update to load model parameters
        self._signal_update(self.version_selection.currentData())
            
    def openDialog(self, dialog):
        dialog.saveParams()

    def _signal_update(self, directory):
        print("Emitting {} from {}".format(directory, self.__class__.__name__))
        self.selected_version = directory.split('\\')[-1]
        print("selected version: ", self.selected_version)
        self.comms.version_change.emit(directory)

    def _update_selected_models(self, model, state):
        truth = 0
        if state == 2:
            truth = 1
        print("Updated selected model {} to state {}".format(model, truth))
        self.selected_models[model] = truth

    def _build_fs_ui(self):
        """Build UI elements for feature selection options.

            # Attributes:
                Each option
                has nested dictionaries for param values and score options.  i.e.
                SelectKBest 
                    params:
                        k : 10,
                    score_options: [
                        'f_classif',
                        'chi2',
                        'SelectFwe'
                    ]
        """
        fs_combo = QComboBox(objectName='feature_selection')

        self.feature_selection_param_form.addRow('Feature selection type', fs_combo)
        for fs, options in self.fs_params['types'].items():
            fs_combo.addItem(fs, fs)
        fs_combo.currentIndexChanged.connect(
            lambda state, x=fs_combo: self._update_fs_ui(x.currentData())
        )
        self._update_fs_ui(fs_combo.currentData())

    
    def _update_fs_ui(self, fs_option):
        """Generates and updates UI based upon FS selection type.
        Function first destroys any mention of previous FS objects from
        both the QFormLayout as well as self.input_widgets.

            # Attributes:
                fs_option: String, feature selection option chosen by user.
 
        FIXME: Currently, FS is always enabled.  Should we have a checkbox?
        """
        current_fs_widgets = dict(self.fs_params['types']) 
        current_fs_widgets.pop(fs_option, None)

        self._update_param('feature_selection', 'type', fs_option)
        for fs_type, fs_params in current_fs_widgets.items():
            scorer = self.input_widgets.pop(fs_type, None)
            self.feature_selection_param_form.removeRow(scorer)
            self.input_widgets.pop(fs_type, None)
            for param, val in fs_params['params'].items():
                widget = self.input_widgets.pop(param, None)
                self.feature_selection_param_form.removeRow(widget)
        for param, val in self.fs_params['types'][fs_option]['params'].items():
            label_string = param
            label = QLabel(' '.join(label_string.split('_')))
            if isinstance(val, int):
                param_field = QSpinBox(objectName=param)
                param_field.setRange(0, 10000)
                param_field.setValue(val)
                param_field.valueChanged.connect(
                    lambda state, x=param, y=param_field: self._update_param(
                        'feature_selection',
                        x, 
                        y.value())
                )
            elif isinstance(val, float):
                param_field = QDoubleSpinBox(objectName=param)
                param_field.setDecimals(len(str(val)) - 2)
                param_field.setValue(val)
                param_field.valueChanged.connect(
                    lambda state, x=param, y=param_field: self._update_param(
                        'feature_selection',
                        x, 
                        y.value())
                )
            self.feature_selection_param_form.addRow(label, param_field)
            self.input_widgets[param] = param_field
        score_combo = QComboBox(objectName=fs_option)
        for score_option in self.fs_params['types'][fs_option]['score_options']:
            score_combo.addItem(score_option, score_option)
        score_combo.currentIndexChanged.connect(
            lambda state, x=score_combo: self._update_param(
                'feature_selection',
                'scorer',
                score_combo.currentData()
            )
        )
        self.feature_selection_param_form.addRow('Scorers', score_combo)
        self.input_widgets[fs_option] = score_combo




    