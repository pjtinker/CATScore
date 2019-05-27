from PySide2.QtCore import (Qt, Slot, Signal, QObject)
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, 
                                QTabWidget, QComboBox, QFormLayout,
                                QLabel, QFileDialog, QHBoxLayout, 
                                QVBoxLayout, QGridLayout, QScrollArea, 
                                QSizePolicy, QPushButton, QButtonGroup,
                                QRadioButton, QSpinBox, QDoubleSpinBox)
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
BASE_FS_DIR = "./package/data/feature_selection/SelectKBest.json"

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

        self.training_data = None

        self.selected_version = 'experimental'
        self.comms = Communicate()

        self.selected_models = {}
        self.model_checkboxes = []

        self.training_params = {}
        self.training_params['sklearn'] = {}
        self.training_params['tensorflow'] = {}
        
        self.sklearn_model_dialogs = []
        self.sklearn_model_dialog_btns = []
        self.sklearn_training_inputs = []


        self.tensorflow_training_inputs = []
        self.tensorflow_model_dialogs = []
        self.tensorflow_model_dialog_btns = []

        self.main_layout = QVBoxLayout()
        self.version_form = QFormLayout()
        self.header_hbox = QHBoxLayout()
        self.header_hbox.addLayout(self.version_form)
        self.header_hbox.addStretch()
        self.tune_models_chkbox = QCheckBox("Tune Models")
        self.header_hbox.addWidget(self.tune_models_chkbox)
        self.tune_models_chkbox.stateChanged.connect(lambda state:
                                            self._enable_tuning_ui(state)
                                            )
        self.main_layout.addLayout(self.header_hbox)
        # Build sklearn ui components
        self.sklearn_hbox = QHBoxLayout()
        self.sklearn_groupbox = QGroupBox("Sklearn")
        self.sklearn_groupbox.setLayout(self.sklearn_hbox)
        
        self.skmodel_groupbox = QGroupBox("Model Selection")
        self.sklearn_hbox.addWidget(self.skmodel_groupbox)
        self.sklearn_model_form = QFormLayout()
        # self.sklearn_model_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignCenter)
        self.sklearn_model_form.setFormAlignment(Qt.AlignTop)
        self.skmodel_groupbox.setLayout(self.sklearn_model_form)

        # Sklearn training and tuning ui components
        self.sklearn_training_groupbox = QGroupBox("Training")
        self.sklearn_training_form = QFormLayout()
        self.sklearn_training_groupbox.setLayout(self.sklearn_training_form)
        self.sklearn_hbox.addWidget(self.sklearn_training_groupbox)
        # self.sklearn_training_radio_btngroup = QButtonGroup()

        self.sklearn_tuning_groupbox = QGroupBox("Tuning")
        self.sklearn_hbox.addWidget(self.sklearn_tuning_groupbox)

        self.main_layout.addWidget(self.sklearn_groupbox)

        # Build Tensorflow ui components
        self.tensorflow_hbox = QHBoxLayout()
        self.tensorflow_groupbox = QGroupBox("Tensorflow")
        self.tensorflow_groupbox.setLayout(self.tensorflow_hbox)
        
        self.tensorflow_model_groupbox = QGroupBox("Model Selection")
        self.tensorflow_hbox.addWidget(self.tensorflow_model_groupbox)
        self.tensorflow_model_form = QFormLayout()
        # self.tensorflow_model_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignCenter)
        # self.tensorflow_model_form.setFormAlignment(Qt.AlignCenter)
        self.tensorflow_model_groupbox.setLayout(self.tensorflow_model_form)
        self.tensorflow_training_groupbox = QGroupBox("Training")
        self.tensorflow_training_form = QFormLayout()
        self.tensorflow_training_groupbox.setLayout(self.tensorflow_training_form)
        self.tensorflow_hbox.addWidget(self.tensorflow_training_groupbox)
        self.tensorflow_tuning_groupbox = QGroupBox("Tuning")
        self.tensorflow_hbox.addWidget(self.tensorflow_tuning_groupbox)

        self.main_layout.addWidget(self.tensorflow_groupbox)
        self.model_form_grid = QGridLayout()

        self.setup_model_selection_ui()
        self.setup_training_ui()

        self.main_layout.addStretch()
        self.run_btn = QPushButton("Run!")
        self.run_btn.setEnabled(False)

        self.main_layout.addWidget(self.run_btn)
        self.setLayout(self.main_layout)

    def setup_model_selection_ui(self):
        self.version_selection_label = QLabel("Select version: ")
        self.version_selection = QComboBox(objectName='version_select')
        self.version_selection.addItem('default', '.\\package\\data\\versions\\default')
        available_versions = os.listdir(".\\package\\data\\versions")
        for version in available_versions:
            v_path = os.path.join('.\\package\\data\\versions', version)
            if os.path.isdir(v_path):
                self.version_selection.addItem(version, v_path)
        self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection: 
                                                            self._update_version(y.currentData())
                                                            )
        # idx = self.version_selection.findData(".\\package\\data\\versions\\default")
        # self.version_selection.setCurrentIndex(idx)
        self.version_form.addRow(self.version_selection_label, self.version_selection)
        
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
                    model_dialog = SkModelDialog(self, model_data, tfidf_data, self.fs_params)
                    self.comms.version_change.connect(model_dialog.update_version)
                    model = model_data['model_class']
                    # Initialize model as unselected
                    self.selected_models[model] = False
                    btn = QPushButton(model)
                    # Partial allows the connection of dynamically generated QObjects
                    btn.clicked.connect(partial(self.open_dialog, model_dialog))
                    chkbox = QCheckBox()
                    chkbox.stateChanged.connect(lambda state, x=model :
                                            self._update_selected_models(x, state))
                    if model_data['model_base'] == 'tensorflow':
                        self.tensorflow_model_form.addRow(chkbox, btn)
                        self.tensorflow_model_dialogs.append(model_dialog)
                        self.tensorflow_model_dialog_btns.append(btn)
                    else:
                        self.sklearn_model_form.addRow(chkbox, btn)
                        self.sklearn_model_dialogs.append(model_dialog)
                        self.sklearn_model_dialog_btns.append(btn)
                    self.model_checkboxes.append(chkbox)
                    row += 1
        except OSError as ose:
            self.logger.error("OSError opening model config files", exc_info=True)
            exceptionWarning('OSError opening model config files!', ose)
        except Exception as e:
            self.logger.error("Error opening model config files", exc_info=True)
            exceptionWarning('Error occured.', e)
            tb = traceback.format_exc()
            print(tb)
        
        # Trigger update to load model parameters
        self._update_version(self.version_selection.currentData())
            

    def setup_training_ui(self):
        """
        Build ui components for training parameters for both Sklearn and Tensorflow
        """
        # Sklearn training first
        cv_n_fold_input = QSpinBox(objectName='n_folds')
        cv_n_fold_input.setRange(2, 10)
        cv_n_fold_input.setValue(5)
        # cv_n_fold_input.setEnabled(False)
        cv_n_fold_input.valueChanged.connect(
            lambda state, x=cv_n_fold_input:
                self._update_training_params('sklearn', 'cv', x.value())
            )
        cv_radio_btn = QRadioButton("Cross-validation", objectName='cv')
        cv_radio_btn.toggled.connect(lambda state, x=cv_n_fold_input: 
                                        self._update_training_params('sklearn', 'cv', x.value())
                                    )
        cv_radio_btn.toggle()
        self.sklearn_training_inputs.append(cv_radio_btn)
        self.sklearn_training_inputs.append(cv_n_fold_input)
        self.sklearn_training_form.addRow(cv_radio_btn, cv_n_fold_input)
        
        sk_validation_radio_btn = QRadioButton("Validation set")
        sk_validation_percent_input = QDoubleSpinBox(objectName='test_split')
        sk_validation_percent_input.setRange(0.05, 1)
        # sk_validation_percent_input.setValue(0.2)
        sk_validation_percent_input.setSingleStep(0.1)
        # sk_validation_percent_input.setEnabled(False)
        sk_validation_percent_input.valueChanged.connect(
            lambda state, x=sk_validation_percent_input:
                self._update_training_params('sklearn', 'validation', x.value())
            )

        sk_validation_radio_btn.toggled.connect(
            lambda state, x=sk_validation_percent_input:
                self._update_training_params('sklearn', 'validation', x.value())
            )
        self.sklearn_training_inputs.append(sk_validation_radio_btn)
        self.sklearn_training_inputs.append(sk_validation_percent_input)
        self.sklearn_training_form.addRow(sk_validation_radio_btn, sk_validation_percent_input)
    
        no_eval_btn = QRadioButton("No evaluation set", objectName='no_eval')
        no_eval_btn.toggled.connect(lambda: self._update_training_params('sklearn', 'no_eval', None))
        self.sklearn_training_inputs.append(no_eval_btn)
        self.sklearn_training_form.addRow(no_eval_btn)
        # TODO: Add _update_training_params to each field for tensorflow.
        tf_val_label = QLabel("Validation split")
        tf_val_input = QDoubleSpinBox(objectName='validation_split')
        tf_val_input.setRange(0.05, 1)
        tf_val_input.setValue(0.2)
        tf_val_input.setSingleStep(0.1)
        self.tensorflow_training_inputs.append([tf_val_input, tf_val_label])
        self.tensorflow_training_form.addRow(tf_val_label, tf_val_input)

        tf_patience_label = QLabel("Patience")
        tf_patience_input = QSpinBox(objectName='patience')
        tf_patience_input.setRange(0, 1000)
        tf_patience_input.setValue(2)
        tf_patience_input.setSingleStep(1)
        self.tensorflow_training_inputs.append([tf_patience_label, tf_patience_input])
        self.tensorflow_training_form.addRow(tf_patience_label, tf_patience_input)

        tf_embedding_type_label = QLabel("Embedding type")
        tf_embedding_combobox = QComboBox(objectName='embedding_type')
        tf_embedding_combobox.addItem('GloVe', 'glove')
        tf_embedding_combobox.addItem('Word2Vec', 'word2vec')
        tf_embedding_combobox.setCurrentIndex(0)
        self.tensorflow_training_inputs.append([tf_embedding_type_label, tf_embedding_combobox])
        self.tensorflow_training_form.addRow(tf_embedding_type_label, tf_embedding_combobox)

        tf_embedding_dims_label = QLabel("Embedding dims")
        tf_embedding_dims_input = QSpinBox(objectName='embedding_dims')
        tf_embedding_dims_input.setRange(100, 300)
        tf_embedding_dims_input.setValue(100)
        tf_embedding_dims_input.setSingleStep(100)
        self.tensorflow_training_inputs.append([tf_embedding_dims_label, tf_embedding_dims_input])
        self.tensorflow_training_form.addRow(tf_embedding_dims_label, tf_embedding_dims_input)

        tf_embedding_trainable_label = QLabel("Train embeddings")
        tf_embedding_trainable_chkbox = QCheckBox(objectName='embedding_trainable')
        self.tensorflow_training_inputs.append([tf_embedding_trainable_label, tf_embedding_trainable_chkbox])
        self.tensorflow_training_form.addRow(tf_embedding_trainable_label, tf_embedding_trainable_chkbox)


    def open_dialog(self, dialog):
        dialog.save_params()


    @Slot(str)
    def add_new_version(self, v_dir):
        """
        Slot to receive new version created Signal.

            # Arguments
                v_dir(String): directory of newly created version.
        """
        version = v_dir.split('\\')[-1]
        self.version_selection.addItem(version, v_dir)

    @Slot(pd.DataFrame)
    def load_data(self, data):
        """
        Slot to receive pandas DataFrame after DataLoader has completed it's work

            # Arguments
                data(pandas.DataFrame): DataFrame of training data
        """
        if(data.empty):
            self.training_data = None
            self.run_btn.setEnabled(False)
        else:
            self.training_data = data
            self.run_btn.setEnabled(True)

    def _update_version(self, directory):
        """
        Parses selected version directory and emits signal to update each ModelDialog

            # Arguments
                directory(String): directory selected by user.
        """
        version = directory.split('\\')[-1]
        
        self.selected_version = version
        
        # Emit signal
        self.comms.version_change.emit(directory)


    def _update_selected_models(self, model, state):
        truth = 0
        if state == 2:
            truth = 1
        print("Updated selected model {} to state {}".format(model, truth))
        self.selected_models[model] = truth


    def _enable_tuning_ui(self, state):
        truth = 0
        if state == 2:
            truth = 1
        self.sklearn_tuning_groupbox.setEnabled(truth)
        self.tensorflow_tuning_groupbox.setEnabled(truth)

    def _update_training_params(self, model_base, param, value):
        
        if model_base is None or param is None:
            return
        print(model_base, param, value)
        try:
            # FIXME: This is super hackish.  Can it be done more eloquently?
            if model_base == 'sklearn':
                self.training_params[model_base] = {}
            self.training_params[model_base][param] = value
        except KeyError as ke:
            print(ke)

        print(json.dumps(self.training_params, indent=2))




    