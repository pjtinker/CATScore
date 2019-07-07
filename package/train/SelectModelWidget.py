import json
import logging
import os
import traceback
from collections import OrderedDict
from functools import partial

import pandas as pd
import pkg_resources
from PyQt5.QtCore import QObject, Qt, QThread, QThreadPool, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtWidgets import (QAction, QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QRadioButton,
                               QScrollArea, QSizePolicy, QSpinBox, QTabWidget,
                               QVBoxLayout)


from package.train.models.SkModelDialog import SkModelDialog
from package.train.models.TfModelDialog import TfModelDialog
from package.train.ModelTrainer import ModelTrainer
from package.utils.catutils import exceptionWarning

# from addict import Dict


BASE_MODEL_DIR = "./package/data/base_models"
BASE_TF_MODEL_DIR = "./package/data/tensorflow_models"
BASE_TFIDF_DIR = "./package/data/feature_extractors/TfidfVectorizer.json"
BASE_FS_DIR = "./package/data/feature_selection/SelectKBest.json"
DEFAULT_MODEL_DIR = ".\\package\\data\\versions\\default"

class Communicate(QObject):
    version_change = pyqtSignal(str)
    enable_training_btn = pyqtSignal(Qt.CheckState)

class SelectModelWidget(QTabWidget):
    """QTabWidget that holds all of the selectable models and the accompanying ModelDialog for each.
    """
    update_statusbar = pyqtSignal(str)
    update_progressbar = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super(SelectModelWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.training_data = pd.DataFrame()

        self.selected_version = DEFAULT_MODEL_DIR
        self.comms = Communicate()

        self.selected_models = {}
        self.selected_models['sklearn'] = {}
        self.selected_models['tensorflow'] = {}
        self.model_checkboxes = []
        # Initialize training parameter dict.  
        # Has entry for both model base types
        self.training_params = {}
        self.training_params['sklearn'] = {}
        self.training_params['sklearn']['type'] = None
        self.training_params['sklearn']['value'] = None
        self.training_params['tensorflow'] = {}
        # Init tuning param dict
        # Currently only using gridsearch
        self.tuning_params = {}
        self.tuning_params['gridsearch'] = {}

        self.sklearn_model_dialogs = []
        self.sklearn_model_dialog_btns = []
        self.sklearn_training_inputs = []


        self.tensorflow_training_inputs = []
        self.tensorflow_model_dialogs = []
        self.tensorflow_model_dialog_btns = []

        self.main_layout = QVBoxLayout()
        self.upper_hbox = QHBoxLayout()

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
        self.main_layout.addLayout(self.upper_hbox)

        self.model_vbox = QVBoxLayout()
        # self.model_vbox.addSpacing(100)
        self.tuning_vbox = QVBoxLayout()

        self.upper_hbox.addLayout(self.model_vbox)
        self.upper_hbox.addSpacing(10)
        self.upper_hbox.addLayout(self.tuning_vbox)
        self.upper_hbox.addSpacing(200)
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

        # self.sklearn_tuning_groupbox = QGroupBox("Tuning")
        # self.sklearn_hbox.addWidget(self.sklearn_tuning_groupbox)

        # self.main_layout.addWidget(self.sklearn_groupbox)
        self.model_vbox.addWidget(self.sklearn_groupbox)

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

        # self.tensorflow_tuning_groupbox = QGroupBox("Tuning")
        # self.tensorflow_hbox.addWidget(self.tensorflow_tuning_groupbox)

        # self.main_layout.addWidget(self.tensorflow_groupbox)
        self.model_vbox.addWidget(self.tensorflow_groupbox)

        self.tuning_groupbox = QGroupBox("Tuning")
        self.tuning_form = QFormLayout()
        self.tuning_groupbox.setLayout(self.tuning_form)
        self.tuning_vbox.addWidget(self.tuning_groupbox)
        self.tuning_groupbox.setEnabled(False)
        self.model_form_grid = QGridLayout()

        self.setup_model_selection_ui()
        self.setup_training_ui()
        self.setup_tuning_ui()
        
        self.main_layout.addStretch()
        self.run_btn = QPushButton("Train Models")
        self.run_btn.clicked.connect(lambda: self.train_models())
        self.run_btn.setEnabled(False)

        self.comms.enable_training_btn.connect(self.set_training_btn_state)
        self.main_layout.addWidget(self.run_btn)
        self.setLayout(self.main_layout)
        
        # Trigger update to load model parameters
        self._update_version(self.version_selection.currentData())

    def setup_model_selection_ui(self):
        """
        Setup model selection ui.

        The order of the parameters in ModelDialog matters.  model_data must come first!
        """
        self.version_selection_label = QLabel("Select version: ")
        self.version_selection = QComboBox(objectName='version_select')
        # Changed default models to a unique directory.  This
        # is where default models will be saved.  
        self.version_selection.addItem('default', '.\\package\\data\\default_models\\default')
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
        # Dynamically generate ModelDialogs for each model in the base model directory.
        # Only loads .json files.
        try:
            row = 0
            for filename in os.listdir(BASE_MODEL_DIR):
                if filename.endswith('.json'):
                    with open(os.path.join(BASE_MODEL_DIR, filename), 'r') as f:
                        print("Loading model:", filename)
                        model_data = json.load(f)
                        model = model_data['model_class']
                        model_base = model_data['model_base']
                        
                        # The order of the arguments matters!  model_data must come first. 
                        if model_base == 'tensorflow':
                            model_dialog = SkModelDialog(self, model_data)
                        else:
                            model_dialog = SkModelDialog(self, model_data, tfidf_data, self.fs_params)
                        self.comms.version_change.connect(model_dialog.update_version)
                        # Initialize model as unselected
                        self.selected_models[model_base][model] = False
                        btn = QPushButton(model, objectName= model + '_btn')
                        # Partial allows the connection of dynamically generated QObjects
                        btn.clicked.connect(partial(self.open_dialog, model_dialog))
                        chkbox = QCheckBox(objectName=model)
                        chkbox.stateChanged.connect(lambda state, x=model, y=model_base :
                                                self._update_selected_models(x, y, state))
                        if model_base == 'tensorflow':
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
        

            

    def setup_training_ui(self):
        """
        Build ui components for training parameters for both Sklearn and Tensorflow
        """
        # Sklearn training first
        self.cv_n_fold_input = QSpinBox(objectName='n_folds')
        self.cv_n_fold_input.setRange(2, 10)
        self.cv_n_fold_input.setValue(5)
        self.cv_n_fold_input.setEnabled(False)
        self.cv_n_fold_input.valueChanged.connect(
            lambda state, x=self.cv_n_fold_input:
                self.update_training_params('sklearn', 'cv', x.value())
            )
        self.cv_radio_btn = QRadioButton("Cross-validation", objectName='cv')
        self.cv_radio_btn.toggled.connect(lambda state, x=self.cv_n_fold_input: 
                                        self._update_sklearn_training_type('cv', x.value())
                                    )
        
        # self.sklearn_training_inputs.append(self.cv_radio_btn)
        # self.sklearn_training_inputs.append(self.cv_n_fold_input)
        self.sklearn_training_form.addRow(self.cv_radio_btn, self.cv_n_fold_input)
        
        self.sk_validation_radio_btn = QRadioButton("Validation set")
        self.sk_validation_percent_input = QDoubleSpinBox(objectName='test_split')
        self.sk_validation_percent_input.setRange(0.05, 1)
        # self.sk_validation_percent_input.setValue(0.2)
        self.sk_validation_percent_input.setSingleStep(0.1)
        self.sk_validation_percent_input.setEnabled(False)
        self.sk_validation_percent_input.valueChanged.connect(
            lambda state, x=self.sk_validation_percent_input:
                self.update_training_params('sklearn', 'validation', x.value())
            )

        self.sk_validation_radio_btn.toggled.connect(
            lambda state, x=self.sk_validation_percent_input:
                self._update_sklearn_training_type('validation', x.value())
            )
        # self.sklearn_training_inputs.append(self.sk_validation_radio_btn)
        # self.sklearn_training_inputs.append(self.sk_validation_percent_input)
        self.sklearn_training_form.addRow(self.sk_validation_radio_btn, self.sk_validation_percent_input)
    
        self.no_eval_btn = QRadioButton("No evaluation set", objectName='no_eval')
        self.no_eval_btn.toggled.connect(lambda: 
                                        self._update_sklearn_training_type(None, None)
                                   )
        # self.sklearn_training_inputs.append(self.no_eval_btn)
        self.sklearn_training_form.addRow(self.no_eval_btn)

        tf_val_label = QLabel("Validation split")
        tf_val_input = QDoubleSpinBox(objectName='validation_split')
        tf_val_input.setRange(0.05, 1)
        tf_val_input.setSingleStep(0.1)
        tf_val_input.valueChanged.connect(
            lambda state, x=tf_val_input:
                self.update_training_params('tensorflow', 'validation_split', x.value())
            )
        tf_val_input.setValue(0.2)
        # self.tensorflow_training_inputs.append([tf_val_input, tf_val_label])
        self.tensorflow_training_form.addRow(tf_val_label, tf_val_input)

        self.tf_patience_label = QLabel("Patience")
        self.tf_patience_input = QSpinBox(objectName='patience')
        self.tf_patience_input.setRange(0, 1000)
        self.tf_patience_input.setSingleStep(1)
        self.tf_patience_input.valueChanged.connect(
            lambda state, x=self.tf_patience_input:
                self.update_training_params('tensorflow', 'patience', x.value())
            )
        self.tf_patience_input.setValue(2)
        # self.tensorflow_training_inputs.append([self.tf_patience_label, self.tf_patience_input])
        self.tensorflow_training_form.addRow(self.tf_patience_label, self.tf_patience_input)

        self.tf_embedding_type_label = QLabel("Embedding type")
        self.tf_embedding_combobox = QComboBox(objectName='embedding_type')
        self.tf_embedding_combobox.addItem('GloVe', 'glove')
        self.tf_embedding_combobox.addItem('Word2Vec', 'word2vec')
        self.tf_embedding_combobox.addItem('Generate', '')
        self.tf_embedding_combobox.setCurrentIndex(1)
        self.tf_embedding_combobox.currentIndexChanged.connect(
            lambda state, x=self.tf_embedding_combobox:
                self.update_training_params('tensorflow', 'embedding_type', x.currentData())
            )
        self.tf_embedding_combobox.setCurrentIndex(0)
        # self.tensorflow_training_inputs.append([self.tf_embedding_type_label, self.tf_embedding_combobox])
        self.tensorflow_training_form.addRow(self.tf_embedding_type_label, self.tf_embedding_combobox)

        self.tf_embedding_dims_label = QLabel("Embedding dims")
        self.tf_embedding_dims_input = QSpinBox(objectName='embedding_dims')
        self.tf_embedding_dims_input.setRange(100, 300)
        self.tf_embedding_dims_input.setSingleStep(100)
        self.tf_embedding_dims_input.valueChanged.connect(
            lambda state, x=self.tf_embedding_dims_input:
                self.update_training_params('tensorflow', 'embedding_dims', x.value())
            )
        self.tf_embedding_dims_input.setValue(100)
        # self.tensorflow_training_inputs.append([self.tf_embedding_dims_label, self.tf_embedding_dims_input])
        self.tensorflow_training_form.addRow(self.tf_embedding_dims_label, self.tf_embedding_dims_input)

        self.tf_embedding_trainable_label = QLabel("Train embeddings")
        self.tf_embedding_trainable_chkbox = QCheckBox(objectName='embedding_trainable')
        self.tf_embedding_trainable_chkbox.stateChanged.connect(
            lambda state, x=self.tf_embedding_trainable_chkbox:
                self.update_training_params('tensorflow', 'embedding_trainable', x.isChecked())
            )
        self.tf_embedding_trainable_chkbox.setChecked(True)
        # self.tensorflow_training_inputs.append([self.tf_embedding_trainable_label, self.tf_embedding_trainable_chkbox])
        self.tensorflow_training_form.addRow(self.tf_embedding_trainable_label, self.tf_embedding_trainable_chkbox)

        self.cv_radio_btn.toggle()

    def setup_tuning_ui(self):
        self.tuning_n_iter_label = QLabel("Number of iterations")
        self.tuning_n_iter_input = QSpinBox(objectName='n_iter')
        self.tuning_n_iter_input.setRange(2, 1000)
        self.tuning_n_iter_input.setSingleStep(1)
        self.tuning_n_iter_input.setValue(10)
        self.tuning_n_iter_input.valueChanged.connect(
            lambda state, x=self.tuning_n_iter_input:
                self.update_tuning_params('gridsearch', 'n_iter', x.value())
        )
        self.tuning_form.addRow(self.tuning_n_iter_label, self.tuning_n_iter_input)


    def open_dialog(self, dialog):
        """
        Opens the passed ModelDialog via the save_params function, allowing the user
        to specify hyperparameters for each available version field.  

            # Arguments
                dialog(ModelDialog): Specified model dialog.
        """
        dialog.save_params()


    @pyqtSlot(str)
    def add_new_version(self, v_dir):
        """
        pyqtSlot to receive new version created pyqtSignal.

            # Arguments
                v_dir(String): directory of newly created version.
        """
        version = v_dir.split('\\')[-1]
        self.version_selection.addItem(version, v_dir)

    @pyqtSlot(pd.DataFrame)
    def load_data(self, data):
        """
        pyqtSlot to receive pandas DataFrame after DataLoader has completed it's work

            # Arguments
                data(pandas.DataFrame): DataFrame of training data
        """
        self.training_data = data
        self.comms.enable_training_btn.emit(True)
        # if(data.empty):
        #     self.run_btn.setEnabled(False)
        # else:
        #     # self.training_data = data
        #     self.run_btn.setEnabled(True)

    @pyqtSlot(Qt.CheckState)
    def set_training_btn_state(self, state):
        if (not self.training_data.empty 
            and 
                (1 in self.selected_models['sklearn'].values() 
            or 
                1 in self.selected_models['tensorflow'].values()) 
            ):
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setEnabled(False)

    @pyqtSlot(str, bool)
    def model_exists(self, model_name, truth):
        btn = self.findChild(QPushButton, model_name + '_btn')
        if btn:
            text = btn.text()
            if text.endswith("*"):
                text = text[:-1]
            if truth:
                btn.setText(text + "*")
            else:
                btn.setText(text)
        else:
            return

    def train_models(self):
        train_models = self.tune_models_chkbox.isChecked()
        self.model_trainer = ModelTrainer(self.selected_models,
                               self.selected_version,
                               self.training_params,
                               self.training_data,
                               train_models,
                               self.tuning_n_iter_input.value())
        self.threadpool.start(self.model_trainer)

    def _update_version(self, directory):
        """
        Parses selected version directory and emits pyqtSignal to update each ModelDialog

            # Arguments
                directory(String): directory selected by user.
        """
        self.selected_version = directory
        # Emit pyqtSignal
        self.comms.version_change.emit(directory)


    def _update_selected_models(self, model, model_base, state):
        """
        Update the models selected by the user.  This function is connected to the
        checkboxes associated with each model.

            # Arguments:
                model(String): name of the selected model
                state(bool): the truth of the selection.  True->selected, False->unselected
        """
        truth = False
        if state == Qt.Checked:
            truth = True
        self.selected_models[model_base][model] = truth
        self.comms.enable_training_btn.emit(truth)

    def _enable_tuning_ui(self, state):
        """
        Helper function to enable/disable the tuning parameter UI if selected 
        by the user.

            # Arguments:
                state(bool): the state of tuning.  False->no tuning, True->tune models
        """
        self.tuning_groupbox.setEnabled(state)


    def update_tuning_params(self, model_base, param, value):
        pass


    def update_training_params(self, model_base, param, value):
        """
        Update the various training parameters with values supplied by the user.
        Needs work as the sklearn training parameters are mutually exclusive.

            # Arguments
                model_base(String): model base for specified training params
                param(String): parameter name
                value(String, int, or double): value of specified parameter
        """
        if model_base is None or param is None:
            return
        print(model_base, param, value)
        try:
            # FIXME: This is super hackish and brittle.  Can it be done more eloquently?
            if model_base == 'sklearn':
                self._update_sklearn_training_type(param, value)
            else:
                self.training_params[model_base][param] = value
        except KeyError as ke:
            print(ke)

        print(json.dumps(self.training_params, indent=2))

    def _update_sklearn_training_type(self, eval_type, value):
        """
        SKlearn model tuning is mutually exclusive.  This helper function
        Enables/disables the appropriate field and updates the appropriate
        parameters of self.training_params

        Currently, for SKlearn models, only cross-validation (cv) or a holdout set
        (validation) or None are model evaluation options.  

            # Arguments
                eval_type(String): The type of model evaluation specified by the user.
                value(int or double): value corresponding to selected type
        """
        truth = False
        if eval_type == 'cv':
            self.cv_n_fold_input.setEnabled(not truth)
            self.sk_validation_percent_input.setEnabled(truth)     
        elif eval_type == 'validation':
            self.cv_n_fold_input.setEnabled(truth)
            self.sk_validation_percent_input.setEnabled(not truth)
        elif eval_type == None:
            self.cv_n_fold_input.setEnabled(False)
            self.sk_validation_percent_input.setEnabled(False)

        self.training_params['sklearn']['type'] = eval_type
        self.training_params['sklearn']['value'] = value
        print(json.dumps(self.training_params, indent=2))
