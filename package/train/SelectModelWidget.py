import json
import logging
import os
import traceback
import time
from collections import OrderedDict
from functools import partial
import hashlib

import pandas as pd
import pkg_resources
from PyQt5.QtCore import QObject, Qt, QThread, QThreadPool, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtWidgets import (QAction, QButtonGroup, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QButtonGroup, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy, QSpinBox, QTabWidget,
                             QVBoxLayout, QTextEdit, QWidget)
from PyQt5.QtGui import QTextCursor, QIcon, QPixmap

from package.train.models.SkModelDialog import SkModelDialog
from package.train.models.TPOTModelDialog import TPOTModelDialog
from package.train.ModelTrainer import ModelTrainer
from package.utils.catutils import exceptionWarning
from package.utils.config import CONFIG


class Communicate(QObject):
    version_change = pyqtSignal(str)
    enable_training_btn = pyqtSignal(Qt.CheckState)
    stop_training = pyqtSignal()
    update_statusbar = pyqtSignal(str)

class SelectModelWidget(QWidget):
    """QTabWidget that holds all of the selectable models and the accompanying ModelDialog for each.
    """
    update_progressbar = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super(SelectModelWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.threadpool = QThreadPool()
        self.logger.info(
            f"Multithreading enabled with a maximum of {self.threadpool.maxThreadCount()} threads.")

        print("Multithreading with maximum %d threads" %
                self.threadpool.maxThreadCount())
        self.training_data = pd.DataFrame()
        self.training_predictions = pd.DataFrame()
        self.selected_version = CONFIG.get('PATHS', 'DefaultModelDirectory')
        self.comms = Communicate()

        self.selected_models = {}
        self.selected_models['sklearn'] = {}
        self.selected_models['tensorflow'] = {}
        self.model_checkboxes = []
        # * Initialize training parameter dict.
        # * Has entry for both model base types
        self.training_params = {}
        self.training_params['sklearn'] = {}
        self.training_params['sklearn']['type'] = None
        self.training_params['sklearn']['value'] = None
        self.training_params['tensorflow'] = {}
        # * Init tuning param dict
        # * Currently only using gridsearch
        self.tuning_params = {}
        self.tuning_params['gridsearch'] = {
            'n_iter': 20,
            'cv': 3,
            'n_jobs': -1,
            'scoring': ['accuracy'],
            'tune_stacker' : False
        }

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
                                                     self._enable_tuning_ui(
                                                         state)
                                                     )
        self.main_layout.addLayout(self.header_hbox)
        self.main_layout.addLayout(self.upper_hbox)

        self.model_vbox = QVBoxLayout()
        self.tuning_vbox = QVBoxLayout()

        self.upper_hbox.addLayout(self.model_vbox)
        self.upper_hbox.addSpacing(10)
        self.upper_hbox.addLayout(self.tuning_vbox)
        self.upper_hbox.addSpacing(200)
        # * Build sklearn ui components
        self.sklearn_hbox = QHBoxLayout()
        self.sklearn_groupbox = QGroupBox("Sklearn")
        self.sklearn_groupbox.setLayout(self.sklearn_hbox)

        self.skmodel_groupbox = QGroupBox("Model Selection")
        self.sklearn_hbox.addWidget(self.skmodel_groupbox)
        self.sklearn_model_form = QFormLayout()
        self.sklearn_model_form.setFormAlignment(Qt.AlignTop)
        self.skmodel_groupbox.setLayout(self.sklearn_model_form)

        # Sklearn training and tuning ui components
        self.sklearn_training_groupbox = QGroupBox("Training")
        self.sklearn_training_form = QFormLayout()
        self.sklearn_training_groupbox.setLayout(self.sklearn_training_form)
        self.sklearn_hbox.addWidget(self.sklearn_training_groupbox)

        self.model_vbox.addWidget(self.sklearn_groupbox)

        # * Build Tensorflow ui components
        self.tensorflow_hbox = QHBoxLayout()
        self.tensorflow_groupbox = QGroupBox("Tensorflow")
        self.tensorflow_groupbox.setLayout(self.tensorflow_hbox)

        self.tensorflow_model_groupbox = QGroupBox("Model Selection")
        self.tensorflow_hbox.addWidget(self.tensorflow_model_groupbox)
        self.tensorflow_model_form = QFormLayout()

        self.tensorflow_model_groupbox.setLayout(self.tensorflow_model_form)
        self.tensorflow_training_groupbox = QGroupBox("Training")
        self.tensorflow_training_form = QFormLayout()
        self.tensorflow_training_groupbox.setLayout(
            self.tensorflow_training_form)
        self.tensorflow_hbox.addWidget(self.tensorflow_training_groupbox)

        # This is the tensorflow groupbox for models and training params.
        # self.model_vbox.addWidget(self.tensorflow_groupbox)

        self.tuning_groupbox = QGroupBox("Tuning")
        self.tuning_form = QFormLayout()
        self.tuning_groupbox.setLayout(self.tuning_form)
        self.tuning_vbox.addWidget(self.tuning_groupbox)
        self.tuning_groupbox.setEnabled(False)
        self.model_form_grid = QGridLayout()

        self.setup_model_selection_ui()
        self.setup_training_ui()
        self.setup_tuning_ui()
        # QTextEdit box for training/tuning status
        self.training_logger = QTextEdit()
        self.training_logger.setReadOnly(True)
        self.training_logger.setAcceptRichText(True)
        self.training_logger.insertHtml(
            "<i>Multithreading with maximum %d threads</i><br>" % self.threadpool.maxThreadCount())
        self.training_logger.setMinimumHeight(400)
        self.main_layout.addWidget(self.training_logger)
        self.clear_btn_hbox = QHBoxLayout()
        self.clear_text_btn = QPushButton('Clear')
        self.clear_text_btn.setMaximumWidth(50)
        self.clear_text_btn.clicked.connect(lambda: self.training_logger.clear())
        self.clear_btn_hbox.addStretch()
        self.clear_btn_hbox.addWidget(self.clear_text_btn)

        self.main_layout.addLayout(self.clear_btn_hbox)

        self.main_layout.addStretch()
        self.run_btn = QPushButton("&Train Models")
        self.run_btn.setMinimumWidth(200)
        self.run_btn.clicked.connect(lambda: self.train_models())
        self.run_btn.setEnabled(False)

        self.stop_btn = QPushButton('Sto&p')
        self.stop_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.comms.enable_training_btn.connect(self.set_training_btn_state)
        self.button_hbox = QHBoxLayout()

        icon = QIcon()
        icon.addPixmap(QPixmap('icons/Programming-Save-icon.png'))
        self.save_results_btn = QPushButton()
        self.save_results_btn.setIcon(icon)
        self.save_results_btn.setEnabled(False)
        self.save_results_btn.clicked.connect(lambda: self.save_predictions())

        self.button_hbox.addWidget(self.run_btn)
        self.button_hbox.addWidget(self.stop_btn)
        self.button_hbox.addStretch()
        self.button_hbox.addWidget(self.save_results_btn)
        self.main_layout.addLayout(self.button_hbox)
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
        # self.version_selection.addItem(
        #     'default', '.\\package\\data\\default_models\\default')
        available_versions = os.listdir(".\\package\\data\\versions")
        for version in available_versions:
            v_path = os.path.join('.\\package\\data\\versions', version)
            if os.path.isdir(v_path):
                self.version_selection.addItem(version, v_path)
        self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection:
                                                           self._update_version(
                                                               y.currentData())
                                                           )
        self.version_form.addRow(
            self.version_selection_label, self.version_selection)

        # Load base TF-IDF features
        # and feature selection data
        try:
            with open(CONFIG.get('PATHS', 'BaseTfidfDirectory'), 'r') as f:
                tfidf_data = json.load(f)
        except IOError as ioe:
            self.logger.error("Error loading base TFIDF params", exc_info=True)
            exceptionWarning(
                'Error occurred while loading base TFIDF parameters.', repr(ioe))
        try:
            with open(CONFIG.get('PATHS', 'BaseFeatureSeletionDirectory'), 'r') as f:
                self.fs_params = json.load(f)
        except IOError as ioe:
            self.logger.error(
                "Error loading base feature selector params", exc_info=True)
            exceptionWarning(
                'Error occurred while loading base feature selector parameters.', repr(ioe))
        # Dynamically generate ModelDialogs for each model in the base model directory.
        # Only considers *.json file extension.
        try:
            row = 0
            for filename in os.listdir(CONFIG.get('PATHS', 'BaseModelDirectory')):
                if filename.endswith('.json'):
                    with open(os.path.join(CONFIG.get('PATHS', 'BaseModelDirectory'), filename), 'r') as f:
                        # print("Loading model:", filename)
                        model_data = json.load(f)
                        model = model_data['model_class']
                        model_base = model_data['model_base']
                        model_module = model_data['model_module']
                        # The order of the arguments matters!  model_data must come first.
                        if model_base == 'tensorflow':
                            continue
                            # model_dialog = SkModelDialog(self, model_data)
                        if model_module == 'tpot':
                            model_dialog = TPOTModelDialog(
                                self, model_data, tfidf_data)
                        else:
                            model_dialog = SkModelDialog(
                                self, model_data, tfidf_data, self.fs_params)
                        self.comms.version_change.connect(
                            model_dialog.update_version)
                        # Initialize model as unselected
                        self.selected_models[model_base][model] = False
                        btn = QPushButton(model, objectName=model + '_btn')
                        # Partial allows the connection of dynamically generated QObjects
                        btn.clicked.connect(
                            partial(self.open_dialog, model_dialog))
                        chkbox = QCheckBox(objectName=model)
                        chkbox.stateChanged.connect(lambda state, x=model, y=model_base:
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
            self.logger.error(
                "OSError opening model config files", exc_info=True)
            exceptionWarning('OSError opening model config files!', ose)
            tb = traceback.format_exc()
            print(tb)
        except Exception as e:
            self.logger.error(
                "Error opening model config files", exc_info=True)
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
                                          self._update_sklearn_training_type(
                                              'cv', x.value())
                                          )
        self.sklearn_training_form.addRow(
            self.cv_radio_btn, self.cv_n_fold_input)

        self.sk_validation_radio_btn = QRadioButton("Validation set")
        self.sk_validation_percent_input = QDoubleSpinBox(
            objectName='test_split')
        self.sk_validation_percent_input.setRange(0.05, 1)
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
        # NOTE: Removing validation split option from evaluation.  It seems less than useful and
        # requires time that could be spent elsewhere as we near the end of our time together.
        # self.sklearn_training_form.addRow(self.sk_validation_radio_btn, self.sk_validation_percent_input)

        self.no_eval_btn = QRadioButton(
            "No evaluation set", objectName='no_eval')
        self.no_eval_btn.toggled.connect(lambda:
                                         self._update_sklearn_training_type(
                                             None, None)
                                         )
        self.sklearn_training_form.addRow(self.no_eval_btn)
        # TENSORFLOW TRAINING UI.  Removed as of 10/04/19

        # Toggle to set params on load
        self.cv_radio_btn.toggle()

        #* Select stacker
        # self.stacker_groupbox = QGroupBox('Stacking algorithm')
        # self.stacker_vbox = QVBoxLayout()
        # self.train_stacker

    def setup_tuning_ui(self):
        self.tuning_n_iter_label = QLabel("Number of iterations:")
        self.tuning_n_iter_input = QSpinBox(objectName='n_iter')
        self.tuning_n_iter_input.setRange(2, 1000)
        self.tuning_n_iter_input.setSingleStep(1)
        self.tuning_n_iter_input.setValue(10)
        self.tuning_n_iter_input.valueChanged.connect(
            lambda state, x=self.tuning_n_iter_input:
                self.update_tuning_params('gridsearch', 'n_iter', x.value())
        )
        self.tuning_form.addRow(self.tuning_n_iter_label,
                                self.tuning_n_iter_input)

        self.tuning_cv_label = QLabel("CV folds:")
        self.tuning_cv_input = QSpinBox(objectName='cv')
        self.tuning_cv_input.setRange(2, 10)
        self.tuning_cv_input.setValue(3)
        self.tuning_cv_input.valueChanged.connect(
            lambda state, x=self.tuning_cv_input:
                self.update_tuning_params('gridsearch', 'cv', x.value())
        )
        self.tuning_form.addRow(self.tuning_cv_label, self.tuning_cv_input)

        self.tuning_n_jobs_label = QLabel("Number of parallel jobs:")
        self.tuning_n_jobs_input = QSpinBox(objectName='n_jobs')
        self.tuning_n_jobs_input.setRange(-1, 4)
        self.tuning_n_jobs_input.setValue(-1)
        self.tuning_n_jobs_input.valueChanged.connect(
            lambda state, x=self.tuning_n_jobs_input:
                self.update_tuning_params('gridsearch', 'n_jobs', x.value())
        )
        self.tuning_form.addRow(self.tuning_n_jobs_label,
                                self.tuning_n_jobs_input)

        self.scoring_metric_groupbox = QGroupBox('Scoring metrics')

        self.scoring_metric_vbox = QVBoxLayout()
        #* The following code is for metric radio buttons.  Left in for posterity
        # self.acc_checkbox = QRadioButton('Accuracy')
        # self.acc_checkbox.setChecked(True)
        # self.acc_checkbox.toggled.connect(
        #     lambda state, x=self.acc_checkbox:
        #     self.update_tuning_params('gridsearch', 'scoring', 'accuracy')
        # )
        # self.scoring_metric_vbox.addWidget(self.acc_checkbox)

        # self.f1_weighted_checkbox = QRadioButton('F1 weighted')
        # self.f1_weighted_checkbox.setChecked(False)
        # self.f1_weighted_checkbox.toggled.connect(
        #     lambda state, x=self.f1_weighted_checkbox:
        #     self.update_tuning_params('gridsearch', 'scoring', 'f1_weighted')
        # )
        # self.scoring_metric_vbox.addWidget(self.f1_weighted_checkbox)

        # self.prec_weighted_checkbox = QRadioButton('Precision weighted')
        # self.prec_weighted_checkbox.setChecked(False)
        # self.prec_weighted_checkbox.toggled.connect(
        #     lambda state, x=self.prec_weighted_checkbox:
        #     self.update_tuning_params(
        #         'gridsearch', 'scoring', 'precision_weighted')
        # )
        # self.scoring_metric_vbox.addWidget(self.prec_weighted_checkbox)

  
        self.acc_checkbox = QCheckBox('Accuracy')
        self.acc_checkbox.setChecked(True)
        self.acc_checkbox.stateChanged.connect(
            lambda state, x=self.acc_checkbox:
            self.update_tuning_params('gridsearch', 'accuracy', state, True)
        )
        self.scoring_metric_vbox.addWidget(self.acc_checkbox)

        self.f1_weighted_checkbox = QCheckBox('F1 weighted')
        self.f1_weighted_checkbox.setChecked(False)
        self.f1_weighted_checkbox.stateChanged.connect(
            lambda state, x=self.f1_weighted_checkbox:
            self.update_tuning_params('gridsearch', 'f1_weighted', state, True)
        )
        self.scoring_metric_vbox.addWidget(self.f1_weighted_checkbox)

        self.prec_weighted_checkbox = QCheckBox('Precision weighted')
        self.prec_weighted_checkbox.setChecked(False)
        self.prec_weighted_checkbox.stateChanged.connect(
            lambda state, x=self.prec_weighted_checkbox:
            self.update_tuning_params(
                'gridsearch', 'precision_weighted', state, True)
        )
        self.scoring_metric_vbox.addWidget(self.prec_weighted_checkbox)

        self.scoring_metric_groupbox.setLayout(self.scoring_metric_vbox)

        self.tune_stacker_checkbox = QCheckBox('Tune Stacking Algorithm')
        self.tune_stacker_checkbox.setChecked(False)
        self.tune_stacker_checkbox.stateChanged.connect(
            lambda state, x=self.tune_stacker_checkbox:
            self.update_tuning_params(
                'gridsearch', 'tune_stacker', state)
        )
        self.tuning_form.addRow(self.scoring_metric_groupbox)
        self.tuning_form.addRow(self.tune_stacker_checkbox)

    def open_dialog(self, dialog):
        """
        Opens the passed ModelDialog via the save_params function, allowing the user
        to specify hyperparameters for each available version field.  

            # Arguments
                dialog: ModelDialog, Specified model dialog.
        """
        dialog.save_params()

    @pyqtSlot(str)
    def add_new_version(self, v_dir):
        """
        pyqtSlot to receive new version created pyqtSignal.

            # Arguments
                v_dir: string, directory of newly created version.
        """
        version = v_dir.split('\\')[-1]
        self.version_selection.addItem(version, v_dir)
        self.version_selection.model().sort(0)

    @pyqtSlot(pd.DataFrame)
    def load_data(self, data):
        """
        pyqtSlot to receive pandas DataFrame after DataLoader has completed it's work

            # Arguments
                data: pandas.DataFrame, training data
        """
        self.training_data = data
        self.comms.enable_training_btn.emit(True)

    @pyqtSlot(Qt.CheckState)
    def set_training_btn_state(self):
        """
        Sets the run button enabled state.
        Checks that there are models selected in sklearn or tensorflow 
        """
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
        """
        Adds styling to button if a trained model exists for the model in the selected version

            # Arguments
                model_name: string, name of the model designated by the button.
                truth: bool, true if there exists any trained model of type model_name in the current
                    version.
        """
        btn = self.findChild(QPushButton, model_name + '_btn')
        if btn:
            text = btn.text()
            if text.endswith("*"):
                text = text[:-2]
            if truth:
                btn.setText(text + " *")
            else:
                btn.setText(text)
        else:
            return

    def train_models(self):
        try:
            tune_models = self.tune_models_chkbox.isChecked()
            self.model_trainer = ModelTrainer(selected_models=self.selected_models,
                                              version_directory=self.selected_version,
                                              training_eval_params=self.training_params,
                                              training_data=self.training_data,
                                              tune_models=tune_models,
                                              tuning_params=self.tuning_params)
            self.model_trainer.signals.update_training_logger.connect(
                self.update_training_logger)
            self.update_progressbar.emit(1, True)
            self.model_trainer.signals.training_complete.connect(
                self.training_complete)
            self.comms.stop_training.connect(self.model_trainer.stop_thread)
            self.run_btn.setEnabled(False)
            self.stop_btn.clicked.connect(lambda: self._abort_training())

            self.training_predictions = pd.DataFrame()
            self.threadpool.start(self.model_trainer)
        except Exception as e:
            self.logger.error("SelectModelWidget.train_models", exc_info=True)
            exceptionWarning('Exception occured when training models.', e)
            tb = traceback.format_exc()
            print(tb)

    @pyqtSlot(str, bool, bool)
    def update_training_logger(self, msg, include_time=True, use_html=True):
        if(include_time):
            current_time = time.localtime()
            outbound = f"{time.strftime('%Y-%m-%d %H:%M:%S', current_time)} - {msg}<br>"
        else:
            outbound = f"{msg}<br>"
        if(use_html):
            self.training_logger.insertHtml(outbound)
            self.training_logger.moveCursor(QTextCursor.End)
        else:
            self.training_logger.insertPlainText(msg)

    @pyqtSlot(pd.DataFrame)
    def training_complete(self, prediction_df=None):
        """
        Resets progressbar, unchecks 'Train models', and emits signal to refresh the parameter
        values in each ModelDialog

            # Arguments
                val: int or float, value used to set progressbar
                pulse: bool, used to toggle progressbar pulse
        """
        self.update_progressbar.emit(0, False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Train models")
        self.tune_models_chkbox.setChecked(False)
        self.save_results_btn.setEnabled(True)

        if(prediction_df is not None and not prediction_df.empty):
            self.training_predictions = prediction_df
        # Emitting a version change here reloads all parameters.  i.e. we update the
        # parameters displayed in the dialog.
        self.comms.version_change.emit(self.selected_version)


    def save_predictions(self):
        try:
            if self.training_predictions.empty:
                exceptionWarning('No predictions to save')
                return
            file_name, filter = QFileDialog.getSaveFileName(
                self, 'Save to CSV', os.getenv('HOME'), 'CSV(*.csv)')
            if file_name:
                self.training_predictions.to_csv(
                    file_name, index_label='testnum', quoting=1, encoding='utf-8')
                self.comms.update_statusbar.emit("Predictions saved successfully.")
        except PermissionError as pe:
            self.logger.warning("SelectModelWidget.save_predictions", exc_info=True)
            exceptionWarning(f'Permission denied while attempting to save {file_name}')
        except Exception as e:
            self.logger.error("SelectModelWidget.save_predictions", exc_info=True)
            exceptionWarning(
                "Exception occured.  SelectModelWidget.save_predictions.", exception=e)
            tb = traceback.format_exc()
            print(tb)


    def _abort_training(self):
        self.comms.stop_training.emit()

    def _update_version(self, directory):
        """
        Parses selected version directory and emits pyqtSignal to update each ModelDialog

            # Arguments
                directory: string, directory selected by user.
        """
        self.selected_version = directory
        # Emit pyqtSignal
        self.comms.version_change.emit(directory)

    def _update_selected_models(self, model, model_base, state):
        """
        Update the models selected by the user.  This function is connected to the
        checkboxes associated with each model.

            # Arguments:
                model: string, name of the selected model
                state: bool, the truth of the selection.  True->selected, False->unselected
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
                state: bool, the state of tuning.  False->no tuning, True->tune models
        """
        self.run_btn.setText("Tune Models" if state else "Train Models")
        self.tuning_groupbox.setEnabled(state)

    def update_tuning_params(self, model_base, param, value, scorer=False):
        if model_base is None or param is None:
            return
        try:
            if scorer:
                if value:
                    self.tuning_params[model_base]['scoring'].append(param)
                else:
                    if param in self.tuning_params[model_base]['scoring']:
                        self.tuning_params[model_base]['scoring'].remove(param)
            else:
                self.tuning_params[model_base][param] = value
            # self.tuning_params[model_base][param] = value
        except KeyError as ke:
            self.tuning_params[model_base][param] = {}
            self.tuning_params[model_base][param] = value
        except Exception as e:
            self.logger.error(
                "SelectModelWidget.update_tuning_params", exc_info=True)
            exceptionWarning('Exception occured when training models.', e)
            tb = traceback.format_exc()
            print(tb)
        print(self.tuning_params)

    def update_training_params(self, model_base, param, value):
        """
        Update the various training parameters with values supplied by the user.
        Needs work as the sklearn training parameters are mutually exclusive.

            # Arguments
                model_base: string, model base for specified training params
                param: string, parameter name
                value: string, int, or double, value of specified parameter
        """
        if model_base is None or param is None:
            return
        # print(model_base, param, value)
        try:
            # FIXME: This is super hackish and brittle.  Can it be done more eloquently?
            if model_base == 'sklearn':
                self._update_sklearn_training_type(param, value)
            else:
                self.training_params[model_base][param] = value
        except KeyError as ke:
            print(ke)

    def _update_sklearn_training_type(self, eval_type, value):
        """
        SKlearn model tuning is mutually exclusive.  This helper function
        Enables/disables the appropriate field and updates the appropriate
        parameters of self.training_params

        Currently, for Sklearn models, only cross-validation (cv) or a holdout set
        (validation) or None are model evaluation options.  

            # Arguments
                eval_type: string, The type of model evaluation specified by the user.
                value: int or double, value corresponding to selected type
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
        else:
            raise ValueError("eval_type %s is invalid" % (eval_type))

        self.training_params['sklearn']['type'] = eval_type
        self.training_params['sklearn']['value'] = value
