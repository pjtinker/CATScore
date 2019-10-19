
"""QDialog for file defined models.
"""
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
from PyQt5.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout,
                             QGroupBox, QWidget, QLineEdit, QGridLayout,
                             QDialog, QSpinBox, QDialogButtonBox, QComboBox,
                             QDoubleSpinBox, QSizePolicy, QLabel)
from PyQt5.QtGui import QColor

import json
import re
import importlib
import traceback
import inspect
import logging
import os
import functools

from package.utils.catutils import CATEncoder
from package.utils.catutils import cat_decoder


class Communicate(QObject):
    check_for_existing_model = pyqtSignal(str, bool)


class BaseModelDialog(QDialog):
    """
    BaseModelDialog is the basic structure behind model dialogs in CATScore.

    # Arguments
        model_params: String, path to default parameters .json file.
        tfidf_params: String, path to default TF-IDF param file.
        fs_params: String, path to default feature selection file.
    """

    def __init__(self,
                 parent=None,
                 *params):
        super(BaseModelDialog, self).__init__(parent)
        # self.logger = logging.getLogger(__name__)
        # self.comms = Communicate()
        # self.comms.check_for_existing_model.connect(self.parent().model_exists)

        # self.model_params = {}
        # self.updated_params = {}
        # self.ui_widgets = []
        # # input_widgets is a list of all dynamically created input widgets for the various model params.
        # # Holds EVERY input widget, regardless of type.  Key = hyperparameter name
        # self.input_widgets = {}
        # self.current_version = 'default'
        # self.params = params
        # self.main_model_name = params[0]['model_class']
        # # print(self.main_model_name)
        # for param in self.params:
        #     cls_name = param['model_class']
        #     full_name = param['model_module'] + '.' + param['model_class']
        #     self.model_params[full_name] = param[cls_name]
        #     self.updated_params[full_name] = {}

        # self.is_dirty = False
        # self.check_for_default()

        # self.setWindowTitle(self.main_model_name)
        # self.buttonBox = QDialogButtonBox(
        #     QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        # self.buttonBox.setObjectName("model_buttonbox")
        # self.buttonBox.accepted.connect(self.accept)
        # self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        # self.buttonBox.rejected.connect(self.reject)
        
        # self.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(
        #     lambda: self.update_version(self.current_version))
        
        # self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(
        #     lambda: self.apply_changes())
        # self.main_layout = QVBoxLayout()
        # self.form_grid = QGridLayout()
        # self.version_item_combobox = QComboBox()
        # self.version_item_combobox.currentIndexChanged.connect(
        #     lambda state, y=self.version_item_combobox: self.load_version_params(
        #         y.currentData())
        # )
        # self.form_grid.addWidget(self.version_item_combobox, 0, 0)

        # self.main_layout.addLayout(self.form_grid)
        # self.main_layout.addWidget(self.buttonBox)
        # self.setLayout(self.main_layout)

    @property
    def get_model_name(self):
        return self.main_model_name

    def get_class(self, params, init_class=None):
        """
        Return instantiated class using importlib
        Loads any static parameters defined by the system.
            # Arguments
                params: dict, dictionary of parameters necessary to instantiate
                        the class.
            # Returns
                model: instantiated class
        """
        model = None
        module = importlib.import_module(params['model_module'])
        module_class = getattr(module, params['model_class'])
        if "static_params" in params:
            model = module_class(**params['static_params'])
        else:
            model = module_class(init_class)
        return model

    def apply_changes(self):
        pass

    def save_params(self):
        """
        Saves the model parameters entered by the user. If default version is selected,
        return without saving.
        """
        if (self.exec_() == QDialog.Accepted):
            self.apply_changes()


    def _split_key(self, key):
        return key.split('__')[1]


    def _update_param(self, param_type, key, value, callback=None, **cbargs):
        if self.current_version != 'default':
            if callback:
                functools.partial(callback, key, value)
            self.updated_params[param_type][key] = value
            self.is_dirty = True

        
    def setupUI(self, param_type, param_dict, form):
     pass

    @pyqtSlot(str)
    def update_version(self, directory):
        """
        Updates the question combobox based upon selected directory.

            # Arguments:
                directory (String): path to top of version directory.  If 'default', 
                load default values.  
        """
        self.is_dirty = False
        self.current_version = directory
        # Clear combobox to be reconstructed or blank if default.
        self.version_item_combobox.clear()
        if self.current_version.split('\\')[-1] == 'default':
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            self.buttonBox.button(QDialogButtonBox.Apply).setEnabled(False)
            model_exists = False
            for val in os.listdir(self.current_version):
                path = os.path.join(self.current_version, val)
                if os.path.isdir(path):
                    for fname in os.listdir(path):
                        if fname == self.main_model_name + '.pkl' or fname == self.main_model_name + '.h5':
                            model_exists = True
                            break
            self.comms.check_for_existing_model.emit(
                self.main_model_name, model_exists)
            return
        else:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
            self.buttonBox.button(QDialogButtonBox.Apply).setEnabled(True)
        try:
            question_directories = [os.path.join(directory, o) for o in os.listdir(
                directory) if os.path.isdir(os.path.join(directory, o))]
            # Sort the directories correctly.
            # Takes the last digit given as the sort key.
            # If no numeric values, regex will return an empty list and sorting will be
            # alphabetic.
            question_directories = sorted(question_directories,
                                          key=lambda item:
                                            (int(re.findall('\d+|D+', item)[0])
                                                if len(re.findall('\d+|D+', item)) > 0
                                                else float('inf'), item)
                                          )
            model_exists = False
            for d in question_directories:
                # self.comms.check_for_existing_model.emit("Test", True)
                combo_text = d.split('\\')[-1]
                for val in os.listdir(d):
                    path = os.path.join(d, val)
                    # print("val in ModelDialog:", path)
                    if os.path.isdir(path):
                        for fname in os.listdir(path):
                            if fname == self.main_model_name + '.pkl' or fname == self.main_model_name + '.h5':
                                combo_text = combo_text + "*"
                                model_exists = True
                self.version_item_combobox.addItem(combo_text, d)

            self.comms.check_for_existing_model.emit(
                self.main_model_name, model_exists)

            self.form_grid.addWidget(self.version_item_combobox, 0, 0)
            self.update()
        except FileNotFoundError as fnfe:
            pass
        except Exception as e:
            self.logger.error(
                "Error loading updated version directories.", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)
        # self.version_item_combobox.show()

    def load_version_params(self, path):
        pass

    def set_input_params(self, param_dict):
        """
        Set input parameter values based on passed parameter dict.  Iterates through
        key/value pairs => input_widget key / input value

            # Arguments
                param_dict(dict): dictionary of parameter key/value pairs.  
                    Key references name of input widget.

            # Returns
                None
        """
        for k, v in param_dict.items():
            # If v is dictionary, function was called using default values.
            # Set v equal to the default value of that parameter.
            # Must check if default is in the dict, as other dicts exist that are not default values.
            if isinstance(v, dict) and 'default' in v:
                v = v['default']
            if isinstance(v, list):
                v = v[-1]
            if k in self.input_widgets:
                cla = self.input_widgets[k]
                if isinstance(cla, QComboBox):
                    idx = cla.findData(v)
                    if idx != -1:
                        cla.setCurrentIndex(idx)
                elif isinstance(cla, QLineEdit):
                    cla.setText(v)
                else:
                    cla.setValue(v)

    def setupPerformanceUI(self):
        train_date_label = QLabel('Last Training Date:')
        train_date = QLabel('None', objectName='last_train_date')
        train_eval_label = QLabel('Training Evaluation Score:')
        train_eval = QLabel('None', objectName='train_eval_score')
        checksum_label = QLabel('Checksum:')
        checksum = QLabel('None', objectName='checksum')
        self.training_meta_form.addRow(train_date_label,train_date)
        self.training_meta_form.addRow(train_eval_label, train_eval)
        self.training_meta_form.addRow(checksum_label, checksum)
        


    @pyqtSlot(bool)
    def check_for_default(self, force_reload=False):
        """
        Checks for the existance of a default value file.  If none found,
        one is created.
        """
        pass



