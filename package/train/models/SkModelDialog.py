
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


class Communicate(QObject):
    check_for_existing_model = pyqtSignal(str, bool)


class SkModelDialog(QDialog):
    """
    SkModelDialog is the basic structure behind model dialogs in CATScore.

    # Arguments
        model_params: String, path to default parameters .json file.
        tfidf_params: String, path to default TF-IDF param file.
        fs_params: String, path to default feature selection file.
    """

    def __init__(self,
                 parent=None,
                 *params):
        super(SkModelDialog, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.comms = Communicate()
        self.comms.check_for_existing_model.connect(self.parent().model_exists)

        self.model_params = {}
        self.updated_params = {}
        self.ui_widgets = []
        # input_widgets is a list of all dynamically created input widgets for the various model params.
        # Holds EVERY input widget, regardless of type.  Key = hyperparameter name
        self.input_widgets = {}
        self.current_version = 'default'
        self.params = params
        self.main_model_name = params[0]['model_class']
        print(self.main_model_name)
        for param in self.params:
            cls_name = param['model_class']
            full_name = param['model_module'] + '.' + param['model_class']
            self.model_params[full_name] = param[cls_name]
            self.updated_params[full_name] = {}

        self.is_dirty = False
        self.check_for_default()

        self.setWindowTitle(self.main_model_name)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("model_buttonbox")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.rejected.connect(self.reject)
        # self.buttonBox.clicked(QDialogButtonBox.Apply).connect(self.apply_changes())
        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(
            lambda: self.apply_changes())
        self.main_layout = QVBoxLayout()
        self.form_grid = QGridLayout()
        self.question_combobox = QComboBox()
        self.question_combobox.currentIndexChanged.connect(
            lambda state, y=self.question_combobox: self.load_version_params(
                y.currentData())
        )
        self.form_grid.addWidget(self.question_combobox, 0, 0)
        row = 1
        col = 0
        for model, types in self.model_params.items():
            for t, params in types.items():
                groupbox = QGroupBox()
                groupbox.setTitle(model.split('.')[-1] + " " + t)
                model_param_form = QFormLayout()
                groupbox.setLayout(model_param_form)
                self.form_grid.addWidget(groupbox, row, col)
                col += 1
                self.ui_widgets.append(groupbox)
                self.ui_widgets.append(model_param_form)
                self.setupUI(model, params, model_param_form)

        self.main_layout.addLayout(self.form_grid)
        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)
        self.move(20, 10)

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
        version = self.current_version.split('\\')[-1]
        if version == 'default':
            print("Default version selected.  Returning...")
            return
        if self.is_dirty:
            filename = self.main_model_name + '.json'
            save_dir = os.path.join(self.question_combobox.currentData(),
                                    self.main_model_name)

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            save_file_path = os.path.join(save_dir,
                                          filename)

            if not os.path.isfile(save_file_path):
                # Get default file and load those values
                default_dir = os.path.join(
                    ".\\package\\data\\default_models\\default", self.main_model_name)
                default_path = os.path.join(
                    default_dir, self.main_model_name + '.json')
                with open(default_path, 'r') as infile:
                    full_default_params = json.load(infile)
                save_data = {
                    "model_base": self.params[0]['model_base'],
                    "model_module": self.params[0]['model_module'],
                    "model_class": self.main_model_name,
                    "question_number": self.question_combobox.currentData().split('\\')[-1],
                    "version": version,
                    "tuned": False,
                    "params": {}
                }
                save_data['params'] = full_default_params['params']
                print("save_data['params'] in save_params")
                print(json.dumps(save_data['params'], indent=2))
            else:
                with open(save_file_path, 'r') as infile:
                    save_data = json.load(infile)

            print("updated_params:")
            print(json.dumps(self.updated_params, indent=2))
            for param_type, params in self.updated_params.items():
                if(params):
                    for param, val in params.items():
                        save_data['params'][param_type][param] = val
            try:
                with open(save_file_path, 'w') as outfile:
                    json.dump(save_data, outfile, indent=2)
            except Exception as e:
                self.logger.error("Error saving updated model parameters for {}.".format(
                    self.main_model_name), exc_info=True)
                print("Exception {}".format(e))
                tb = traceback.format_exc()
                print(tb)

        self.is_dirty = False
        return

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
        # print("updateParams key, value: {}, {}, {}".format(param_type, key, value))
        #class_key = '__' + key + '__'
        if self.current_version != 'default':
            if callback:
                functools.partial(callback, key, value)

            self.updated_params[param_type][key] = value
            self.is_dirty = True

    def _selectkbest_callback(self, key, value):
        if value == 0:
            value = 'all'
        
    def setupUI(self, param_type, param_dict, form):
        """
        Build UI elements using parameters dict of scikit models

            # Attributes:
                param_type: String, type of param to update
                param_dict: dict, dictionary of parameter/default values from model.
                default_params: dict, dictionary of default parameters defined by me.
        """
        try:
            for k, v in param_dict.items():
                label_string = k
                label = QLabel(label_string)
                val_type = v['type']
                if val_type == 'dropdown':
                    input_field = QComboBox(objectName=k)
                    for name, value in v['options'].items():
                        input_field.addItem(name, value)
                    idx = input_field.findData(v['default'])
                    if idx != -1:
                        input_field.setCurrentIndex(idx)
                    input_field.currentIndexChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x,
                            y.currentData())
                    )
                    # form.addRow(label, input_field)
                    # self.input_widgets[k] = input_field

                elif val_type == 'double':
                    input_field = QDoubleSpinBox(objectName=k)
                    input_field.setDecimals(v['decimal_len'])
                    input_field.setRange(v['min'], v['max'])
                    if v['default'] is not None:
                        input_field.setValue(v['default'])
                    input_field.setSingleStep(v['step_size'])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x,
                            y.value())
                    )

                elif val_type == 'int':
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(v['min'], v['max'])
                    if v['default'] is not None:
                        input_field.setValue(v['default'])
                    input_field.setSingleStep(v['step_size'])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x,
                            y.value())
                    )
                elif val_type == 'range':
                    label_string = k
                    label = QLabel(label_string + ' : 1,')
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(v['min'], v['max'])
                    if v['default'] is not None:
                        input_field.setValue(v['default'][-1])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._update_param(
                                param_type,
                                x,
                                [1, y.value()])
                    )
                elif val_type == 'static':
                    label_string = k
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(str(v['default']))
                    # input_field.textColor(QColor.red())
                    input_field.setEnabled(False)
                    self._update_param(param_type, k, v['default'])
                if v['tooltip'] is not None:
                    input_field.setToolTip(v['tooltip'])
                form.addRow(label, input_field)
                self.input_widgets[k] = input_field
        except Exception as e:
            self.logger.error("Error generating {} dialog.", exc_info=True)
            print("Exception {} occured with key {} and value {}".format(e, k, v))
            tb = traceback.format_exc()
            print(tb)

    @pyqtSlot(str)
    def update_version(self, directory):
        """
        Updates the question combobox based upon selected directory.

            # Arguments:
                directory (String): path to top of version directory.  If 'default', 
                load default values.  
        """
        print("Directory received by update_version", directory)
        self.is_dirty = False
        self.current_version = directory
        # Clear combobox to be reconstructed or blank if default.
        self.question_combobox.clear()
        if self.current_version.split('\\')[-1] == 'default':
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
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
                            # print("Filename in ModelDialog:", fname)
                            if fname == self.main_model_name + '.pkl' or fname == self.main_model_name + '.h5':
                                combo_text = combo_text + "*"
                                model_exists = True
                self.question_combobox.addItem(combo_text, d)

            self.comms.check_for_existing_model.emit(
                self.main_model_name, model_exists)

            self.form_grid.addWidget(self.question_combobox, 0, 0)
            self.update()
        except FileNotFoundError as fnfe:
            pass
        except Exception as e:
            self.logger.error(
                "Error loading updated version directories.", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)
        # self.question_combobox.show()

    def load_version_params(self, path):
        """
        Loads parameters from the selected version for a specific question.  
        Resets parameters to default prior to loading.
        If default or None is selected, returns after reload.  

            # Attributes
                path: String, path to version parameters.  
        """
        # Reset input parameters
        for model, types in self.model_params.items():
            for t, params in types.items():
                self.set_input_params(params)
        # If true, default (or none available) selected, thus Return
        if path == None or path == 'default':
            self.is_dirty = False
            return

        filename = self.main_model_name + '.json'
        model_data = {}
        try:
            try:

                with open(os.path.join(path, self.main_model_name, filename), 'r') as f:
                    model_data = json.load(f)
                model_class = model_data['model_class']
                for kind, params in model_data['params'].items():
                    self.set_input_params(params)
            except FileNotFoundError as fnfe:
                # No parameter file exists.  Pass.
                pass
            self.is_dirty = False
        except Exception as e:
            self.logger.error("Error updating parameters", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)

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
            if isinstance(v, dict):
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

    @pyqtSlot(bool)
    def check_for_default(self, force_reload=False):
        """
        Checks for the existance of a default value file.  If none found,
        one is created.
        """
        default_dir = os.path.join(
            ".\\package\\data\\default_models\\default", self.main_model_name)
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)

        default_path = os.path.join(
            default_dir, self.main_model_name + '.json')

        if not os.path.isfile(default_path) or force_reload:
            save_data = {
                "model_base": self.params[0]['model_base'],
                "model_module": self.params[0]['model_module'],
                "model_class": self.main_model_name,
                "question_number": "default",
                "version": "default",
                "meta" :{
                    "training_meta": {},
                    "tuning_meta" : {}
                },
                "params": {}
            }
            for model, types in self.model_params.items():
                print("check_for_defaults data:")
                print(f"{model}")
                print(types)
                for t, params in types.items():
                    # True if model spec has more than one category of parameters.  Only TF models at this point.
                    if not model in save_data['params'].keys():
                        save_data['params'][model] = {}
                    for param_name, data in params.items():
                        save_data['params'][model][param_name] = data['default']
            try:
                with open(default_path, 'w') as outfile:
                    json.dump(save_data, outfile, indent=2)
            except Exception as e:
                self.logger.error("Error saving updated model parameters for {}.".format(
                    self.main_model_name), exc_info=True)
                print("Exception {}".format(e))
                tb = traceback.format_exc()
                print(tb)


if __name__ == "__main__":
    import sys
    # Qt Application
    SVC = {
        "model_base": "sklearn",
        "model_module": "sklearn.svm",
        "model_class": "SVC",
        "SVC": {
            "Hyperparameters": {
                "C": {
                    "type": "double",
                    "default": 1.0,
                    "min": 0,
                    "max": 1000,
                    "step_size": 1,
                    "decimal_len": 1
                },
                "shrinking": {
                    "type": "dropdown",
                    "default": True,
                    "options": {
                        "True": True,
                        "False": False
                    }
                },
                "probability": {
                    "type": "dropdown",
                    "default": True,
                    "options": {
                        "True": True,
                        "False": False
                    }
                },
                "tol": {
                    "type": "double",
                    "default": 0.001,
                    "min": 0,
                    "max": 100,
                    "step_size": 0.001,
                    "decimal_len":  5
                },
                "cache_size": {
                    "type": "int",
                    "default": 200,
                    "min": 100,
                    "max": 100000,
                    "step_size": 100
                }
            }
        }
    }
    tfidf = {
        "model_base": "sklearn",
        "model_module": "sklearn.feature_extraction.text",
        "model_class": "TfidfVectorizer",
        "TfidfVectorizer": {
            "Hyperparameters": {
                "ngram_range": {
                    "type": "range",
                    "min": 1,
                    "max": 9,
                    "default": 1
                },
                "encoding": {
                    "type": "dropdown",
                    "default": "utf-8",
                    "options": {
                        "utf-8": "utf-8",
                        "latin-1": "latin-1"
                    }
                }
            }
        }
    }
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SkModelDialog(None, SVC, tfidf)
    window.show()
    sys.exit(app.exec_())
