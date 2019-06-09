
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json
import re
import importlib 
import traceback
import inspect
import logging
import os

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
            name = param['model_class']
            self.model_params[name] = param[name]
            self.updated_params[name] = {}

        self.setWindowTitle(self.main_model_name)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("model_buttonbox")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.rejected.connect(self.reject)

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
                groupbox.setTitle(model + " " + t)
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
    def getModelName(self):
        return self.main_model_name


    def getClass(self, params, init_class=None):
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


    def save_params(self):
        """
        Saves the model parameters entered by the user. If default version is selected,
        return without saving.
        """
        if (self.exec_() == QDialog.Accepted):
            print("Updated Params as they hit save_params:")
            print(json.dumps(self.updated_params, indent=2))
            version = self.current_version.split('\\')[-1]
            if version == 'default':
                print("Default version selected.  Returning...")
                return
            filename = self.main_model_name + '.json'
            save_dir = os.path.join(self.question_combobox.currentData(),
                                    filename)
            save_data = {
                "model_base" : self.params[0]['model_base'],
                "model_module": self.params[0]['model_module'],
                "model_class" : self.main_model_name,
                "question_number" : self.question_combobox.currentData().split('\\')[-1],
                "version" : version,
                "params" : {}
            }
            for param_type, params in self.updated_params.items():
                save_data['params'][param_type] = params
            try:
                with open(save_dir, 'w') as outfile:
                    json.dump(save_data, outfile, indent=2)
            except Exception as e:
                self.logger.error("Error saving updated model parameters for {}.".format(self.main_model_name), exc_info=True)
                print("Exception {}".format(e))
                tb = traceback.format_exc()
                print(tb)
        else:
            pass


    def _split_key(self, key):
        return key.split('__')[1]


    def _update_param(self, param_type, key, value):
        print("updateParams key, value: {}, {}, {}".format(param_type, key, value))
        #class_key = '__' + key + '__'
        if self.current_version != 'default':
            self.updated_params[param_type][key] = value


    def setupUI(self, param_type, param_dict, form):
        """Build UI elements using parameters dict of scikit models

            # Attributes:
                param_type: String, type of param to update
                param_dict: dict, dictionary of parameter/default values from model.
                default_params: dict, dictionary of default parameters defined by me.
        """
        try:
            for k, v in param_dict.items():
                label_string = k 
                label = QLabel(label_string)
                type = v['type']
                if type == 'dropdown':
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
                    form.addRow(label, input_field)
                    self.input_widgets[k] = input_field

                elif type == 'double':
                    input_field = QDoubleSpinBox(objectName=k)
                    input_field.setDecimals(v['decimal_len'])
                    input_field.setRange(v['min'], v['max'])
                    input_field.setValue(v['default'])
                    input_field.setSingleStep(v['step_size'])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x, 
                            y.value())
                    )

                elif type == 'int':
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(v['min'], v['max'])
                    input_field.setValue(v['default'])
                    input_field.setSingleStep(v['step_size'])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x, 
                            y.value())
                    )
                elif type == 'range':
                    label_string = k
                    label = QLabel(label_string + ' : 1,')
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(v['min'], v['max'])
                    input_field.setValue(v['default'])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._update_param(
                                param_type,
                                x, 
                                y.value())
                            )
                form.addRow(label, input_field)
                self.input_widgets[k] = input_field
        except Exception as e:
            self.logger.error("Error generating {} dialog.", exc_info=True)
            print("Exception {} occured with key {} and value {}".format(e, k, v))
            tb = traceback.format_exc()
            print(tb)

    @Slot(str)
    def update_version(self, directory):
        """
        Updates the question combobox based upon selected directory.

            # Arguments:
                directory (String): path to top of version directory.  If 'default', 
                load default values.  
        """
        print("Directory received by update_version", directory)
        self.current_version = directory
        # Clear combobox to be reconstructed or blank if default.
        self.question_combobox.clear()
        if self.current_version == 'default':
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        else:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        try:
            question_directories = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
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
            for d in question_directories:
                self.question_combobox.addItem(d.split('\\')[-1], d)
            self.form_grid.addWidget(self.question_combobox, 0, 0)
            self.update()
        except FileNotFoundError as fnfe:
            pass
        except Exception as e:
            self.logger.error("Error loading updated version directories.", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)
        # self.question_combobox.show()

    def load_version_params(self, path):
        """Loads parameters from the selected version for a specific question.  If default is selected,
            reloads base model parameters.  

            # Attributes
                path: String, path to version parameters.  If None, reload default values.
        """
        # Reset input parameters
        for model, types in self.model_params.items():
            for t, params in types.items():
                self.set_input_params(params)
        # If true, default (or none available) selected, thus Return
        if path == None or path == 'default':
            return

        filename = self.main_model_name + '.json'
        model_data = {}
        try:
            try:
                with open(os.path.join(path, filename), 'r') as f:
                    model_data = json.load(f)
                model_class = model_data['model_class']
                for kind, params in model_data['params'].items():
                    self.set_input_params(params)
            except FileNotFoundError as fnfe:
                pass

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
        for k,v in param_dict.items():
            # If v is dictionary, function was called using default values.
            # Set v equal to the default value of that parameter.
            if isinstance(v, dict):
                v = v['default']
            if k in self.input_widgets:
                cla = self.input_widgets[k]
                if isinstance(cla, QComboBox):
                    idx = cla.findData(v)
                    if idx != -1:
                        cla.setCurrentIndex(idx)
                else:
                    cla.setValue(v)


if __name__ == "__main__":
    import sys
    # Qt Application
    SVC = {                  
    "model_base": "sklearn",
    "model_module": "sklearn.svm",
    "model_class" : "SVC",
    "SVC" : {
        "Hyperparameters" : {
            "C" : {
                "type" : "double",
                "default" : 1.0,
                "min" : 0,
                "max" : 1000,
                "step_size" : 1,
                "decimal_len" : 1
            },
            "shrinking" : {
                "type" : "dropdown",
                "default" : True,
                "options" : {
                    "True" : True,
                    "False" : False
                }
            },
            "probability" : {
                "type" : "dropdown",
                "default" : True,
                "options" : {
                    "True" : True,
                    "False" : False
                }
            },
            "tol" : {
                "type" : "double",
                "default" : 0.001,
                "min" : 0,
                "max" : 100,
                "step_size" : 0.001,
                "decimal_len":  5
            },
            "cache_size" : {
                "type" : "int",
                "default" : 200,
                "min" : 100,
                "max" : 100000,
                "step_size" : 100
            }
        }
    }
    }
    tfidf = {
        "model_base" : "sklearn",
        "model_module" : "sklearn.feature_extraction.text",
        "model_class" : "TfidfVectorizer",
        "TfidfVectorizer" : {
            "Hyperparameters" : {
                "ngram_range" : {
                    "type" : "range",
                    "min" : 1,
                    "max" : 9,
                    "default" : 1
                },
                "encoding" : {
                    "type" : "dropdown",
                    "default" : "utf-8",
                    "options" : {
                        "utf-8" : "utf-8",
                        "latin-1" : "latin-1"
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
