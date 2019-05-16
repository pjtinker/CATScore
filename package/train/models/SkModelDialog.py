
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json
import importlib 
import traceback
import inspect
import logging
import os

class SkModelDialog(QDialog):
    """SkModelDialog is the basic structure behind model dialogs in CATScore.

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
        self.ui_widgets = {}
        for param in params:
            name = param['model_class']
            self.model_params[name] = param[name]
            self.updated_params[name] = {}
            groupbox = QGroupBox()
            
        # print("Params: ")
        # print(json.dumps(self.model_params, indent=2))
        # return
        # self.model_data = model_data
        # self.tfidf_data = tfidf_data
        # self.model_params = self.model_data[self.model_data['model_class']]
        # self.model = self.getClass(self.model_params)

        # TODO: Add static params to model_params
        # self.updated_params = {}
        # self.updated_params[self.model_data['model_class']] = {}
        # if self.tfidf_data:
        #     self.tfidf_params = self.tfidf_data[self.tfidf_data['model_class']]
        #     # self.tfidf = self.getClass(self.tfidf_params)
        #     self.updated_params[self.tfidf_data['model_class']] = {}

        # input_widgets is a list of all dynamically created input widgets for the various model params.
        # Holds EVERY input widget, regardless of type.  Key = hyperparameter name
        self.input_widgets = {}

        self.setWindowTitle(self.model_data['model_class'])
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("model_buttonbox")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.model_groupbox = QGroupBox()
        self.model_groupbox.setTitle("Model hyperparameters")
        self.tfidf_groupbox = QGroupBox()
        self.tfidf_groupbox.setTitle('TF-IDF hyperparameters')

        self.main_layout = QVBoxLayout()
        self.form_grid = QGridLayout()
        self.model_param_form = QFormLayout()
        self.tfidf_param_form = QFormLayout()
        

        self.question_combobox = QComboBox()
        self.question_combobox.currentIndexChanged.connect(
            lambda state, y=self.question_combobox: self.load_version_params(
                y.currentData())
        )
        self.form_grid.addWidget(self.question_combobox, 0, 0)
        self.model_groupbox.setLayout(self.model_param_form)
        self.tfidf_groupbox.setLayout(self.tfidf_param_form)

        self.form_grid.addWidget(self.model_groupbox, 1, 0)
        self.form_grid.addWidget(self.tfidf_groupbox, 1, 1)
        self.setupUI(self.model_data['model_class'], self.model_params, self.model_param_form)
        self.setupUI(self.tfidf_data['model_class'], self.tfidf_params, self.tfidf_param_form)
        
        self.main_layout.addLayout(self.form_grid)
        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)
        self.move(20, 10)


    def getModelName(self):
        return self.model_params['model_class']


    def getClass(self, params, init_class=None):
        """Return instantiated class using importlib
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


    def saveParams(self):
        """Saves the parameters entered by the user.  
        FIXME:
        """
        if (self.exec_() == QDialog.Accepted):
            print("Updated Params as they hit saveParams:")
            print(json.dumps(self.updated_params, indent=2))
        else:
            print("Denied!")


    def _split_key(self, key):
        return key.split('__')[1]


    def _update_param(self, param_type, key, value):
        print("updateParams key, value: {}, {}, {}".format(param_type, key, value))
        #class_key = '__' + key + '__'
        self.updated_params[param_type][key] = value


    def setupUI(self, param_type, param_dict, form):
        """Build UI elements using parameters dict of scikit models

            # Attributes:
                param_type: String, type of param to update
                param_dict: dict, dictionary of parameter/default values from model.
                default_params: dict, dictionary of default parameters defined by me.
        """
        # if "restricted_params" in default_params:
        #     for k in default_params['restricted_params']:
        #         param_dict.pop(k, None)
        try:
            for k, v in param_dict['params'].items():
                label_string = k 
                label = QLabel(label_string)
                type = v['type']
                if type == 'dropdown':
                    input_field = QComboBox(objectName=k)
                    for name, value in v['options'].items():
                        input_field.addItem(name, value)
                    input_field.setCurrentIndex(v['default'])
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
                # else:
                #     input_field = QLineEdit(objectName=k)
                #     input_field.setText(v)
                #     # lambda for connecting signals.  Param three will pass None instead of an empty
                #     # string if no response given by user.
                #     input_field.textChanged.connect(
                #         lambda state, x=k, y=input_field:
                #             self._update_param(
                #                 param_type,
                #                 x, 
                #                 (None if y.text() == '' else y.text())
                #             )
                #     )
                elif type == 'range':
                    # FIXME: create validator to only allow certain input?
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
                # elif inspect.isclass(v) or isinstance(v, type):
                #     continue

                form.addRow(label, input_field)
                self.input_widgets[k] = input_field
        except Exception as e:
            self.logger.error("Error generating {} dialog.", exc_info=True)
            print("Exception {} occured with key {} and value {}".format(e, k, v))
            tb = traceback.format_exc()
            print(tb)

    @Slot(str)
    def update_version(self, directory):
        print("Directory received by update_version", directory)
        self.question_combobox.clear()
        question_directories = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
        for d in question_directories:
            self.question_combobox.addItem(d.split('\\')[-1], d)
        self.form_grid.addWidget(self.question_combobox, 0, 0)
        # self.question_combobox.show()

    def load_version_params(self, path):
        """Loads parameters from the selected version for a specific question.  If default is selected,
            reloads base model parameters.  

            # Attributes
                path: String, path to version parameters.  If None, reload default values.
        """
        if path == None:
            return
        # Reset input parameters
        self.set_input_params(self.model_data[self.model_data['model_class']]['params'])
        self.set_input_params(self.tfidf_data[self.tfidf_data['model_class']]['params'])
        # self.set_input_params(self.fs_params)

        filename = self.model_data['model_class'] + '.json'
        model_data = {}
        try:
            try:
                with open(os.path.join(path, filename), 'r') as f:
                    model_data = json.load(f)
                model_class = model_data['model_class']
                if 'model_params' in model_data:
                    self.set_input_params(model_data['model_params'])

                if "tfidf_params" in model_data:
                    self.set_input_params(model_data['tfidf_params'])
            except FileNotFoundError as fnfe:
                pass
                # model_data['model_class'] = self.tfidf_data['model_class']
                # model_data['model_params'] = self.model.get_params()
                # model_data['tfidf_params'] = self.tfidf.get_params()
                # model_data['fs_params'] = self.fs_params

        except Exception as e:
            self.logger.error("Error updating parameters", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)

    def set_input_params(self, param_dict):
        for k,v in param_dict.items():
            # If v is dictionary, function was called using default values.
            # Set v equal to the default value of that parameter.
            if isinstance(v, dict):
                v = v['default']
            if k in self.input_widgets:
                cla = self.input_widgets[k]
                if isinstance(cla, QComboBox):
                    cla.setCurrentIndex(v)
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
        "static_params" : {
            "kernel" : "linear"
        },
        "restricted_params" : [
            "decision_function_shape", "kernel", "degree", "gamma", "coef0"
        ],
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
                "default" : 1,
                "options" : {
                    "True" : True,
                    "False" : False
                }
            },
            "probability" : {
                "type" : "dropdown",
                "default" : 0,
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
            "restricted_params" : [
                "preprocessor", "binary", "dtype", 
                "analyzer", "tokenizer", "stop_words",
                "lowercase", "input", "vocabulary", "token_pattern"
            ],
            "Hyperparameters" : {
                "ngram_range" : {
                    "type" : "range",
                    "min" : 1,
                    "max" : 9,
                    "default" : 1
                },
                "encoding" : {
                    "type" : "dropdown",
                    "default" : 0,
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
    window = SkModelDialog(model_data=SVC, tfidf_data=tfidf)
    window.show()
    sys.exit(app.exec_())
