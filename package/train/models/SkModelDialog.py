
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
                 model_params={}, 
                 tfidf_params={},
                *custom_params):
        super(SkModelDialog, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model_params = model_params
        self.model = self.getClass(self.model_params)

        self.updated_params = {}
        self.updated_params['model_params'] = {}
        if tfidf_params:
            self.tfidf_params = tfidf_params
            self.tfidf = self.getClass(self.tfidf_params)
            self.updated_params['tfidf_params'] = {}

        # input_widgets is a list of all dynamically created input widgets for the various model params.
        # Holds EVERY input widget, regardless of type.  Key = hyperparameter name
        self.input_widgets = {}

        self.setWindowTitle(self.model_params['model_class'])
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
        self.setupUI('model_params', self.model.get_params(), self.model_params, self.model_param_form)
        self.setupUI('tfidf_params', self.tfidf.get_params(), self.tfidf_params, self.tfidf_param_form)
        
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


    def setupUI(self, param_type, param_dict, default_params, form):
        """Build UI elements using parameters dict of scikit models

            # Attributes:
                param_type: String, type of param to update
                param_dict: dict, dictionary of parameter/default values from model.
                default_params: dict, dictionary of default parameters defined by me.
        """
        if "restricted_params" in default_params:
            for k in default_params['restricted_params']:
                param_dict.pop(k, None)
        for k, v in param_dict.items():
            try:
                label_string = k 
                label = QLabel(label_string)
                if isinstance(v, bool):
                    input_field = QComboBox(objectName=k)
                    input_field.addItem('True', True)
                    input_field.addItem('False', False)
                    if v:
                        input_field.setCurrentIndex(0)
                    else:
                        input_field.setCurrentIndex(1)
                    input_field.currentIndexChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x, 
                            y.currentData())
                    )

                elif isinstance(v, float):
                    input_field = QDoubleSpinBox(objectName=k)
                    input_field.setDecimals(len(str(v)) - 2)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x, 
                            y.value())
                    )

                elif isinstance(v, int):
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(0, 10000)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._update_param(
                            param_type,
                            x, 
                            y.value())
                    )
                elif isinstance(v, tuple):
                    # FIXME: create validator to only allow certain input?
                    label_string = k
                    label = QLabel(label_string)
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(",".join(str(x) for x in v))
                    input_field.setMaxLength(3)
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._update_param(
                                'tfidf',
                                x, 
                                [] if y.text() == '' else list(map(int, y.text().split(',')))
                            )
                    )
                elif inspect.isclass(v) or isinstance(v, type):
                    continue
                else:
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(v)
                    # lambda for connecting signals.  Param three will pass None instead of an empty
                    # string if no response given by user.
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._update_param(
                                param_type,
                                x, 
                                (None if y.text() == '' else y.text())
                            )
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
        self.set_input_params(self.model.get_params())
        self.set_input_params(self.tfidf.get_params())
        # self.set_input_params(self.fs_params)

        filename = self.getModelName() + '.json'
        model_data = {}
        try:
            try:
                with open(os.path.join(path, filename), 'r') as f:
                    model_data = json.load(f)
            except FileNotFoundError as fnfe:
                model_data['model_class'] = self.getModelName()
                model_data['model_params'] = self.model.get_params()
                model_data['tfidf_params'] = self.tfidf.get_params()
                # model_data['fs_params'] = self.fs_params

            model_class = model_data['model_class']
            if 'model_params' in model_data:
                self.set_input_params(model_data['model_params'])

            if "tfidf_params" in model_data:
                self.set_input_params(model_data['tfidf_params'])

        except Exception as e:
            self.logger.error("Error updating parameters", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)

    def set_input_params(self, param_dict):
        for k,v in param_dict.items():
            if k in self.input_widgets:
                cla = self.input_widgets[k]
                if isinstance(cla, QComboBox):
                    idx = cla.findData(v)
                    if idx != -1:
                        cla.setCurrentIndex(idx)
                elif isinstance(cla, QLineEdit):
                    if isinstance(v, tuple) or isinstance(v, list):
                        cla.setText(",".join(str(x) for x in v))
                    else:
                        cla.setText(v)
                else:
                    cla.setValue(v)


if __name__ == "__main__":
    import sys
    # Qt Application
    SVC = {                  
        "model_base": "sklearn",
        "model_module": "sklearn.svm",
        "model_class" : "SVC",
        "static_params" : {
            "kernel" : "linear"
        }
    }
    tfidf = {
        "model_base" : "sklearn",
        "model_module" : "sklearn.feature_extraction.text",
        "model_class" : "TfidfVectorizer"
    }
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SkModelDialog(model_params=SVC, tfidf_params=tfidf)
    window.show()
    sys.exit(app.exec_())
