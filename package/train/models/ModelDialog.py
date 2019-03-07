
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PySide2 import QtCore
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json
import importlib 
import traceback
import inspect

class ModelDialog(QDialog):
    """ModelDialog is the basic structure behind model dialogs in CATScore.
    # Arguments
        model_params: String, path to default parameters .json file.
    """
    def __init__(self, parent=None, model_params={}, tfidf_params={}):
        super(ModelDialog, self).__init__(parent)
        self.params = model_params
        if tfidf_params:
            self.tfidf_params = tfidf_params
            self.tfidf = self.getClass(self.tfidf_params)
        self.model = self.getClass(self.params)

        self.updated_model_params = {}
        self.updated_tfidf_params = {}
        # input_widgets is a list of all dynamically created input widgets for the parameters.
        self.input_widgets = {}

        self.setWindowTitle(self.params['model_class'])
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
        self.setupUI()

        self.model_groupbox.setLayout(self.model_param_form)
        self.tfidf_groupbox.setLayout(self.tfidf_param_form)

        self.form_grid.addWidget(self.model_groupbox, 0, 0)
        self.form_grid.addWidget(self.tfidf_groupbox, 0, 1)
        self.main_layout.addLayout(self.form_grid)

        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)
        
    def getClass(self, params):
        """Return instantiated class using importlib
        """
        module = importlib.import_module(params['model_module'])
        module_class = getattr(module, params['model_class'])
        if "static_params" in params:
            model = module_class(**params['static_params'])
        else:
            model = module_class()
        return model


    def saveParams(self):
        """Saves the parameters entered by the user.  
        """
        if (self.exec_() == QDialog.Accepted):
            print("Updated Params as they hit saveParams:")
            print(json.dumps(self.updated_params, indent=2))

        else:
            print("Denied!")

    def setupUI(self):
        model_params = self.model.get_params()
        for k, v in model_params.items():
            print(k, v)
            try:
                # label_string = k
                label_string = k 
                label = QLabel(label_string)
                if (v is True or v is False):
                    input_field = QComboBox(objectName=k)
                    input_field.addItem('True', True)
                    input_field.addItem('False', False)
                    if v:
                        input_field.setCurrentIndex(0)
                    else:
                        input_field.setCurrentIndex(1)
                    input_field.currentIndexChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'model',
                            x, 
                            y.currentData())
                    )

                elif isinstance(v, float):
                    input_field = QDoubleSpinBox(objectName=k)
                    input_field.setDecimals(len(str(v)) - 2)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'model',
                            x, 
                            y.value())
                    )

                elif isinstance(v, int):
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(0, 10000)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'model',
                            x, 
                            y.value())
                    )

                else:
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(v)
                    # lambda for connecting signals.  Param two will pass None instead of an empty
                    # string if no response given by user.
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'model',
                                x, 
                                (None if y.text() == '' else y.text())
                            )
                    )
                   
                self.model_param_form.addRow(label, input_field)
                self.input_widgets[k] = input_field
            except Exception as e:
                print("Exception {} occured with key {} and value {}".format(e, k, v))
                tb = traceback.format_exc()
                print(tb)
        for k, v in self.tfidf.get_params().items():
            try:
                label_string = k
                label = QLabel(label_string)
                if (v is True or v is False):
                    input_field = QComboBox(objectName=k)
                    input_field.addItem('True', True)
                    input_field.addItem('False', False)
                    if v:
                        input_field.setCurrentIndex(0)
                    else:
                        input_field.setCurrentIndex(1)
                    input_field.currentIndexChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'tfidf',
                            x, 
                            y.currentData())
                    )

                elif isinstance(v, float):
                    label_string = k
                    label = QLabel(label_string)
                    input_field = QDoubleSpinBox(objectName=k)
                    
                    input_field.setDecimals(len(str(v)) - 2)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'tfidf',
                            x, 
                            y.value())
                    )

                elif isinstance(v, int):
                    label_string = k
                    label = QLabel(label_string)
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(0, 10000)
                    input_field.setValue(v)
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'tfidf',
                            x, 
                            y.value())
                    )
                  
                elif isinstance(v, tuple):
                    ##FIXME: create validator to only allow certain input?
                    label_string = k
                    label = QLabel(label_string)
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(",".join(str(x) for x in v))
                    input_field.setMaxLength(3)
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'tfidf',
                                x, 
                                [] if y.text() == '' else list(map(int, y.text().split(',')))
                            )
                    )
                elif inspect.isclass(v) or isinstance(v, type):
                    continue
                else:
                    label_string = k
                    label = QLabel(label_string)
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(v)
                    # lambda for connecting signals.  Param two will pass None instead of an empty
                    # string if no response given by user.
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'tfidf',
                                x, 
                                (None if y.text() == '' else y.text())
                            )
                    )
                self.tfidf_param_form.addRow(label, input_field)
                self.input_widgets[k] = input_field
            except Exception as e:
                print("Exception {} occured with key {} and value {}".format(e, k, v))

    def _splitKey(self, key):
        return key.split('__')[1]

    def _updateParam(self, param_type, key, value):
        print("updateParams key, value: {}, {}, {}".format(param_type, key, value))
        class_key = '__' + key + '__'
        if param_type == 'model':
            self.updated_model_params[class_key] = value
        else:
            self.updated_tfidf_params[class_key] = value

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
    window = ModelDialog(model_params=SVC, tfidf_params=tfidf)
    window.show()
    sys.exit(app.exec_())
