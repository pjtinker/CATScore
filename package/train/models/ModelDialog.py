
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PySide2 import QtCore
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json

class ModelDialog(QDialog):
    """ModelDialog is the basic structure behind model dialogs in CATScore.
    # Arguments
        model_params: String, path to default parameters .json file.
    """
    def __init__(self, model_params={}, parent=None):
        super(ModelDialog, self).__init__(parent)
        self.params = model_params
        print(model_params)
        # input_widgets is a list of all dynamically created input widgets for the parameters.
        self.input_widgets = {}
        self.setWindowTitle(self.params.model_name)
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
        
    def saveParams(self):
        """Saves the parameters entered by the user.  
        """
        if (self.exec_() == QDialog.Accepted):
            print("Params as they hit saveParams:")
            print(json.dumps(self.params, indent=2))

        else:
            print("Denied!")

    def setupUI(self):
        
        for k, v in self.params.model_params.items():
            try:
                label_string = self._splitKey(k)
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
                            'model_params',
                            x, 
                            y.currentData())
                    )

                elif isinstance(v, float):
                    input_field = QDoubleSpinBox(objectName=k)
                    input_field.setDecimals(len(str(v)) - 2)
                    input_field.setValue(self.params.model_params[k])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'model_params',
                            x, 
                            y.value())
                    )

                elif isinstance(v, int):
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(0, 10000)
                    input_field.setValue(self.params.model_params[k])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'model_params',
                            x, 
                            y.value())
                    )

                else:
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(self.params.model_params[k])
                    # lambda for connecting signals.  Param two will pass None instead of an empty
                    # string if no response given by user.
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'model_params',
                                x, 
                                (None if y.text() == '' else y.text())
                            )
                    )
                   
                self.model_param_form.addRow(label, input_field)
                self.input_widgets[k] = input_field
            except Exception as e:
                print("Exception {} occured with key {} and value {}".format(e, k, v))
        for k, v in self.params.tfidf_params.items():
            try:
                label_string = self._splitKey(k)
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
                            'tfidf_params',
                            x, 
                            y.currentData())
                    )

                elif isinstance(v, float):
                    label_string = self._splitKey(k)
                    label = QLabel(label_string)
                    input_field = QDoubleSpinBox(objectName=k)
                    
                    input_field.setDecimals(len(str(v)) - 2)
                    input_field.setValue(self.params.tfidf_params[k])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'tfidf_params',
                            x, 
                            y.value())
                    )

                elif isinstance(v, int):
                    label_string = self._splitKey(k)
                    label = QLabel(label_string)
                    input_field = QSpinBox(objectName=k)
                    input_field.setRange(0, 10000)
                    input_field.setValue(self.params.tfidf_params[k])
                    input_field.valueChanged.connect(
                        lambda state, x=k, y=input_field: self._updateParam(
                            'tfidf_params',
                            x, 
                            y.value())
                    )
                  
                elif isinstance(v, list):
                    ##FIXME: create validator to only allow certain input?
                    label_string = self._splitKey(k)
                    label = QLabel(label_string)
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(",".join(str(x) for x in self.params.tfidf_params[k]))
                    input_field.setMaxLength(3)
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'tfidf_params',
                                x, 
                                [] if y.text() == '' else list(map(int, y.text().split(',')))
                            )
                    )
                else:
                    label_string = self._splitKey(k)
                    label = QLabel(label_string)
                    input_field = QLineEdit(objectName=k)
                    input_field.setText(self.params.tfidf_params[k])
                    # lambda for connecting signals.  Param two will pass None instead of an empty
                    # string if no response given by user.
                    input_field.textChanged.connect(
                        lambda state, x=k, y=input_field:
                            self._updateParam(
                                'tfidf_params',
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
        print("updateParams key, value: {}, {}".format(key, value))
        self.params[param_type][key] = value

if __name__ == "__main__":
    import sys
    # Qt Application
    SVC = {               
                "model_name": "SVC",
                "model_params": {

                    "clf__verbose": False,
                    "clf__shrinking": True,
                    "clf__coef0": 0.0,
                    "clf__degree": 3,
                    "clf__random_state": None,
                    "clf__class_weight": None,
                    "clf__tol": 0.001,
                    "clf__kernel": "linear",
                    "clf__gamma": "auto",
                    "clf__cache_size": 200,
                    "clf__max_iter": -1,
                    "clf__probability": False,
                    "clf__C": 1.0
                },
                "tfidf_params": {
                        "tfidf__strip_accents": None,
                        "tfidf__ngram_range": [1, 1],
                        "tfidf__max_df": 1.0,
                        "tfidf__max_features": None,
                        "tfidf__input": "content",
                        "tfidf__min_df": 1,
                        "tfidf__vocabulary": None,
                        "tfidf__token_pattern": "(?u)\\b\\w\\w+\\b",
                        "tfidf__encoding": "utf-8",
                        "tfidf__preprocessor": None,
                        "tfidf__lowercase": True,
                        "tfidf__analyzer": "word",
                        "tfidf__smooth_idf": True,
                        "tfidf__decode_error": "strict",
                        "tfidf__stop_words": None,
                        "tfidf__sublinear_tf": False,
                        "tfidf__use_idf": True,
                        "tfidf__tokenizer": None,
                        "tfidf__norm": "l2",
                        "tfidf__binary": False
        },
        "tensorflow" : {

        }
    }
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ModelDialog()
    window.show()
    sys.exit(app.exec_())
