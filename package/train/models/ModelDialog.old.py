
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QFormLayout, QGroupBox, QWidget, 
                               QDialog, QDialogButtonBox, QComboBox, QDoubleSpinBox, QSizePolicy, QLabel)
import json

class ModelDialog(QDialog):
    """ModelDialog is the basic structure behind model dialogs in CATScore.
    # Arguments
        param_path: String, path to default parameters .json file.
    """
    def __init__(self, param_path={}, parent=None):
        super(ModelDialog, self).__init__(parent)
        self.default_params = self._getParams(param_path)
        print("Default params:")
        print(json.dumps(self.default_params, indent=2))
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("model_buttonbox")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.model_groupbox = QGroupBox()
        self.model_groupbox.setTitle("Model hyperparameters")
        self.tfidf_groupbox = QGroupBox()
        self.tfidf_groupbox.setTitle('TF-IDF hyperparameters')

        self.main_layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        
    def saveParams(self):
        """Saves the parameters entered by the user.  
        """
        print("Default params after save:")
        print(json.dumps(self.default_params, indent=2))
        if (self.exec_() == QDialog.Accepted):
            input_widgets = (self.form_layout.itemAt(i).widget() for i in range(self.form_layout.count()))
            for widget in input_widgets:
                if isinstance(widget, QDoubleSpinBox):
                    print('"' + widget.objectName() + '": {},'.format(widget.value()))
                if isinstance(widget, QComboBox):
                    print('"' + widget.objectName() + '": {},'.format(widget.currentData()))

        else:
            print("Denied!")

    def _getParams(self, path):
        print("Path: {}".format(path))
        svc_params = {
                        "clf__C": 1,
                        "clf__class_weight": None,
                        "clf__kernel": "linear",
                        "clf__tol": 0.001,
                        "clf__cache_size": 200,
                        "tfidf__max_df": 0.5,
                        "tfidf__min_df": 2,
                        "tfidf__ngram_range": [
                            1,
                            2
                        ],
                        "tfidf__stop_words": None,
                        "tfidf__strip_accents": "unicode",
                        "tfidf__use_idf": False
                    }
        return svc_params

