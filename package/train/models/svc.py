import pandas as pd

from PySide2 import QtCore
from PySide2.QtWidgets import (QPushButton, QVBoxLayout, QFormLayout, QGroupBox, QWidget, 
                               QDialog, QDialogButtonBox, QComboBox, QDoubleSpinBox, QSizePolicy, QLabel)
import json
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""

class SVC(QDialog):
    
    def __init__(self, default_params={}, parent=None):
        super(SVC, self).__init__(parent)
        self.default_params = default_params
        print("Default params:")
        print(json.dumps(default_params, indent=2))
        # self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        self.buttonBox.setObjectName("svc_buttonbox")

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.main_layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        self.c_label = QLabel("C:")
        self.c_input = QDoubleSpinBox(objectName='clf__c')
        self.c_input.setDecimals(1)
        self.c_input.setValue(default_params['clf__C'])

        self.tol_label = QLabel("Tolerance:")
        self.tol_input = QDoubleSpinBox(objectName='clf__tol')
        self.tol_input.setDecimals(4)
        self.tol_input.setValue(default_params['clf__tol'])

        self.form_layout.addRow(self.c_label, self.c_input)
        self.form_layout.addRow(self.tol_label, self.tol_input)
        
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)

    def getCValue(self):
        print("Value from combobox: {}".format(self.c_input.value()))
        return self.c_input.value()

    def saveParams(self):
        if (self.exec_() == QDialog.Accepted):
            input_widgets = (self.form_layout.itemAt(i).widget() for i in range(self.form_layout.count()))
            for widget in input_widgets:
                if not isinstance(widget, QLabel):
                    print('"' + widget.objectName() + '": {},'.format(widget.value()))
        else:
            print("Denied!")
