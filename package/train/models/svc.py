import pandas as pd

from PySide2 import QtCore
from PySide2.QtWidgets import (QPushButton, QApplication, QVBoxLayout, QFormLayout, QGroupBox, QWidget, 
                               QDialog, QDialogButtonBox, QComboBox, QDoubleSpinBox, QSizePolicy, QLabel)
import json
# from package.train.models.ModelDialog import ModelDialog
from ModelDialog import ModelDialog
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""

class SVC(ModelDialog):
    
    def __init__(self, param_path={}, parent=None):
        super(SVC, self).__init__(param_path, parent)


        self.c_label = QLabel("C:")
        self.c_input = QDoubleSpinBox(objectName='clf__c')
        self.c_input.setDecimals(1)
        self.c_input.setValue(self.default_params['clf__C'])

        self.tol_label = QLabel("Tolerance:")
        self.tol_input = QDoubleSpinBox(objectName='clf__tol')
        self.tol_input.setDecimals(4)
        self.tol_input.setValue(self.default_params['clf__tol'])

        self.class_weight_label = QLabel("Cache weight:")
        self.class_weight_combo = QComboBox(objectName='clf__class_weight')
        self.class_weight_combo.addItem('None', self.default_params['clf__class_weight'])
        self.class_weight_combo.addItem('Balanced', 'balanced')

        self.form_layout.addRow(self.c_label, self.c_input)
        self.form_layout.addRow(self.tol_label, self.tol_input)
        self.form_layout.addRow(self.class_weight_label, self.class_weight_combo)
        self.model_groupbox.setLayout(self.form_layout)
        self.main_layout.addWidget(self.model_groupbox)
        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)

if __name__ == "__main__":
    import sys
    # Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SVC()
    window.show()
    sys.exit(app.exec_())