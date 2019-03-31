
"""QDialog for model parameters for sklearn's Support Vector Classifier
"""
from PySide2.QtCore import Slot
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json
import importlib 
import traceback
import inspect
import logging

class TfModelDialog(QDialog):
    """TfModelDialog is the basic structure behind model dialogs in CATScore.
    # Arguments
        model_params: String, path to default parameters .json file.
    """
    def __init__(self, parent=None, params={}):
        super(TfModelDialog, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model_params = params['model_params']
        self.optimizer_params = params['optimizer_params']
        self.model_class = params['model_class']
        self.updated_model_params = {}
        # input_widgets is a list of all dynamically created input widgets for the various model params.
        self.input_widgets = {}

        self.setWindowTitle(params['model_class'])
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("tf_model_buttonbox")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.model_groupbox = QGroupBox()
        self.model_groupbox.setTitle("Model hyperparameters")

        self.optimizer_groupbox = QGroupBox()
        self.optimizer_groupbox.setTitle("Optimizer hyperparameters")

        self.main_layout = QVBoxLayout()
        self.outer_hbox = QHBoxLayout()
        self.model_param_form = QFormLayout()
        self.optimizer_param_form = QFormLayout()

        self.setupModelUI()
        self.setupOptimizerUI()

        self.model_groupbox.setLayout(self.model_param_form)
        self.optimizer_groupbox.setLayout(self.optimizer_param_form)

        self.outer_hbox.addWidget(self.model_groupbox)
        self.outer_hbox.addStretch()
        self.outer_hbox.addWidget(self.optimizer_groupbox)

        self.main_layout.addLayout(self.outer_hbox)
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)
        self.move(20, 10)
        
    def getModelName(self):
        return self.model_params['model_class']

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
            print(json.dumps(self.updated_model_params, indent=2))

        else:
            print("Denied!")

    def setupModelUI(self):
        self.setupUI(self.model_params, self.model_param_form)

    def setupOptimizerUI(self):
        self.setupUI(self.optimizer_params, self.optimizer_param_form)
    
    def setupMetricsUI(self):
        pass

    def setupUI(self, params, form):
        for k, v in params.items():
            try:
                label_string = k 
                label = QLabel(' '.join(label_string.split('_')))
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
                   
                form.addRow(label, input_field)
                self.input_widgets[k] = input_field
            except Exception as e:
                self.logger.error("Error generating Tensorflow Dialog", exc_info=True)
                print("Exception {} occured with key {} and value {}".format(e, k, v))
                tb = traceback.format_exc()
                print(tb)

    def _splitKey(self, key):
        return key.split('__')[1]

    def _updateParam(self, param_type, key, value):
        print("updateParams key, value: {}, {}, {}".format(param_type, key, value))
        class_key = '__' + key + '__'
        if param_type == 'model':
            self.updated_model_params[class_key] = value

    @Slot(str)
    def update_version(self, directory):
        print("update_version in {} called with {}".format(self.model_class, directory))


if __name__ == "__main__":
    import sys
    # Qt Application
    SeqCNN = {
        "model_base" : "tensorflow",
        "model_class" : "Seperable CNN",
        "embedding_data_dir" : ".\\package\\data\\embeddings\\glove6b\\",
        "model_params" : {
            "epochs" : 1000,
            "batch_size" : 128,
            "blocks" : 2,
            "filters" : 64,
            "dropout_rate" : 0.2,
            "kernel_size" : 3,
            "pool_size" : 3
        },
        "optimizer_params" : {
            "optimizer_type" : "Adam",
            "lr" : 1e-3,
            "beta_1" : 0.9,
            "beta_2" : 0.999,
            "epsilon" : None,
            "decay" : 0.0,
            "amsgrad" : False
        },
        "metrics" : ["acc"]
        }

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TfModelDialog(params=SeqCNN)
    window.show()
    sys.exit(app.exec_())
