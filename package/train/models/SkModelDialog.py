
'''QDialog for file defined models.
'''
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

from package.utils.catutils import CATEncoder
from package.utils.catutils import cat_decoder
from package.train.models.BaseModelDialog import BaseModelDialog
from package.utils.config import CONFIG



class Communicate(QObject):
    check_for_existing_model = pyqtSignal(str, bool)


class SkModelDialog(BaseModelDialog):
    '''
    SkModelDialog is the basic structure behind model dialogs in CATScore.

    # Arguments
        model_params: String, path to default parameters .json file.
        tfidf_params: String, path to default TF-IDF param file.
        fs_params: String, path to default feature selection file.
    '''
    # ! TODO: Update to use CONFIG values
    def __init__(self,
                 parent=None,
                 *params):
        super(BaseModelDialog, self).__init__(parent)
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
        self.buttonBox.setObjectName('model_buttonbox')
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.rejected.connect(self.reject)
        
        self.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(
            lambda: self.update_version(self.current_version))
        
        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(
            lambda: self.apply_changes())
        self.main_layout = QVBoxLayout()
        self.form_grid = QGridLayout()
        self.version_item_combobox = QComboBox()
        self.version_item_combobox.currentIndexChanged.connect(
            lambda state, y=self.version_item_combobox: self.load_version_params(
                y.currentData())
        )
        self.form_grid.addWidget(self.version_item_combobox, 0, 0)
        self.performance_hbox = QHBoxLayout()
        self.training_meta_form = QFormLayout()
        self.tuning_meta_form = QFormLayout()
        self.performance_hbox.addLayout(self.training_meta_form)
        self.performance_hbox.addLayout(self.tuning_meta_form)
        self.performance_meta_groupbox = QGroupBox('Performance Meta')
        self.performance_meta_groupbox.setLayout(self.performance_hbox)
        # NOTE: Removed Performance UI display for now.  
        # self.form_grid.addWidget(self.performance_meta_groupbox, 1, 0)
        # self.setupPerformanceUI()

        row = 1
        col = 0
        for model, types in self.model_params.items():
            for t, params in types.items():
                groupbox = QGroupBox()
                groupbox.setTitle(model.split('.')[-1] + ' ' + t)
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


    def apply_changes(self):
        version = self.current_version.split('\\')[-1]
        if version == 'default':
            # print('Default version selected.  Returning...')
            return
        if self.is_dirty:
            filename = self.main_model_name + '.json'
            save_dir = os.path.join(self.version_item_combobox.currentData(),
                                    self.main_model_name)

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            save_file_path = os.path.join(save_dir,
                                          filename)

            if not os.path.isfile(save_file_path):
                # Get default file and load those values
                default_dir = os.path.join(
                    '.\\package\\data\\default_models\\default', self.main_model_name)
                default_path = os.path.join(
                    default_dir, self.main_model_name + '.json')
                with open(default_path, 'r') as infile:
                    full_default_params = json.load(infile)
                save_data = {
                    'model_base': self.params[0]['model_base'],
                    'model_module': self.params[0]['model_module'],
                    'model_class': self.main_model_name,
                    'question_number': self.version_item_combobox.currentData().split('\\')[-1],
                    'version': version,
                    'tuned': False,
                    'params': {}
                }
                save_data['params'] = full_default_params['params']
            else:
                with open(save_file_path, 'r') as infile:
                    save_data = json.load(infile)

            for param_type, params in self.updated_params.items():
                if(params):
                    for param, val in params.items():
                        save_data['params'][param_type][param] = val
            try:
                with open(save_file_path, 'w') as outfile:
                    json.dump(save_data, outfile, cls=CATEncoder, indent=2)
            except Exception as e: 
                self.logger.error('Error saving updated model parameters for {}.'.format(
                    self.main_model_name), exc_info=True)
                print('Exception {}'.format(e))
                tb = traceback.format_exc()
                print(tb)

        self.is_dirty = False
        return

        
    def setupUI(self, param_type, param_dict, form):
        '''
        Build UI elements using parameters dict of scikit models

            # Attributes:
                param_type: String, type of param to update
                param_dict: dict, dictionary of parameter/default values from model.
                default_params: dict, dictionary of default parameters defined by me.
        '''
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
            self.logger.error('Error generating {} dialog.', exc_info=True)
            print('Exception {} occured with key {} and value {}'.format(e, k, v))
            tb = traceback.format_exc()
            print(tb)


    def load_version_params(self, path):
        '''
        Loads parameters from the selected version for a specific question.  
        Resets parameters to default prior to loading.
        If default or None is selected, returns after reload.  

            # Attributes
                path: String, path to version parameters.  
        '''
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
            with open(os.path.join(path, self.main_model_name, filename), 'r') as f:
                model_data = json.load(f)
            model_class = model_data['model_class']
            for kind, params in model_data['params'].items():
                self.set_input_params(params)

            self.is_dirty = False
        except FileNotFoundError as fnfe:
            pass
        except Exception as e:
            self.logger.error(f'Error updating {model} parameters', exc_info=True)
            print('Exception {}'.format(e))
            tb = traceback.format_exc()
            print(tb)


    @pyqtSlot(bool)
    def check_for_default(self, force_reload=False):
        '''
        Checks for the existance of a default value file.  If none found,
        one is created.
        '''
        default_dir = os.path.join(
            '.\\package\\data\\default_models\\default', self.main_model_name)
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)

        default_path = os.path.join(
            default_dir, self.main_model_name + '.json')

        if not os.path.isfile(default_path) or force_reload:
            self.logger.info(f'{self.main_model_name} building default parameter spec files.  force_reload = {force_reload}')
            save_data = {
                'model_base': self.params[0]['model_base'],
                'model_module': self.params[0]['model_module'],
                'model_class': self.main_model_name,
                'question_number': 'default',
                'version': 'default',
                'meta' :{
                    'training_meta': {
                        'last_train_date' : None,
                        'train_eval_score' : None,
                        'checksum' : None
                    },
                    'tuning_meta' : {
                        'last_tune_date': None,
                        'n_iter' : None,
                        'tuning_duration' : None,
                        'tune_eval_score' : None

                    }
                },
                'params': {}
            }
            for model, types in self.model_params.items():
                # print('check_for_defaults data:')
                # print(f'{model}')
                # print(types)
                for t, params in types.items():
                    # True if model spec has more than one category of parameters.  Only TF models at this point.
                    if not model in save_data['params'].keys():
                        save_data['params'][model] = {}
                    for param_name, data in params.items():
                        save_data['params'][model][param_name] = data['default']
            try:
                with open(default_path, 'w') as outfile:
                    json.dump(save_data, outfile, indent=2, cls=CATEncoder)
            except Exception as e:
                self.logger.error('Error saving updated model parameters for {}.'.format(
                    self.main_model_name), exc_info=True)
                print('Exception {}'.format(e))
                tb = traceback.format_exc()
                print(tb)

