'''
CATTrain module contains the all functionality associated with the training
 and tuning of machine-learning models.

@author pjtinker
'''

import sys
import importlib
import traceback
import argparse
import logging
import json
import os
import shutil
import errno

import pandas as pd

from PyQt5.QtCore import (Qt, pyqtSlot, pyqtSignal)
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QDialog, QHeaderView, QAction,
                               QMainWindow, QSizePolicy, QProgressBar, QWidget,
                               QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
                               QLabel, QDialogButtonBox, QMessageBox, QComboBox, QPushButton)

from package.train.TrainWidget import TrainWidget
from package.utils.catutils import exceptionWarning
from package.utils.config import CONFIG


VERSION_BASE_DIR = CONFIG.get('PATHS', 'BaseVersionDirectory')
# DEFAULT_QUESTION_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q6',
#                     'Q7', 'Q9', 'Q11', 'Q14', 'Q15'] 
DEFAULT_QUESTION_LABELS = CONFIG.get('VARIABLES', 'DefaultQuestionLabels').split(',')
class CatTrain(QMainWindow):
    """ The central widget for the training component of CATScore
        Most of the functionality is contained in this class
    """
    def __init__(self, parent=None):
        super(CatTrain, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.title = 'CAT Train'
        self.setWindowTitle(self.title)
        geometry = QApplication.desktop().availableGeometry(self)
        parent_left = self.parent().geometry().left()
        parent_top = self.parent().geometry().top()
        self.setGeometry(parent_left, parent_top, geometry.width() * 0.45, geometry.height() * 0.82)
        self.statusBar().showMessage('Cut me, Mick!')
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)
        
        self.file_menu = self.menuBar().addMenu('&File')
        self.reload_all_modules_action = QAction('Reload system modules', self)
        self.file_menu.addAction(self.reload_all_modules_action)
        self.reload_all_modules_action.triggered.connect(
            lambda: self.reload_all_modules()
        )
        self.print_all_modules_action = QAction("Print system modules", self)
        self.file_menu.addAction(self.print_all_modules_action)
        self.print_all_modules_action.triggered.connect(
            lambda: self.print_all_modules()
        )

        self.version_menu = self.menuBar().addMenu('&Version')
    
        self.version_widget = CreateVersionWidget(self)
        self.create_version_action = QAction('Create New Version', self)
        self.version_menu.addAction(self.create_version_action)
        self.create_version_action.triggered.connect(
            lambda: self.open_create_version_dialog(self.version_widget)
        )
        self.version_copy_widget = CopyVersionWidget(self)
        self.copy_version_action = QAction('Copy Version', self)
        self.version_menu.addAction(self.copy_version_action)
        self.copy_version_action.triggered.connect(
            lambda: self.open_copy_version_dialog(self.version_copy_widget)
        )
    
        self.statusBar().addPermanentWidget(self.progressBar)
        self.train_widget = TrainWidget(self)
        self.version_widget.version_created.connect(self.train_widget.model_widget.add_new_version)
        self.version_copy_widget.version_copied.connect(self.train_widget.model_widget.add_new_version)
        self.train_widget.data_loader.update_progressbar.connect(self.update_progress_bar)
        self.train_widget.model_widget.update_progressbar.connect(self.update_progress_bar)
        self.setCentralWidget(self.train_widget)

    def closeEvent(self, event):
        pass

    @pyqtSlot(int, bool)
    def update_progress_bar(self, val, pulse):
        if pulse:
            self.progressBar.setRange(0,0)
        else:
            self.progressBar.setRange(0, 1)

        self.progressBar.setValue(val)

    def open_create_version_dialog(self, dialog):
        dialog.create_version()

    def open_copy_version_dialog(self, dialog):
        dialog.copy_version()


    def reload_all_modules(self):
        try:
            for module in sys.modules.values():
                importlib.reload(module)
        except NotImplementedError as nie:
            self.logger.debug(
                "CatTrain.reload_all_modules:", exc_info=True)
            tb = traceback.format_exc()
            print(tb)
        except Exception as e:
            self.logger.error(
                "CatTrain.reload_all_modules:", exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    def print_all_modules(self):
        try:
            for module in sys.modules.values():
                print(module)
        except Exception as e:
            self.logger.error(
                "CatTrain.print_all_modules:", exc_info=True)
            tb = traceback.format_exc()
            print(tb)

class CopyVersionWidget(QDialog):
    '''
    Fully copy an existing version, spec files and models.
    '''
    version_copied = pyqtSignal(str)
    def __init__(self, parent=None):
        super(CopyVersionWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.setWindowTitle('Copy CAT Version')
        self.version_name = None
        self.main_layout = QVBoxLayout()
        self.version_form = QFormLayout()

        self.main_layout.addLayout(self.version_form)
        self.setupUI()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("version_copy_warning")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.setWindowTitle("Version Copy Warning")
        self.buttonBox.rejected.connect(self.reject)

        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)

    def _update_version_name(self, value):
        self.version_name = value
           

    def _version_check(self, version):
        """
        Checks that user has both supplied a version name and that, if supplied, 
        the name is unique.

            # Arguments
                version(String): version name supplied by user

            # Returns
                bool: True if version is supplied and unique, else False
        """
        if version == '':
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        v = os.path.join(VERSION_BASE_DIR, version)
        if os.path.exists(v):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Version {} already exists!".format(version))
            msg_box.setWindowTitle('Version Warning')
            msg_box.exec_()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        return True


    def copy_version(self):
        """
        Copy version specified by the user.
        """
        if(self.exec_() == QDialog.Accepted):
            src_dir = self.version_selection.currentData()
            dest_dir = os.path.join(VERSION_BASE_DIR, self.version_name_input.text())
            try:
                shutil.copytree(src_dir, dest_dir)
                self.version_copied.emit(dest_dir)
            except OSError as e:
                if e.errno == errno.ENOTDIR:
                    shutil.copy(src_dir, dest_dir)
                else:
                    exceptionWarning(f'Unable to copy {src_dir} to {dest_dir}', title='Copy version exception')
                    tb = traceback.format_exc()
                    print(tb)
            except Exception as e:
                exceptionWarning('Error occured when copying version.', e, title='Copy version exception')
            finally:
                self.version_name = None
                self.version_selection.addItem(self.version_name_input.text(), dest_dir)


    def setupUI(self):
        self.version_name_label = QLabel("Version name: ")
        self.version_name_input = QLineEdit(objectName='version_name')
        self.version_name_input.textChanged.connect(
            lambda state, y=self.version_name_input:
                self._update_version_name(
                    (y.text() if self._version_check(y.text()) else None)
                )
        )
        self.version_form.addRow(self.version_name_label, self.version_name_input)

        self.version_selection_label = QLabel("Select version: ")
        self.version_selection = QComboBox(objectName='version_select')

        available_versions = os.listdir(VERSION_BASE_DIR)
        for version in available_versions:
            v_path = os.path.join(VERSION_BASE_DIR, version)
            if os.path.isdir(v_path):
                self.version_selection.addItem(version, v_path)
        # self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection:
        #                                                    self._update_version_name(
        #                                                        y.currentData())
        #                                                    )
        self.version_form.addRow(self.version_selection_label, self.version_selection)

class CreateVersionWidget(QDialog):
    """
    Create version dialog used when generating a new CAT version.
    Allows user to input the expected question labels which will
    become the directory structure for storing models and parameters.
    """
    version_created = pyqtSignal(str)
    def __init__(self, parent=None):
        super(CreateVersionWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.setWindowTitle("New CAT Version")
        self.version_name = None
        self.question_labels = {}
        self.input_widgets = {}
        self.main_layout = QVBoxLayout()
        self.version_groupbox = QGroupBox("Create new version")
        self.version_form = QFormLayout()

        self.version_groupbox.setLayout(self.version_form)
        self.main_layout.addWidget(self.version_groupbox)
        self.setupUI()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("version_warning")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.setWindowTitle("Version Warning")
        self.buttonBox.rejected.connect(self.reject)

        self.main_layout.addWidget(self.buttonBox)
        self.setLayout(self.main_layout)

    def _verify_params(self):
        """
        Checks that each input field has some value and that the version name
        has been specified.

            # Returns: None
        """
        for label, widget in self.input_widgets.items():
            if widget.text() == '':
                self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
                return False
        if self.version_name:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
            return True

    def _update_version_name(self, value):
        self.version_name = value

    def _verify_unique_params(self, key, value):
        """
        Checks that field name is unique.  This is necessary as field values
        are used for version directory and data structure.

            # Arguments
                key(String): dict key for appropriate input widget
                value(String): field name.  Must be unique per version.
        """
        if value.lower() in [x.lower() for x in self.question_labels.values()]:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            exceptionWarning('Field names must be unique!')
            return
        try:
            self._verify_params()
        except Exception as e:
           exceptionWarning('Error updating version params.', e, title="Update version warning")
           
    def _version_check(self, version):
        """
        Checks that user has both supplied a version name and that, if supplied, 
        the name is unique.

            # Arguments
                version(String): version name supplied by user

            # Returns
                bool: True if version is supplied and unique, else False
        """
        if version == '':
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        v = os.path.join(VERSION_BASE_DIR, version)
        if os.path.exists(v):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Version {} already exists!".format(version))
            msg_box.setWindowTitle('Version Warning')
            msg_box.exec_()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        self._verify_params()
        return True

    def _update_fields(self, state):
        """
        Appends or removes an input widget for version field labels.

            # Arguments
                state(bool): If True, add a field, else remove the last field

            # Returns
                None
        """
        current_row_idx = self.version_form.rowCount()
        if state:
            label = QLabel('Field ' + str(current_row_idx) + ':')
            q_input = QLineEdit(objectName=str(current_row_idx - 1))
            q_input.textChanged.connect(
                lambda state, x=current_row_idx-1, y=q_input:
                    self._verify_unique_params(
                        x, 
                        (None if y.text() == '' else y.text())
                    )
            )
            self.version_form.insertRow(current_row_idx, label, q_input)
            self.input_widgets[str(current_row_idx - 1)] = q_input
            q_input.setFocus()
        else:
            if current_row_idx == 1:
                return
            item = self.input_widgets[str(current_row_idx - 2)]
            try:
                del self.input_widgets[item.objectName()]
                self.version_form.removeRow(current_row_idx - 1)
                self.version_form.update()
            except Exception as e:
                exceptionWarning('Error updating version params.', e, title="Update version warning")
        self._verify_params()

    def _generate_fields(self):
        """
        Generate fields based on default version scheme.
        """
        for idx, q_label in enumerate(DEFAULT_QUESTION_LABELS):
            self.question_labels[idx] = q_label
            label = QLabel('Field ' + str(idx+ 1) + ':')
            q_input = QLineEdit(objectName=str(idx))
            q_input.setText(q_label)
            q_input.textChanged.connect(
                lambda state, x=idx, y=self.version_name_input:
                    self._verify_unique_params(
                        x, 
                        (None if y.text() == '' else y.text())
                    )
            )
            self.version_form.addRow(label, q_input)
            self.input_widgets[str(idx)] = q_input 
        self._verify_params()

    def setupUI(self):
        self.version_name_label = QLabel("Version name: ")
        self.version_name_input = QLineEdit(objectName='version_name')
        self.version_name_input.textChanged.connect(
            lambda state, y=self.version_name_input:
                self._update_version_name(
                    (y.text() if self._version_check(y.text()) else None)
                )
        )
        self.version_form.addRow(self.version_name_label, self.version_name_input)
        self._generate_fields()
        self.new_question_row_btn = QPushButton('Add field', self)
        self.new_question_row_btn.clicked.connect(lambda: self._update_fields(True))
        self.remove_question_row_btn = QPushButton('Remove field', self)
        self.remove_question_row_btn.clicked.connect(lambda: self._update_fields(False))
        
        self.button_hbox = QHBoxLayout()
 
        self.button_hbox.addWidget(self.new_question_row_btn)
        self.button_hbox.addWidget(self.remove_question_row_btn)
        self.main_layout.addLayout(self.button_hbox)

    def create_version(self):
        """
        Create the new version specified by the user.
        """
        if(self.exec_() == QDialog.Accepted):
            v_dir = os.path.join(VERSION_BASE_DIR, self.version_name)
            try:
                if not os.path.exists(v_dir):
                    os.makedirs(v_dir)
                for k,v in self.input_widgets.items():
                    sub_dir = os.path.join(v_dir, v.text())
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir)
                self.version_created.emit(v_dir)
            except Exception as e:
                exceptionWarning('Error occured when creating new version.', e, title='Create version exception')
            finally:
                self.question_labels = {}
                self.version_name = None
                self.version_name_input.setText('')
                for k,v in self.input_widgets.items():
                    self.version_form.removeRow(v)
                self.input_widgets = {}
                self.version_form.update()
                self._generate_fields()



if __name__ == "__main__":
    import sys
    # Qt Application
    # app = QApplication(sys.argv)
    # # app.setStyle('Fusion')
    # window = CatTrain()
    # window.show()
    # sys.exit(app.exec_())