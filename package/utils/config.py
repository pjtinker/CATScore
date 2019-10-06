import datetime
import json
import logging
import os
import sys

try:
    import configparser
except ImportError:
    import ConfigParser as configparser 

DEFAULT_CONFIG_FILE = '.\\package\\utils\\default.ini'

def get_config_file():
    return os.environ.get('CONFIG_FILE', DEFAULT_CONFIG_FILE)

CONFIG_FILE = get_config_file()

def create_config(config_file=None):
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.read(config_file or CONFIG_FILE)
    return parser

CONFIG = create_config()