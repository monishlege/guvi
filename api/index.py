import os
import sys

# Ensure repository root is on sys.path so "main" and other modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from main import app
