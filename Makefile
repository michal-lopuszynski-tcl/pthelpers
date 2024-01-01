MODULE_NAME=pthelpers

PY_DIRS=src/pthelpers tests setup.py

PY_MYPY_FLAKE8=src/pthelpers tests setup.py

FILES_TO_CLEAN=pthelpers.egg-info dist

include Makefile.inc
