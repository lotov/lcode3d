import os

FLAKE8_OPTIONS = '--exclude=*.preprocessed.pyx'
FLAKE8_TARGETS = 'lcode misc'

FLAKE8_PYTHON = FLAKE8_OPTIONS + ' ' + FLAKE8_TARGETS
FLAKE8_CYTHON_IGNORES = ('E999', 'E225', 'E112', 'E402')
FLAKE8_CYTHON = ('--filename="*.pyx,*.pxd" ' +
                 '--ignore=' + ','.join(FLAKE8_CYTHON_IGNORES) + ' ' +
                 FLAKE8_PYTHON)


def test_flake8_python():
    assert os.system('flake8 ' + FLAKE8_PYTHON) == 0


def test_flake8_cython():
    assert os.system('flake8 ' + FLAKE8_CYTHON) == 0
