import os

FLAKE8_OPTIONS = '--exclude=*.preprocessed.pyx'
FLAKE8_TARGETS = 'lcode misc'

FLAKE8_COMMON = FLAKE8_OPTIONS + ' ' + FLAKE8_TARGETS
FLAKE8_CYTHON_IGNORES = ('E999', 'E225', 'E112', 'E402', 'E741')
FLAKE8_CYTHON = ('--filename="*.pyx,*.pxd" ' +
                 '--ignore=' + ','.join(FLAKE8_CYTHON_IGNORES) + ' ' +
                 FLAKE8_COMMON)
FLAKE8_PYTHON_IGNORES = ('N802', 'N803', 'N806', 'E741')
FLAKE8_PYTHON = ('--ignore=' + ','.join(FLAKE8_PYTHON_IGNORES) + ' ' +
                 FLAKE8_COMMON)


def test_flake8_python():
    assert os.system('flake8 ' + FLAKE8_PYTHON) == 0


def test_flake8_cython():
    assert os.system('flake8 ' + FLAKE8_CYTHON) == 0
