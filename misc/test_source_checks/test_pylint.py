PYLINTRC_OPTIONS = []
PYLINTRC_FILE = 'misc/test_source_checks/pylintrc'
PYLINTRC_OPTIONS += ['--rcfile=' + PYLINTRC_FILE]


def test_pylint():
    import pylint.lint
    try:
        pylint.lint.Run(PYLINTRC_OPTIONS + ['lcode'])
    except SystemExit as se:
        if se.code:
            raise RuntimeError('PylintError')
