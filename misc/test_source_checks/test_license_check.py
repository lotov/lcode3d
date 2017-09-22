#!/usr/bin/python3
"""
Checks source tree for files that lack a license header.
"""

import os
import re
import sys


DEFAULT_LICENSE_NOTICE_PATH = 'misc/test_source_checks/LCODE_AGPL_NOTICE'
MAX_EXTRA_BYTES = 2 * 80  # Allow several lines before the notice

exclusions = (
    r'\.c$',
    r'\.calltree$',
    r'\.h5$',
    r'\.mp4$',
    r'\.o$',
    r'\.png$',
    r'\.prof$',
    r'\.pyc$',
    r'\.so$',
    r'^./.*\.egg-info/',
    r'^./.mailmap$',
    r'^./AUTHORS$',
    r'^./Bz.*\.npy$',
    r'^./ChangeLog$',
    r'^./Ez.*\.npy$',
    r'^./LICENSE\.txt$',
    r'^./README.md$',
    r'^./\.eggs/',
    r'^./\.git/',
    r'^./\.gitignore$',
    r'^./\.gitlab-ci\.yml$',
    r'^./__main__\.py$',
    r'^./build/',
    r'^./configs/.*\.py$',
    r'^./dist/',
    r'^./dump/',
    r'^./lcode/.*\.html$',
    r'^./lcode/default_config.py$',
    r'^./lcode\.spec$',
    r'^./misc/c_plasma_solver/compare_dumps\.py$',
    r'^./misc/c_plasma_solver/plasma_solver$',
    r'^./misc/dockerfiles/',
    r'^./misc/test_against_c_plasma_solver\.py$',
    r'^./misc/test_against_higher_precision/Ez_00_high\.npy$',
    r'^./misc/test_source_checks/',
    r'^./requirements.txt$',
    r'^./result_.*\.dat$',
    r'^./setup.cfg$',
    r'^./setup.py$',
    r'^./setup_hooks.py$',
    r'^./transverse.*/',
)


def test_license_notices(license_notice_filename=DEFAULT_LICENSE_NOTICE_PATH):
    with open(license_notice_filename, encoding='utf-8') as f:
        license_notice = f.read()

    error = False

    for root, _, files in os.walk('.'):
        for fname in files:
            path = os.path.join(root, fname)
            if any(re.search(exclusion, path) for exclusion in exclusions):
                continue

            with open(path, encoding='utf-8') as f:
                try:
                    top = f.read(len(license_notice) + MAX_EXTRA_BYTES)
                except UnicodeDecodeError:
                    raise RuntimeError('Cannot decode ' + path)

                if not top:
                    continue  # Allow no license notice in empty files

                if license_notice not in top:
                    print('License notice missing/broken in', path)
                    error = True

    if error:
        raise RuntimeError('License notices are not OK')
    else:
        print('License notices are OK')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], '<LICENSE TEMPLATE FILE>')
        sys.exit(1)
    license_notice_filename = sys.argv[1]

    test_license_notices(license_notice_filename)
